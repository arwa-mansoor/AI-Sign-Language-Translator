"""On-screen interface of the live app (src/app.py).

Layout: the mirrored camera view on the left is kept clear for the signer —
only the tracking (a neon skeleton with fingertip trails), viewfinder corners
and a LIVE badge at its edges. Everything else is in the side panel: a header,
the status, the recognised word in English and Urdu with a confidence ring (and
a particle burst when a sign is recognised), the recent words, and the signs
the model knows (English + Urdu), ticking off those already recognised; for
large vocabularies the list shows the model's top-3 guesses instead.

Text is drawn with Pillow: cv2.putText has only Hershey fonts and no Urdu
glyphs or right-to-left letter joining. English uses Segoe UI, Urdu the
Windows "Urdu Typesetting" Nastaliq font (falling back to Noto / Arial).

Keys: d  sign list on/off      s  skeleton on/off     f  full screen
      c  clear recent words     q / ESC  quit
"""

import dataclasses
import functools
import math
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

WINDOW = "PSL Sign Language Translator"
TITLE, SUBTITLE = "PSL Translator", "Pakistan Sign Language  ·  AI"
DISPLAY_HEIGHT = 600        # camera view height on screen; f = full screen
PANEL_WIDTH = 560           # side panel width at a 720 px view height
DEMO_PANEL_MAX_WORDS = 30   # larger vocabularies get a top-3 panel instead
CAPTION_SECONDS = 8.0       # how long a recognised word stays in the card
RECOGNISED_SECONDS = 1.5    # "Recognised" status / green frame after a confirmation
POP_SECONDS = 0.35          # new-word pop animation
RING_SECONDS = 0.6          # confidence ring fill animation
TRAIL_FRAMES = 14           # fingertip trail length

FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
FONTS = {
    "regular": ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"),
    "bold": ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"),
    "urdu": ("UrdType.ttf", "NotoNastaliqUrdu-Regular.ttf", "NotoSansArabic-Regular.ttf",
             "segoeui.ttf", "arial.ttf"),
}

# BGR palette: deep navy -> purple background, cyan / magenta / green accents
WHITE, GREY, DIM = (248, 246, 244), (195, 180, 172), (135, 118, 110)
BG_TOP, BG_BOTTOM = (42, 22, 14), (52, 16, 30)
CARD_TOP, CARD_BOTTOM = (70, 40, 30), (78, 30, 46)
CHIP = (74, 46, 38)
CYAN, MAGENTA = (235, 205, 40), (215, 80, 205)
GREEN, AMBER, RED = (125, 225, 90), (40, 185, 255), (70, 70, 240)
URDU_GOLD = (110, 212, 255)
POSE_LINE = (240, 220, 230)
STATUS_COLOURS = {"idle": DIM, "signing": CYAN, "hold": AMBER}


@dataclasses.dataclass(frozen=True)
class Word:
    label: str  # dataset label, e.g. pk-hfad-1_airplane
    en: str     # first English token
    ur: str     # first Urdu token


# ----------------------------------------------------------------- text ----

def _load_font(names, size: int):
    for name in names:
        for candidate in (FONT_DIR / name, name):
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


class TextPainter:
    """Cached Pillow text sprites blended onto BGR images."""

    def __init__(self):
        self._fonts: dict = {}
        self._sprites: dict = {}
        self.rtl = {"direction": "rtl", "language": "ur"} if features.check("raqm") else {}
        if not self.rtl:
            print("note: Pillow lacks raqm text shaping; Urdu letters may not join correctly")

    def _font(self, style: str, size: int):
        key = (style, size)
        if key not in self._fonts:
            self._fonts[key] = _load_font(FONTS[style], size)
        return self._fonts[key]

    def sprite(self, text: str, style: str, size: int, colour) -> np.ndarray:
        """BGRA image of the text; colour is one BGR colour or a (left, right)
        pair for a horizontal gradient. Latin text keeps the font's full line
        height (stable vertical alignment); Urdu is cropped to its ink (Nastaliq
        line metrics are very tall)."""
        gradient = isinstance(colour[0], (tuple, list))
        key = (text, style, size, tuple(map(tuple, colour)) if gradient else tuple(colour))
        if key in self._sprites:
            return self._sprites[key]
        font = self._font(style, size)
        layout = self.rtl if style == "urdu" else {}
        pad = 4
        left, top, right, bottom = font.getbbox(text, **layout)
        if style != "urdu":
            ascent, descent = font.getmetrics()
            top, bottom = 0, ascent + descent
        img = Image.new("RGBA", (max(right - left, 1) + 2 * pad, bottom - top + 2 * pad))
        fill = (255, 255, 255, 255) if gradient else (*colour[::-1], 255)
        ImageDraw.Draw(img).text((pad - left, pad - top), text, font=font, fill=fill, **layout)
        sprite = np.asarray(img)[:, :, [2, 1, 0, 3]].copy()  # RGBA -> BGRA
        if gradient:
            t = np.linspace(0.0, 1.0, sprite.shape[1], dtype=np.float32)[None, :, None]
            ramp = np.array(colour[0], np.float32) * (1 - t) + np.array(colour[1], np.float32) * t
            sprite[:, :, :3] = np.broadcast_to(ramp, sprite[:, :, :3].shape).astype(np.uint8)
        self._sprites[key] = sprite
        return sprite

    def size(self, text: str, style: str, size: int) -> tuple[int, int]:
        h, w = self.sprite(text, style, size, WHITE).shape[:2]
        return w, h

    def draw(self, img, text: str, style: str, size: int, colour, x: int, y: int,
             anchor: str = "lm", scale: float = 1.0) -> tuple[int, int]:
        """Draw text; anchor = horizontal (l/m/r) + vertical (t/m/b) reference of (x, y)."""
        if not text:
            return 0, 0
        sprite = self.sprite(text, style, size, colour)
        if scale != 1.0:
            sprite = cv2.resize(sprite, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h, w = sprite.shape[:2]
        x0 = x - {"l": 0, "m": w // 2, "r": w}[anchor[0]]
        y0 = y - {"t": 0, "m": h // 2, "b": h}[anchor[1]]
        blend(img, sprite, x0, y0)
        return w, h


# --------------------------------------------------------------- shapes ----

def blend(img: np.ndarray, sprite_bgra: np.ndarray, x0: int, y0: int) -> None:
    """Alpha-blend a BGRA sprite onto img at (x0, y0), clipped to the image."""
    h, w = sprite_bgra.shape[:2]
    ix0, iy0 = max(x0, 0), max(y0, 0)
    ix1, iy1 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if ix1 <= ix0 or iy1 <= iy0:
        return
    part = sprite_bgra[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    alpha = part[:, :, 3].astype(np.float32) * (1.0 / 255.0)
    roi = img[iy0:iy1, ix0:ix1]
    roi[:] = cv2.blendLinear(np.ascontiguousarray(part[:, :, :3]), np.ascontiguousarray(roi),
                             alpha, 1.0 - alpha)


def rounded_rect(img, x0: int, y0: int, x1: int, y1: int, colour, alpha: float = 1.0,
                 radius: int = 12) -> None:
    """Filled, anti-aliased rounded rectangle; colour may be a (top, bottom)
    pair for a vertical gradient. Optionally translucent."""
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), img.shape[1]), min(int(y1), img.shape[0])
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return
    colour = (tuple(map(int, colour[0])), tuple(map(int, colour[1]))) \
        if isinstance(colour[0], (tuple, list)) else tuple(map(int, colour))
    fill, a, rest = _rounded_layer(w, h, max(0, min(radius, w // 2, h // 2)), colour,
                                   round(alpha, 2))
    roi = img[y0:y1, x0:x1]
    roi[:] = cv2.blendLinear(fill, np.ascontiguousarray(roi), a, rest)


@functools.lru_cache(maxsize=512)
def _rounded_layer(w: int, h: int, r: int, colour: tuple, alpha: float):
    """(fill, weight, 1 - weight) for blending a rounded rectangle, cached."""
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    a = mask.astype(np.float32) * (alpha / 255.0)
    return _fill(w, h, colour), a, 1.0 - a


def _fill(w: int, h: int, colour) -> np.ndarray:
    """Solid colour, or a vertical gradient for a (top, bottom) colour pair."""
    if isinstance(colour[0], tuple):
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        grad = np.array(colour[0], np.float32) * (1 - t) + np.array(colour[1], np.float32) * t
        return np.ascontiguousarray(np.broadcast_to(grad, (h, w, 3)).astype(np.uint8))
    return np.full((h, w, 3), colour, np.uint8)


@functools.lru_cache(maxsize=8)
def _background(h: int, w: int) -> np.ndarray:
    return _fill(w, h, (BG_TOP, BG_BOTTOM))


def check_mark(img, cx: int, cy: int, size: int, colour, thickness: int = 2) -> None:
    pts = np.array([(cx - size, cy), (cx - size // 3, cy + size * 2 // 3),
                    (cx + size, cy - size * 2 // 3)], np.int32)
    cv2.polylines(img, [pts], False, colour, thickness, cv2.LINE_AA)


def mix(colour, other, t: float):
    """Blend two BGR colours (t=0 -> colour, t=1 -> other)."""
    return tuple(int(a * (1 - t) + b * t) for a, b in zip(colour, other))


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


# -------------------------------------------------------------- overlay ----

class Overlay:
    """Renders the app's screen and handles its keys."""

    def __init__(self, vocabulary: list[Word], show_en: bool = True, show_ur: bool = True,
                 height: int = DISPLAY_HEIGHT):
        self.text = TextPainter()
        self.vocabulary = vocabulary if len(vocabulary) <= DEMO_PANEL_MAX_WORDS else []
        self.show_en, self.show_ur = show_en, show_ur
        self.height = height
        self.k = height / 720.0  # all sizes below are designed for a 720 px view
        self.show_list = True    # d: the sign list / top-3 section of the panel
        self.skeleton = True
        self.fullscreen = False
        self.history: deque[Word] = deque(maxlen=6)
        self.recognised: set[str] = set()
        self.current: Word | None = None
        self.confidence = 0.0
        self.confirmed_at = -1e9
        self.trails = {"left": deque(maxlen=TRAIL_FRAMES), "right": deque(maxlen=TRAIL_FRAMES)}
        self.particles: list = []  # [x, y, vx, vy, born, colour, radius] in panel pixels
        self._burst_pending = False
        self._last_render = time.perf_counter()
        self._rng = np.random.default_rng()
        self._window_size = None
        self._list_cache = (None, None)  # (key, image): redrawn only when its content changes

    # ---- state updates from the app ----

    def confirm(self, word: Word, confidence: float) -> None:
        """A newly confirmed sign: card, pop, particle burst, history, tick."""
        self.current, self.confidence, self.confirmed_at = word, confidence, time.perf_counter()
        self.history.append(word)
        self.recognised.add(word.label)
        self._burst_pending = True

    def refresh(self, word: Word, confidence: float) -> None:
        """The confirmed sign is still being recognised (no new pop)."""
        if self.current == word:
            self.confidence = confidence

    # ---- window ----

    def open_window(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    def show(self, canvas: np.ndarray) -> None:
        size = (canvas.shape[1], canvas.shape[0])
        if size != self._window_size and not self.fullscreen:
            cv2.resizeWindow(WINDOW, *size)
            self._window_size = size
        cv2.imshow(WINDOW, canvas)

    def handle_key(self, key: int) -> str | None:
        """Returns "quit" for q / ESC; toggles everything else."""
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("d"):
            self.show_list = not self.show_list
        elif key == ord("s"):
            self.skeleton = not self.skeleton
        elif key == ord("c"):
            self.history.clear()
            self.recognised.clear()
            self.current = None
        elif key == ord("f"):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
            self._window_size = None
        return None

    # ---- camera view ----

    def fit(self, frame: np.ndarray) -> np.ndarray:
        """Scale the camera frame to the display height (tracking is drawn after)."""
        scale = self.height / frame.shape[0]
        return cv2.resize(frame, (round(frame.shape[1] * scale), self.height),
                          interpolation=cv2.INTER_LINEAR)

    def px(self, v: float) -> int:
        return max(1, round(v * self.k))

    def draw_tracking(self, view, result, pose_connections, hand_connections) -> None:
        """Neon skeleton (glow + bright core) and fingertip trails on the mirrored view."""
        h, w = view.shape[:2]
        # pose 0-10 (face) still feed the model but are not drawn: dots on the face look like a mask
        pose_connections = [(a, b) for a, b in pose_connections if a >= 11 and b >= 11]
        groups = (("pose", result.pose_landmarks, pose_connections, POSE_LINE, 23),
                  ("left", result.left_hand_landmarks, hand_connections, MAGENTA, 21),
                  ("right", result.right_hand_landmarks, hand_connections, CYAN, 21))
        points = {}
        for name, landmarks, _, _, keep in groups:
            if landmarks:  # hips and legs are off screen and unused by the model
                points[name] = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in landmarks[:keep]]
        for hand in ("left", "right"):
            self.trails[hand].append(points[hand][8] if hand in points else None)  # index tip
        if not self.skeleton:
            return
        t = max(1, round(h / 300))
        glow = view.copy()
        for name, _, connections, colour, _ in groups:
            for a, b in connections:
                if name in points and a < len(points[name]) and b < len(points[name]):
                    cv2.line(glow, points[name][a], points[name][b], colour, t * 6, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.35, view, 0.65, 0, dst=view)
        for hand, colour in (("left", MAGENTA), ("right", CYAN)):
            trail = list(self.trails[hand])
            jump = 0.12 * w  # a longer step between frames is a re-detection, not motion
            for i in range(1, len(trail)):
                if trail[i - 1] is not None and trail[i] is not None                         and math.dist(trail[i - 1], trail[i]) < jump:
                    age = i / len(trail)
                    cv2.line(view, trail[i - 1], trail[i], mix((40, 20, 20), colour, age),
                             max(1, round(t * 4 * age)), cv2.LINE_AA)
        for name, _, connections, colour, _ in groups:
            if name not in points:
                continue
            core = mix(colour, WHITE, 0.45)
            for a, b in connections:
                if a < len(points[name]) and b < len(points[name]):
                    cv2.line(view, points[name][a], points[name][b], core, t, cv2.LINE_AA)
            for p in points[name][11 if name == "pose" else 0:]:
                cv2.circle(view, p, t + 1, WHITE, -1, cv2.LINE_AA)

    def _decorate_view(self, view, status: str, since: float, now: float) -> None:
        """Viewfinder corners and a LIVE badge — only at the edges of the camera view."""
        h, w = view.shape[:2]
        recognised = since < RECOGNISED_SECONDS
        colour = GREEN if recognised else STATUS_COLOURS.get(status, DIM)
        inset, arm, t = self.px(16), self.px(46), self.px(4)
        for cx, cy, dx, dy in ((inset, inset, 1, 1), (w - inset, inset, -1, 1),
                               (inset, h - inset, 1, -1), (w - inset, h - inset, -1, -1)):
            cv2.line(view, (cx, cy), (cx + dx * arm, cy), colour, t, cv2.LINE_AA)
            cv2.line(view, (cx, cy), (cx, cy + dy * arm), colour, t, cv2.LINE_AA)
        if recognised:  # a green frame flashes around the view and fades
            fade = 1.0 - since / RECOGNISED_SECONDS
            thickness = max(1, round(self.px(6) * fade))
            cv2.rectangle(view, (0, 0), (w - 1, h - 1), GREEN, thickness, cv2.LINE_AA)
        # LIVE badge
        x, y = inset + self.px(16), inset + self.px(14)
        bw, bh = self.px(92), self.px(34)
        rounded_rect(view, x, y, x + bw, y + bh, (20, 12, 10), 0.6, bh // 2)
        pulse = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(now * 4.0))
        cv2.circle(view, (x + self.px(18), y + bh // 2), self.px(7), mix((30, 30, 60), RED, pulse),
                   -1, cv2.LINE_AA)
        self.text.draw(view, "LIVE", "bold", self.px(20), WHITE, x + self.px(32), y + bh // 2)

    # ---- the whole screen ----

    def render(self, view: np.ndarray, status: str, fps: float, top3=()) -> np.ndarray:
        """Camera view (left) + side panel with everything else (right)."""
        now = time.perf_counter()
        dt, self._last_render = min(now - self._last_render, 0.1), now
        since = now - self.confirmed_at
        self._decorate_view(view, status, since, now)

        H, W = view.shape[0], self.px(PANEL_WIDTH)
        panel = _background(H, W).copy()
        m = self.px(28)
        self._header(panel, m, fps)
        self._status(panel, status, since, now, m, self.px(104))
        card_top = self.px(132)
        card_bottom = card_top + self.px(184 if self.show_list else 290)
        self._detected(panel, status, since, now, m, card_top, card_bottom)
        self._particles(panel, dt, now, m, card_top, card_bottom)
        self._history(panel, m, card_bottom + self.px(14))
        if self.show_list:
            list_top, list_bottom = card_bottom + self.px(76), H - self.px(40)
            if self.vocabulary:
                self._vocabulary(panel, m, list_top, list_bottom)
            else:
                self._guesses(panel, m, list_top, top3)
        self.text.draw(panel, "d list   s skeleton   f full screen   c clear   q quit",
                       "regular", self.px(17), DIM, m, H - self.px(20))
        return np.concatenate([view, panel], axis=1)

    def _header(self, panel, m: int, fps: float) -> None:
        s = self.px(46)  # logo: gradient tile with the initials
        x, y = m, self.px(20)
        rounded_rect(panel, x, y, x + s, y + s, (CYAN, MAGENTA), 1.0, self.px(12))
        self.text.draw(panel, "PSL", "bold", self.px(17), WHITE, x + s // 2, y + s // 2,
                       anchor="mm")
        tx = x + s + self.px(14)
        self.text.draw(panel, TITLE, "bold", self.px(31), (WHITE, CYAN), tx, y + self.px(13))
        self.text.draw(panel, SUBTITLE, "regular", self.px(17), GREY, tx, y + self.px(37))
        self.text.draw(panel, f"{fps:.0f} fps", "regular", self.px(17), DIM,
                       panel.shape[1] - m, y + self.px(13), anchor="rm")
        cv2.line(panel, (m, self.px(80)), (panel.shape[1] - m, self.px(80)), (80, 50, 52), 1,
                 cv2.LINE_AA)

    def _status(self, panel, status: str, since: float, now: float, m: int, cy: int) -> None:
        if since < RECOGNISED_SECONDS:
            text, colour = "Recognised!", GREEN
        else:
            text = {"idle": "Ready - raise your hands", "signing": "Watching your sign...",
                    "hold": "Reading the sign..."}.get(status, status)
            colour = STATUS_COLOURS.get(status, DIM)
        r = self.px(9)
        if status == "signing" and since >= RECOGNISED_SECONDS:  # pulse while watching
            ring = self.px(9 + 7 * ((now * 1.5) % 1.0))
            cv2.circle(panel, (m + self.px(10), cy), ring, mix(colour, BG_TOP, (now * 1.5) % 1.0),
                       self.px(2), cv2.LINE_AA)
        cv2.circle(panel, (m + self.px(10), cy), r, colour, -1, cv2.LINE_AA)
        self.text.draw(panel, text, "bold", self.px(25), colour if colour != DIM else GREY,
                       m + self.px(32), cy)

    def _detected(self, panel, status: str, since: float, now: float, m: int,
                  y0: int, y1: int) -> None:
        """The recognised word: English on top, Urdu below, confidence ring bottom-left."""
        x0, x1 = m, panel.shape[1] - m
        showing = (self.current is not None and since < CAPTION_SECONDS
                   and not (status == "signing" and since > RECOGNISED_SECONDS))
        if showing and since < RECOGNISED_SECONDS:  # glowing rim fades after a confirmation
            g = self.px(5)
            rounded_rect(panel, x0 - g, y0 - g, x1 + g, y1 + g, GREEN,
                         0.85 * (1.0 - since / RECOGNISED_SECONDS), self.px(22))
        rounded_rect(panel, x0, y0, x1, y1, (CARD_TOP, CARD_BOTTOM), 1.0, self.px(18))
        if not showing:
            mid = (y0 + y1) // 2
            dots = "." * (1 + int(now * 2.5) % 3)
            first = f"Watching{dots}" if status in ("signing", "hold") else "Sign a word"
            self.text.draw(panel, first, "bold", self.px(36), GREY, (x0 + x1) // 2,
                           mid - self.px(18), anchor="mm")
            self.text.draw(panel, "lower your hands between signs", "regular", self.px(21), DIM,
                           (x0 + x1) // 2, mid + self.px(26), anchor="mm")
            return
        pop = 1.0 + 0.25 * max(0.0, 1.0 - since / POP_SECONDS) ** 2
        word, h = self.current, y1 - y0
        room = (x1 - x0) - self.px(56)
        if self.show_en:
            size = self.px(62 if self.show_ur else 96)
            w = self.text.size(word.en, "bold", size)[0]
            self.text.draw(panel, word.en, "bold", size, (WHITE, CYAN), x0 + self.px(28),
                           y0 + round(h * (0.25 if self.show_ur else 0.40)),
                           scale=pop * min(1.0, room / w))
        if self.show_ur and word.ur:
            size = self.px(72 if self.show_en else 130)
            w = self.text.size(word.ur, "urdu", size)[0]
            self.text.draw(panel, word.ur, "urdu", size, URDU_GOLD, x1 - self.px(28),
                           y0 + round(h * (0.60 if self.show_en else 0.45)), anchor="rm",
                           scale=pop * min(1.0, room / w))
        # confidence ring, filling up after a confirmation
        r, t = self.px(23), self.px(6)
        cx, cy = x0 + self.px(28) + r, y1 - self.px(14) - r
        shown = self.confidence * ease_out(since / RING_SECONDS)
        colour = GREEN if self.confidence >= 0.7 else AMBER
        cv2.circle(panel, (cx, cy), r, mix(CARD_BOTTOM, WHITE, 0.15), t, cv2.LINE_AA)
        cv2.ellipse(panel, (cx, cy), (r, r), -90, 0, 360 * shown, colour, t, cv2.LINE_AA)
        self.text.draw(panel, f"{round(shown * 100)}", "bold", self.px(17), WHITE, cx, cy,
                       anchor="mm")
        self.text.draw(panel, "% match", "regular", self.px(19), GREY, cx + r + self.px(12), cy)

    def _particles(self, panel, dt: float, now: float, m: int, y0: int, y1: int) -> None:
        """Confetti burst from the word card when a sign is recognised."""
        if self._burst_pending:
            self._burst_pending = False
            x0, x1 = m, panel.shape[1] - m
            colours = (CYAN, MAGENTA, GREEN, URDU_GOLD, WHITE)
            for _ in range(56):
                edge = self._rng.integers(4)  # top, bottom, left, right edge of the card
                u = self._rng.uniform()
                x, y, nx, ny = ((x0 + u * (x1 - x0), y0, 0, -1), (x0 + u * (x1 - x0), y1, 0, 1),
                                (x0, y0 + u * (y1 - y0), -1, 0), (x1, y0 + u * (y1 - y0), 1, 0))[edge]
                speed = self._rng.uniform(60, 260) * self.k
                spread = self._rng.uniform(-0.6, 0.6)
                self.particles.append([x, y, (nx + spread * ny) * speed, (ny + spread * nx) * speed,
                                       now, colours[self._rng.integers(len(colours))],
                                       self._rng.uniform(2.0, 4.5) * self.k])
        alive = []
        for p in self.particles:
            age = (now - p[4]) / 1.1
            if age >= 1.0:
                continue
            p[2] *= 0.97
            p[3] += 140 * self.k * dt  # light gravity
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            cv2.circle(panel, (int(p[0]), int(p[1])), max(1, round(p[6] * (1 - age * 0.5))),
                       mix(p[5], BG_BOTTOM, age), -1, cv2.LINE_AA)
            alive.append(p)
        self.particles = alive

    def _history(self, panel, m: int, y: int) -> None:
        """'Recent' label and a row of word chips, newest last (oldest dropped if too long)."""
        y0, y1 = y, y + self.px(46)
        label_w, _ = self.text.draw(panel, "Recent", "regular", self.px(20), DIM, m, (y0 + y1) // 2)
        x = m + label_w + self.px(12)
        if not self.history:
            return
        size, gap, pad = self.px(23), self.px(10), self.px(14)
        labels = [w.en if self.show_en else w.ur for w in self.history]
        style = "regular" if self.show_en else "urdu"
        widths = [self.text.size(t, style, size)[0] + 2 * pad for t in labels]
        while len(widths) > 1 and sum(widths) + gap * (len(widths) - 1) > panel.shape[1] - m - x:
            labels, widths = labels[1:], widths[1:]
        for i, (text, w) in enumerate(zip(labels, widths)):
            newest = i == len(labels) - 1
            rounded_rect(panel, x, y0, x + w, y1, (GREEN, mix(GREEN, CYAN, 0.6)) if newest else CHIP,
                         1.0, radius=(y1 - y0) // 2)
            self.text.draw(panel, text, style, size, BG_TOP if newest else WHITE,
                           x + w // 2, (y0 + y1) // 2, anchor="mm")
            x += w + gap

    def _vocabulary(self, panel, m: int, top: int, bottom: int) -> None:
        """Two-column list of the signs the model knows; drawn once per change, then copied."""
        current = self.current.label if self.current is not None else None
        key = (current, frozenset(self.recognised), panel.shape[1], top, bottom)
        if self._list_cache[0] != key:
            block = panel[top:bottom].copy()
            self._draw_vocabulary(block, m, current)
            self._list_cache = (key, block)
        panel[top:bottom] = self._list_cache[1]

    def _draw_vocabulary(self, block, m: int, current: str | None) -> None:
        n = len(self.vocabulary)
        done_n = len(self.recognised & {w.label for w in self.vocabulary})
        self.text.draw(block, "Signs I know", "bold", self.px(25), WHITE, m, self.px(16))
        self.text.draw(block, f"{done_n} / {n}", "bold", self.px(22), GREEN if done_n else GREY,
                       block.shape[1] - m, self.px(16), anchor="rm")
        bar_y, bar_w = self.px(36), block.shape[1] - 2 * m  # progress bar
        rounded_rect(block, m, bar_y, m + bar_w, bar_y + self.px(6), CHIP, 1.0, self.px(3))
        if done_n:
            rounded_rect(block, m, bar_y, m + max(self.px(6), round(bar_w * done_n / n)),
                         bar_y + self.px(6), (CYAN, GREEN), 1.0, self.px(3))
        rows = (n + 1) // 2
        top = self.px(50)
        row = min(self.px(50), (block.shape[0] - top) // max(rows, 1))
        col_w = (block.shape[1] - 2 * m - self.px(12)) // 2
        for i, word in enumerate(self.vocabulary):
            x0 = m + (i // rows) * (col_w + self.px(12))
            y0 = top + (i % rows) * row
            cy = y0 + row // 2
            is_current = word.label == current
            if is_current:
                rounded_rect(block, x0 - self.px(6), y0 + 1, x0 + col_w + self.px(6), y0 + row - 1,
                             mix(CHIP, CYAN, 0.35), 1.0, radius=self.px(8))
            done = word.label in self.recognised
            if done:
                check_mark(block, x0 + self.px(9), cy, max(4, row // 5), GREEN, max(2, self.px(3)))
            else:
                cv2.circle(block, (x0 + self.px(9), cy), max(2, row // 10), DIM, -1, cv2.LINE_AA)
            if self.show_en:
                self.text.draw(block, word.en, "regular", max(10, round(row * 0.62)),
                               WHITE if done or is_current else GREY, x0 + self.px(28), cy)
            if self.show_ur and word.ur:
                self.text.draw(block, word.ur, "urdu", max(12, round(row * 0.82)), URDU_GOLD,
                               x0 + col_w, cy, anchor="rm")

    def _guesses(self, panel, m: int, top: int, top3) -> None:
        """Large vocabularies: the model's top-3 guesses for the latest window."""
        self.text.draw(panel, "Top guesses", "bold", self.px(25), WHITE, m, top + self.px(16))
        y, bw = top + self.px(62), panel.shape[1] - 2 * m
        for word, p in top3:
            self.text.draw(panel, word.en, "regular", self.px(24), WHITE, m, y)
            if self.show_ur and word.ur:
                self.text.draw(panel, word.ur, "urdu", self.px(30), URDU_GOLD,
                               panel.shape[1] - m, y, anchor="rm")
            rounded_rect(panel, m, y + self.px(22), m + bw, y + self.px(31), CHIP, 1.0, 4)
            rounded_rect(panel, m, y + self.px(22), m + max(4, round(bw * p)), y + self.px(31),
                         (CYAN, GREEN) if p >= 0.7 else AMBER, 1.0, 4)
            y += self.px(70)
