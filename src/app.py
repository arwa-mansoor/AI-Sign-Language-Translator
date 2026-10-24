"""Phase 5 — real-time webcam sign-language recognition app.

Captures webcam frames with OpenCV, extracts pose + hand landmarks via
MediaPipe in the same layout as the training CSVs, and keeps the last few
seconds of timestamped frames.  Periodically it resamples them to the
dataset's 25 fps (so signing speed in frames matches training whatever the
camera / landmarker throughput), cuts out the most recent sign (the last
hand-active span), runs it through the exact training feature pipeline
(data.dataset.preprocess) and the model, and overlays the predicted English +
Urdu text on the live video feed.  When a sign is confirmed it is spoken in
English (system voice) and Urdu (clips from src/make_urdu_audio.py).

Uses the MediaPipe Tasks HolisticLandmarker (mediapipe >= 1.0; the older
mp.solutions.holistic API no longer exists).  The task bundle is downloaded
once from Google's model CDN into models/mediapipe/ on first run.

Run:
    python src/app.py                       # newest checkpoint, both languages + speech
    python src/app.py --run bilstm          # specific run
    python src/app.py --lang en             # English-only output
    python src/app.py --lang ur             # Urdu-only text and speech
    python src/app.py --lang both           # English + Urdu (default)
    python src/app.py --speak en            # show both, speak English only
    python src/app.py --no-tts              # disable speech
    python src/app.py --threshold 0.7       # higher confidence gate
    python src/app.py --camera 1            # second webcam
    python src/app.py --interval 3          # inference every 3 frames
    python src/app.py --self-test           # no camera: verify model + landmarker load

Keys: d panel, s skeleton, f full screen, c clear, q / ESC quit.
"""

import argparse
import json
import queue
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import (  # noqa: E402
    FPS,
    HANDS_SLICE,
    IDLE_MARGIN,
    LEFT_HAND_SLICE,
    N_LANDMARKS,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
    SPLITS_DIR,
    URDU_AUDIO_DIR,
    detected,
    display_text,
    preprocess,
)
from evaluate import available_runs  # noqa: E402
from train import MODELS_DIR, load_trained_model  # noqa: E402
from ui import Overlay, Word  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LABEL_MAP_FILE = SPLITS_DIR / "label_map.json"

# --------------------------------------------------------- constants ----

INFERENCE_INTERVAL = 4          # run inference every N camera frames
CONFIDENCE_THRESHOLD = 0.6      # minimum softmax confidence to surface
STABLE_CONFIRM_FRAMES = 3       # same prediction N times in a row → confirmed
WINDOW_SECONDS = 4.0            # history kept; 99% of training signs are shorter
SIGN_GAP_FRAMES = 10            # no-hand frames (at 25 fps, 0.4 s) that end a sign;
                                # 99.5% of training signs have no longer internal gap
HOLD_SECONDS = 1.5              # keep classifying a finished sign this long
MIN_SIGN_FRAMES = 12            # shortest span worth classifying (training minimum: 21)

BUNDLE_DIR = MODELS_DIR / "mediapipe"
BUNDLE_FILE = BUNDLE_DIR / "holistic_landmarker.task"
BUNDLE_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
              "holistic_landmarker/float16/latest/holistic_landmarker.task")

POSE_CONNECTIONS = [(c.start, c.end)
                    for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
HAND_CONNECTIONS = [(c.start, c.end)
                    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


# ----------------------------------------------------------- helpers ----

def load_label_map():
    """class_id → {"label": ..., "en": [...], "ur": [...]} from label_map.json."""
    with open(LABEL_MAP_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    entries = []
    for label, info in raw.items():
        entries.append((info["id"], label, info.get("en", []), info.get("ur", [])))
    entries.sort(key=lambda e: e[0])
    return entries


def load_vocab(run_name):
    """If the run was trained with --classes-file, return its ordered label list."""
    vocab_file = MODELS_DIR / f"{run_name}.vocab.json"
    if vocab_file.is_file():
        with open(vocab_file, encoding="utf-8") as f:
            return json.load(f)
    return None


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


# ------------------------------------------ landmark extraction ----

def ensure_holistic_bundle() -> Path:
    """Download the HolisticLandmarker task bundle on first use (~14 MB)."""
    if not BUNDLE_FILE.is_file():
        BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading MediaPipe holistic bundle -> {BUNDLE_FILE} ...")
        urllib.request.urlretrieve(BUNDLE_URL, BUNDLE_FILE)
        print(f"  done ({BUNDLE_FILE.stat().st_size / 1e6:.1f} MB)")
    return BUNDLE_FILE


def create_landmarker() -> vision.HolisticLandmarker:
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_holistic_bundle())),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def _fill(arr: np.ndarray, offset: int, landmarks, limit: int):
    for i, lm in enumerate(landmarks[:limit]):
        arr[offset + i] = [lm.x, lm.y, lm.z,
                           lm.visibility or 0.0, lm.presence or 0.0]


def extract_landmarks_holistic(result) -> np.ndarray:
    """Build a (75, 5) array matching the training CSV column order.

    The dataset CSVs hold MediaPipe *world* landmarks (metres), so the world
    variants are used here — image-space landmarks would be a train/serve
    mismatch.  Layout: pose 0-32, left hand 33-53, right hand 54-74, each
    (x, y, z, visibility, presence).  Missing landmarks stay zero-filled,
    exactly like the training data.
    """
    arr = np.zeros((N_LANDMARKS, 5), dtype=np.float32)
    _fill(arr, 0, result.pose_world_landmarks, 33)
    _fill(arr, 33, result.left_hand_world_landmarks, 21)
    _fill(arr, 54, result.right_hand_world_landmarks, 21)
    return arr


def draw_landmarks(frame, result, mirrored: bool = False):
    """Draw image-space pose/hand skeletons on the BGR frame (Tasks API has no
    drawing_utils, so connections are stroked manually).  Set mirrored=True when
    the frame was flipped for display but detection ran on the original."""
    h, w = frame.shape[:2]
    thickness = max(1, round(h / 300))

    def to_px(landmarks):
        return [(int((1.0 - lm.x) * w) if mirrored else int(lm.x * w),
                 int(lm.y * h)) for lm in landmarks]

    for landmarks, connections, point_bgr, line_bgr, keep in (
        # upper body only: hips and legs are off screen and not used by the model
        (result.pose_landmarks, POSE_CONNECTIONS, (230, 230, 230), (190, 190, 190), 23),
        (result.left_hand_landmarks, HAND_CONNECTIONS, (255, 190, 90), (240, 160, 60), 21),
        (result.right_hand_landmarks, HAND_CONNECTIONS, (90, 190, 255), (60, 160, 240), 21),
    ):
        if not landmarks:
            continue
        pts = to_px(landmarks)[:keep]
        for a, b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], line_bgr, thickness, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, thickness + 1, point_bgr, -1, cv2.LINE_AA)


# -------------------------------------------- sliding window (live only) ----

def resample_window(times_ms, frames, fps: float = FPS) -> np.ndarray:
    """Frames captured at the webcam's variable rate -> the same span at the
    dataset's fps, ending at the newest frame.  Landmarks are interpolated
    linearly between neighbouring frames; a landmark group missing on either
    side takes the nearer frame's value instead (no blending with zero-fills)."""
    times = np.asarray(times_ms, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float32)  # (N, 75, 5)
    n_out = int((times[-1] - times[0]) * fps / 1000.0) + 1
    grid = times[-1] - np.arange(n_out)[::-1] * (1000.0 / fps)
    right = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
    left = right - 1
    w = np.clip((grid - times[left]) / np.maximum(times[right] - times[left], 1e-6), 0.0, 1.0)
    out = frames[left] * (1 - w)[:, None, None] + frames[right] * w[:, None, None]
    nearest = frames[np.where(w < 0.5, left, right)]
    for group in (POSE_SLICE, LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        both = detected(frames[left][:, group, :3]) & detected(frames[right][:, group, :3])
        out[~both, group] = nearest[~both, group]
    return out


def last_sign(seq: np.ndarray, max_gap: int = SIGN_GAP_FRAMES):
    """The most recent sign in a 25 fps window: the last hand-active span
    (no-hand gaps up to max_gap frames bridged) with IDLE_MARGIN frames of lead-in,
    plus how many frames ago the hands were last seen.  None if no hands at all."""
    hands = np.flatnonzero(detected(seq[:, HANDS_SLICE, :3]))
    if len(hands) == 0:
        return None, None
    breaks = np.flatnonzero(np.diff(hands) - 1 > max_gap)
    start = hands[breaks[-1] + 1] if len(breaks) else hands[0]
    return seq[max(start - IDLE_MARGIN, 0):], len(seq) - 1 - hands[-1]


# -------------------------------------------------------- TTS engine ----

class TTSEngine:
    """Speaks confirmed signs on one background thread, never blocking the video.

    English: the Windows system voice (SAPI) driven directly through comtypes
    (a reused pyttsx3 engine cuts off or skips every word after the first, and
    a new pyttsx3 engine per word hangs); elsewhere one pyttsx3 engine.
    Urdu: Windows has no Urdu voice, so clips pre-generated by
    src/make_urdu_audio.py (data/audio/ur/<label>.mp3) are played offline.
    A sign confirmed while another is still being spoken replaces any sign
    still waiting, so the voice never lags behind the screen.
    """

    def __init__(self, languages=("en",)):
        self.languages = tuple(languages)
        self.enabled = bool(self.languages)
        self._pending: queue.Queue = queue.Queue(maxsize=1)
        self._missing_urdu: set = set()
        if not self.enabled:
            return
        ready, error = threading.Event(), []
        threading.Thread(target=self._run, args=(ready, error), daemon=True).start()
        if not ready.wait(15) or error:
            print(f"TTS unavailable: {error[0] if error else 'voice did not start'}")
            self.enabled = False

    def speak(self, english: str, label: str):
        """Queue one confirmed sign: its English word and/or its Urdu clip."""
        if not self.enabled:
            return
        items = []
        if "en" in self.languages and english:
            items.append(("en", english))
        if "ur" in self.languages:
            clip = URDU_AUDIO_DIR / f"{label}.mp3"
            if clip.is_file():
                items.append(("ur", clip))
            elif label not in self._missing_urdu:
                self._missing_urdu.add(label)
                print(f"no Urdu audio for {label} — run python src/make_urdu_audio.py")
        if not items:
            return
        try:
            self._pending.get_nowait()  # drop a sign that is still waiting
        except queue.Empty:
            pass
        try:
            self._pending.put_nowait(items)
        except queue.Full:
            pass

    def _run(self, ready: threading.Event, error: list):
        try:
            players = {}
            if "en" in self.languages:
                players["en"] = self._make_voice()
            if "ur" in self.languages:
                players["ur"] = self._make_clip_player()
        except Exception as e:  # missing package, no audio device, not Windows, ...
            error.append(e)
            ready.set()
            return
        ready.set()
        while True:
            for kind, payload in self._pending.get():
                try:
                    players[kind](payload)
                except Exception as e:
                    print(f"TTS error: {e}")

    @staticmethod
    def _make_voice():
        """Returns a blocking say(text); must run on the thread that will speak."""
        if sys.platform == "win32":
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()  # COM objects belong to the thread that creates them
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
            voice.Rate = 1  # -10..10; slightly faster than the default
            return lambda text: voice.Speak(text, 0)
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)

        def say(text):
            engine.say(text)
            engine.runAndWait()
        return say

    @staticmethod
    def _make_clip_player():
        """Returns a blocking play(mp3_path) using Windows MCI (no extra packages)."""
        if sys.platform != "win32":
            raise RuntimeError("Urdu playback uses Windows MCI; use --speak en elsewhere")
        import ctypes
        winmm = ctypes.windll.winmm

        def mci(command: str):
            err = winmm.mciSendStringW(command, None, 0, 0)
            if err:
                message = ctypes.create_unicode_buffer(256)
                winmm.mciGetErrorStringW(err, message, 256)
                raise RuntimeError(f"{message.value} ({command})")

        def play(path):
            mci(f'open "{path}" type mpegvideo alias psl_urdu')
            try:
                mci("play psl_urdu wait")
            finally:
                winmm.mciSendStringW("close psl_urdu", None, 0, 0)
        return play


# --------------------------------------------------------- main app ----

def self_test(model, cfg, class_lookup, holistic) -> int:
    """Camera-free check: push synthetic frames through landmarker + model."""
    print("\nself-test (no camera):")
    frames, times = [], []
    n_frames = int(WINDOW_SECONDS * 30)
    for i in range(n_frames):  # a 30 fps camera
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = holistic.detect_for_video(mp_image, i * 33 + 1)
        frames.append(extract_landmarks_holistic(result))
        times.append(i * 33 + 1)
    window = resample_window(times, frames)
    features = preprocess(window, cfg.max_frames, cfg.velocity, cfg.include_confidence)
    print(f"  landmarker + pipeline: {n_frames} frames @30 fps -> {len(window)} @{FPS} fps "
          f"-> {features.shape}")

    t0 = time.perf_counter()
    model.predict_on_batch(features[np.newaxis])  # first call compiles
    compile_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    probs = softmax(model.predict_on_batch(features[np.newaxis]))[0]
    print(f"  inference latency:     {(time.perf_counter() - t0) * 1000:.0f} ms "
          f"(first call incl. compile {compile_ms:.0f} ms)")
    top = np.argsort(probs)[::-1][:3]
    print(f"  model inference:       {len(probs)} classes, top-3 "
          f"{[(class_lookup.get(int(i), {}).get('label', i), round(float(probs[i]), 3)) for i in top]}")
    print("  (blank frames -> no landmarks detected, so the prediction is "
          "meaningless; this only proves the wiring works)")

    holistic.close()
    print("\nself-test PASSED — plug in a camera and run without --self-test")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Real-time PSL sign recognition")
    parser.add_argument("--run", help="checkpoint run name (default: newest)")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"minimum confidence to show prediction (default {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--interval", type=int, default=INFERENCE_INTERVAL,
                        help=f"inference every N frames (default {INFERENCE_INTERVAL})")
    parser.add_argument("--lang", choices=["en", "ur", "both"], default="both",
                        help="output language: en (English only), ur (Urdu only), "
                             "or both (default: both)")
    parser.add_argument("--speak", choices=["en", "ur", "both", "none"],
                        help="spoken language(s) (default: same as --lang); Urdu needs "
                             "clips from src/make_urdu_audio.py")
    parser.add_argument("--no-tts", action="store_true",
                        help="disable text-to-speech (auto-disabled in --lang ur mode)")
    parser.add_argument("--self-test", action="store_true",
                        help="verify checkpoint + landmarker on a synthetic frame, no camera")
    return parser.parse_args(argv)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    # ---- load model ----
    runs = available_runs()
    if not runs:
        sys.exit(f"no trained checkpoints in {MODELS_DIR} — run training first")
    run_name = args.run or runs[0]
    if run_name not in runs:
        sys.exit(f"unknown run {run_name!r}; available: {', '.join(runs)}")

    print(f"loading checkpoint: {run_name} ...")
    model, cfg = load_trained_model(run_name)
    vocab = load_vocab(run_name)
    label_entries = load_label_map()
    print(f"model ready: {cfg.model}, {cfg.num_classes} classes, "
          f"feature_dim {cfg.feature_dim}")

    # Build class_id → (label, en, ur) lookup
    if vocab:
        # Subset run: class ids are dense 0..K-1 by vocab position
        class_lookup = {}
        for i, label in enumerate(vocab):
            info = next((e for e in label_entries if e[1] == label), None)
            if info:
                class_lookup[i] = {"label": label, "en": info[2], "ur": info[3]}
    else:
        class_lookup = {}
        for cid, label, en, ur in label_entries:
            class_lookup[cid] = {"label": label, "en": en, "ur": ur}

    # ---- init MediaPipe HolisticLandmarker (pose + hands in one model) ----
    holistic = create_landmarker()

    if args.self_test:
        return self_test(model, cfg, class_lookup, holistic)

    # ---- language mode ----
    show_en = args.lang in ("en", "both")
    show_ur = args.lang in ("ur", "both")

    # ---- init speech: English via the system voice, Urdu via pre-made clips ----
    speak = "none" if args.no_tts else (args.speak or args.lang)
    tts = TTSEngine({"en": ("en",), "ur": ("ur",), "both": ("en", "ur"), "none": ()}[speak])
    print(f"language: {args.lang} | speech: {speak if tts.enabled else 'off'}")
    if tts.enabled and "ur" in tts.languages:
        missing = [info["label"] for info in class_lookup.values()
                   if not (URDU_AUDIO_DIR / f"{info['label']}.mp3").is_file()]
        if missing:
            print(f"note: {len(missing)} of {len(class_lookup)} signs have no Urdu audio yet — "
                  "run python src/make_urdu_audio.py" + (f" --run {run_name}" if vocab else " --all"))

    # ---- state ----
    words = {cid: Word(info["label"],
                       display_text(info["label"], "en", info["en"]),
                       display_text(info["label"], "ur", info["ur"]))
             for cid, info in class_lookup.items()}
    overlay = Overlay([words[cid] for cid in sorted(words)], show_en, show_ur)
    history: deque = deque()     # (timestamp_ms, (75, 5) landmarks) over WINDOW_SECONDS
    frame_count = 0
    status = "idle"              # idle | signing | hold (classifying a finished sign)
    top3: list = []              # [(Word, probability)] of the latest inference
    stable_count = 0             # consecutive inferences with the same prediction
    last_predicted_label = ""
    start_time = time.perf_counter()
    last_timestamp_ms = -1       # VIDEO mode needs increasing timestamps

    # ---- open camera ----
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {args.camera}")
    overlay.open_window()
    print(f"camera {args.camera} opened — keys: d panel, s skeleton, f full screen, "
          "c clear, q / ESC quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("frame capture failed")
                break

            # Landmarks must come from the UNMIRRORED frame: MediaPipe labels
            # hands by image side, and the training videos were not mirrored,
            # so detecting on a flipped frame would swap left/right hands.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract landmarks via HolisticLandmarker (pose + both hands).
            # VIDEO mode needs strictly increasing millisecond timestamps.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            holistic_result = holistic.detect_for_video(mp_image, timestamp_ms)

            landmarks = extract_landmarks_holistic(holistic_result)
            history.append((timestamp_ms, landmarks))
            while timestamp_ms - history[0][0] > WINDOW_SECONDS * 1000:
                history.popleft()
            frame_count += 1

            # ---- periodic inference on the most recent sign ----
            sign = None
            if frame_count % args.interval == 0 and len(history) >= 2:
                times, frames = zip(*history)
                sign, idle_frames = last_sign(resample_window(times, frames))
                if sign is None or idle_frames > HOLD_SECONDS * FPS:
                    # hands down: the next sign needs fresh confirmations
                    status, stable_count, last_predicted_label = "idle", 0, ""
                    sign = None
                else:
                    status = "signing" if idle_frames <= SIGN_GAP_FRAMES else "hold"
                    if len(sign) < MIN_SIGN_FRAMES:
                        sign = None

            if sign is not None:
                features = preprocess(sign, cfg.max_frames, cfg.velocity, cfg.include_confidence)
                # compiled forward pass (~30 ms); an eager model(x) call runs the
                # BiLSTM recurrence op by op and blocks the video loop for seconds
                probs = softmax(model.predict_on_batch(features[np.newaxis]))[0]
                top3 = [(words[int(i)], float(probs[i]))
                        for i in np.argsort(probs)[::-1][:3] if int(i) in words]
                pred_id = int(np.argmax(probs))
                confidence = float(probs[pred_id])
                word = words.get(pred_id)

                if confidence >= args.threshold and word is not None:
                    # Stability check: same prediction several times in a row
                    if word.label == last_predicted_label:
                        stable_count += 1
                    else:
                        stable_count, last_predicted_label = 1, word.label
                    if stable_count == STABLE_CONFIRM_FRAMES:
                        overlay.confirm(word, confidence)
                        tts.speak(word.en, word.label)
                    elif stable_count > STABLE_CONFIRM_FRAMES:
                        overlay.refresh(word, confidence)
                else:
                    stable_count, last_predicted_label = 0, ""

            # ---- draw: mirrored selfie view, skeleton, caption, panel ----
            view = overlay.fit(cv2.flip(frame, 1))
            overlay.draw_tracking(view, holistic_result, POSE_CONNECTIONS, HAND_CONNECTIONS)
            span_ms = history[-1][0] - history[0][0]
            fps = (len(history) - 1) * 1000.0 / span_ms if span_ms > 0 else 0.0
            overlay.show(overlay.render(view, status, fps, top3))

            if overlay.handle_key(cv2.waitKey(1) & 0xFF) == "quit":
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("\ncamera released, goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
