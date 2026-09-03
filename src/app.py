"""Phase 5 — real-time webcam sign-language recognition app.

Captures webcam frames with OpenCV, extracts pose + hand landmarks via
MediaPipe in the exact same order and normalization as the training pipeline
(src/data/dataset.py), buffers a sliding window of MAX_FRAMES frames, runs
model inference periodically, and overlays the predicted English + Urdu text
on the live video feed.  A pyttsx3 text-to-speech callout speaks the English
translation when a new stable prediction is confirmed.

Uses the MediaPipe Tasks HolisticLandmarker (mediapipe >= 1.0; the older
mp.solutions.holistic API no longer exists).  The task bundle is downloaded
once from Google's model CDN into models/mediapipe/ on first run.

Run:
    python src/app.py                       # newest checkpoint, both languages, TTS on
    python src/app.py --run bilstm          # specific run
    python src/app.py --lang en             # English-only output
    python src/app.py --lang ur             # Urdu-only output (TTS auto-disabled)
    python src/app.py --lang both           # English + Urdu (default)
    python src/app.py --no-tts              # disable text-to-speech
    python src/app.py --threshold 0.7       # higher confidence gate
    python src/app.py --camera 1            # second webcam
    python src/app.py --interval 5          # inference every 5 frames
    python src/app.py --self-test           # no camera: verify model + landmarker load

Press 'q' or ESC to quit.
"""

import argparse
import json
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
    MAX_FRAMES,
    N_LANDMARKS,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    SPLITS_DIR,
)
from evaluate import available_runs  # noqa: E402
from train import MODELS_DIR, load_trained_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LABEL_MAP_FILE = SPLITS_DIR / "label_map.json"

# --------------------------------------------------------- constants ----

INFERENCE_INTERVAL = 8          # run inference every N frames
CONFIDENCE_THRESHOLD = 0.6      # minimum softmax confidence to surface
STABLE_CONFIRM_FRAMES = 3       # same prediction N times in a row → confirmed
TTS_COOLDOWN_SEC = 2.0          # minimum seconds between spoken words
WINDOW_MAX_FRAMES = MAX_FRAMES  # 80 — must match training

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

    def to_px(landmarks):
        return [(int((1.0 - lm.x) * w) if mirrored else int(lm.x * w),
                 int(lm.y * h)) for lm in landmarks]

    for landmarks, connections, point_bgr, line_bgr in (
        (result.pose_landmarks, POSE_CONNECTIONS, (0, 255, 0), (0, 200, 0)),
        (result.left_hand_landmarks, HAND_CONNECTIONS, (255, 100, 0), (255, 150, 0)),
        (result.right_hand_landmarks, HAND_CONNECTIONS, (0, 100, 255), (0, 150, 255)),
    ):
        if not landmarks:
            continue
        pts = to_px(landmarks)
        for a, b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], line_bgr, 1)
        for p in pts:
            cv2.circle(frame, p, 2, point_bgr, -1)


# --------------------------------------- normalization (matches dataset.py) ----

def normalize_frame(seq: np.ndarray) -> np.ndarray:
    """Normalize a (T, 75, 5) sequence the same way as dataset.normalize().

    Center pose on shoulder midpoint, scale everything by mean shoulder width.
    Hand landmarks are already hand-centered by MediaPipe, only scaled.
    """
    seq = seq.copy()
    pose = seq[:, :33]
    left = seq[:, 33:54]
    right = seq[:, 54:75]

    pose_valid = np.abs(pose[:, :, :3]).sum(axis=(1, 2)) > 0  # (T,)
    shoulders = pose[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :3]   # (T, 2, 3)
    center = shoulders.mean(axis=1)                              # (T, 3)
    dist = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)  # (T,)
    scale = float(dist[pose_valid].mean()) if pose_valid.any() else 1.0
    if scale < 1e-6:
        scale = 1.0

    pose[pose_valid, :, :3] = (pose[pose_valid, :, :3] - center[pose_valid, None, :]) / scale
    for hand in (left, right):
        hand_valid = np.abs(hand[:, :, :3]).sum(axis=(1, 2)) > 0
        hand[hand_valid, :, :3] /= scale
    return seq


def fit_length(seq: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Uniformly resample longer sequences, zero-pad shorter ones."""
    n = len(seq)
    if n > max_frames:
        idx = np.linspace(0, n - 1, max_frames).round().astype(int)
        return seq[idx]
    if n < max_frames:
        pad = np.zeros((max_frames - n, *seq.shape[1:]), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)
    return seq


def to_features(seq: np.ndarray) -> np.ndarray:
    """(T, 75, 5) → (T, 225) xyz-only features."""
    return seq[:, :, :3].reshape(len(seq), -1)


# -------------------------------------------------------- TTS engine ----

class TTSEngine:
    """Background text-to-speech with cooldown to avoid repeating the same word."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_spoken = ""
        self._last_time = 0.0
        self._engine = None
        self._lock = threading.Lock()
        if enabled:
            self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 160)
        except Exception as e:
            print(f"TTS unavailable: {e}")
            self.enabled = False

    def speak(self, text: str):
        if not self.enabled or not text or text == self._last_spoken:
            return
        now = time.time()
        if now - self._last_time < TTS_COOLDOWN_SEC:
            return
        with self._lock:
            self._last_spoken = text
            self._last_time = now
            # Run in a thread so we don't block the video loop
            threading.Thread(target=self._do_speak, args=(text,), daemon=True).start()

    def _do_speak(self, text: str):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass


# --------------------------------------------------------- main app ----

def self_test(model, cfg, class_lookup, holistic) -> int:
    """Camera-free check: push synthetic frames through landmarker + model."""
    print("\nself-test (no camera):")
    frames = []
    for i in range(WINDOW_MAX_FRAMES):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = holistic.detect_for_video(mp_image, i * 33 + 1)
        frames.append(extract_landmarks_holistic(result))
    seq = fit_length(normalize_frame(np.array(frames)))
    features = seq.reshape(len(seq), -1) if cfg.feature_dim == 375 else to_features(seq)
    print(f"  landmarker + pipeline: {WINDOW_MAX_FRAMES} frames -> {features.shape}")

    probs = softmax(model(features[np.newaxis], training=False).numpy())[0]
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

    # ---- init TTS (English only; auto-disabled in Urdu-only mode) ----
    tts_enabled = not args.no_tts and show_en
    tts = TTSEngine(enabled=tts_enabled)
    if args.lang == "ur" and not args.no_tts:
        print("note: TTS auto-disabled in --lang ur mode (English speech not applicable)")
    print(f"language: {args.lang} | TTS: {'on' if tts.enabled else 'off'}")

    # ---- state ----
    buffer: deque[np.ndarray] = deque(maxlen=WINDOW_MAX_FRAMES)
    frame_count = 0
    current_prediction = ""      # displayed text
    current_urdu = ""
    current_confidence = 0.0
    stable_count = 0             # consecutive frames with same prediction
    last_predicted_label = ""    # last confirmed label (for TTS dedup)
    start_time = time.perf_counter()
    last_timestamp_ms = -1       # VIDEO mode needs increasing timestamps

    # ---- open camera ----
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {args.camera}")
    print(f"camera {args.camera} opened — press 'q' or ESC to quit")

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
            buffer.append(landmarks)
            frame_count += 1

            # Mirror for a natural selfie view, and draw with mirrored x
            frame = cv2.flip(frame, 1)
            draw_landmarks(frame, holistic_result, mirrored=True)

            # ---- periodic inference ----
            if frame_count % args.interval == 0 and len(buffer) >= 10:
                seq = np.array(list(buffer))  # (T, 75, 5)
                seq = fit_length(normalize_frame(seq))  # (80, 75, 5)
                features = to_features(seq)              # (80, 225)

                # Handle feature_dim mismatch (375 if model was trained with confidence)
                if cfg.feature_dim == 375:
                    features = seq.reshape(len(seq), -1)

                x = features[np.newaxis]  # (1, 80, feat_dim)
                logits = model(x, training=False).numpy()
                probs = softmax(logits)[0]
                pred_id = int(np.argmax(probs))
                confidence = float(probs[pred_id])

                if confidence >= args.threshold:
                    info = class_lookup.get(pred_id)
                    if info:
                        label = info["label"]
                        en_text = info["en"][0] if info["en"] else label
                        ur_text = info["ur"][0] if info["ur"] else ""

                        # Stability check: same prediction multiple times
                        if label == last_predicted_label:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_predicted_label = label

                        if stable_count >= STABLE_CONFIRM_FRAMES:
                            current_prediction = en_text
                            current_urdu = ur_text
                            current_confidence = confidence
                            tts.speak(en_text)
                else:
                    # Below threshold — clear display
                    current_prediction = ""
                    current_urdu = ""
                    current_confidence = 0.0
                    stable_count = 0
                    last_predicted_label = ""

            # ---- overlay text ----
            h, w = frame.shape[:2]
            y_offset = h - 50
            if current_prediction:
                # English text (left side)
                if show_en:
                    en_display = f"Sign: {current_prediction}"
                    conf_display = f"({current_confidence:.0%})"
                    cv2.putText(frame, en_display, (20, y_offset - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    cv2.putText(frame, conf_display, (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)
                # Urdu text (right side)
                if show_ur and current_urdu:
                    cv2.putText(frame, current_urdu, (w - 300, y_offset - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "...", (20, y_offset - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 1)

            # Frame counter and mode indicator
            mode_label = f"[{args.lang.upper()}] frames: {len(buffer)}/{WINDOW_MAX_FRAMES}"
            cv2.putText(frame, mode_label,
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("PSL Sign Language Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
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
