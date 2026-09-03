"""Phase 5 — real-time webcam sign-language recognition app.

Captures webcam frames with OpenCV, extracts pose + hand landmarks via
MediaPipe in the exact same order and normalization as the training pipeline
(src/data/dataset.py), buffers a sliding window of MAX_FRAMES frames, runs
model inference periodically, and overlays the predicted English + Urdu text
on the live video feed.  A pyttsx3 text-to-speech callout speaks the English
translation when a new stable prediction is confirmed.

Run:
    python src/app.py                       # newest checkpoint
    python src/app.py --run bilstm          # specific run
    python src/app.py --threshold 0.7       # higher confidence gate
    python src/app.py --camera 1            # second webcam
    python src/app.py --no-tts              # disable text-to-speech

Press 'q' or ESC to quit.
"""

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

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

def extract_landmarks_holistic(holistic_results):
    """Build a (75, 5) array matching the training CSV column order.

    MediaPipe Holistic gives pose + hands in one pass with consistent
    world-coordinate frames — same layout as the dataset CSVs:
      pose 0-32, left hand 33-53, right hand 54-74
    Each landmark: (x, y, z, visibility, presence).
    Missing landmarks are zero-filled.
    """
    arr = np.zeros((N_LANDMARKS, 5), dtype=np.float32)

    # Pose landmarks 0-32
    if holistic_results.pose_landmarks:
        for i, lm in enumerate(holistic_results.pose_landmarks.landmark):
            if i < 33:
                arr[i] = [lm.x, lm.y, lm.z, lm.visibility, lm.presence]

    # Left hand landmarks 33-53
    if holistic_results.left_hand_landmarks:
        for i, lm in enumerate(holistic_results.left_hand_landmarks.landmark):
            if i < 21:
                arr[33 + i] = [lm.x, lm.y, lm.z, lm.visibility, lm.presence]

    # Right hand landmarks 54-74
    if holistic_results.right_hand_landmarks:
        for i, lm in enumerate(holistic_results.right_hand_landmarks.landmark):
            if i < 21:
                arr[54 + i] = [lm.x, lm.y, lm.z, lm.visibility, lm.presence]

    return arr


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

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Real-time PSL sign recognition")
    parser.add_argument("--run", help="checkpoint run name (default: newest)")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"minimum confidence to show prediction (default {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--interval", type=int, default=INFERENCE_INTERVAL,
                        help=f"inference every N frames (default {INFERENCE_INTERVAL})")
    parser.add_argument("--no-tts", action="store_true", help="disable text-to-speech")
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

    # ---- init MediaPipe Holistic (pose + hands in one model) ----
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ---- init TTS ----
    tts = TTSEngine(enabled=not args.no_tts)

    # ---- state ----
    buffer: deque[np.ndarray] = deque(maxlen=WINDOW_MAX_FRAMES)
    frame_count = 0
    current_prediction = ""      # displayed text
    current_urdu = ""
    current_confidence = 0.0
    stable_count = 0             # consecutive frames with same prediction
    last_predicted_label = ""    # last confirmed label (for TTS dedup)

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

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            # Extract landmarks via Holistic (pose + both hands)
            holistic_result = holistic.process(rgb)

            landmarks = extract_landmarks_holistic(holistic_result)
            buffer.append(landmarks)
            frame_count += 1

            # Draw landmarks on frame for visual feedback
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if holistic_result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, holistic_result.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 200, 0), thickness=1),
                )
            if holistic_result.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, holistic_result.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 100, 0), thickness=1, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 150, 0), thickness=1),
                )
            if holistic_result.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, holistic_result.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 100, 255), thickness=1, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 150, 255), thickness=1),
                )

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
            if current_prediction:
                # English text
                en_display = f"Sign: {current_prediction}"
                conf_display = f"({current_confidence:.0%})"
                cv2.putText(frame, en_display, (20, h - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.putText(frame, conf_display, (20, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)
                # Urdu text (right-aligned)
                if current_urdu:
                    cv2.putText(frame, current_urdu, (w - 300, h - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "...", (20, h - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 1)

            # Frame counter and buffer info
            cv2.putText(frame, f"frames: {len(buffer)}/{WINDOW_MAX_FRAMES}",
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
