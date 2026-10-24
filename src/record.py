"""Record your own takes of signs with the webcam, to fine-tune the model on you.

The dataset has one signer, so the model knows only their way of signing. A few
takes of each demo word by you teach it yours. For each word the tool records
--takes takes; each take is:
    1. lower your hands until the screen says GO
    2. sign once (as in the GIF from src/show_sign.py), then lower your hands
    3. recording stops by itself 0.5 s after your hands are down
That is the same rest -> sign -> rest shape as the dataset clips. Landmarks are
saved exactly as the live app extracts them, resampled to 25 fps, in the dataset
CSV layout: data/recordings/<label>/take-NN.csv (gitignored). Re-running adds
more takes after the existing ones.

Then train with them and check the takes held out of training:
    python src/train.py --classes-file models/vocab-demo20.txt --recordings data/recordings --run-name demo20-me --epochs 60 --patience 60
    python src/evaluate.py --run demo20-me --split recordings
    python src/app.py --run demo20-me

Run:  python src/record.py airplane name           # 10 takes of each word
      python src/record.py airplane --takes 12 --camera 1
Keys: SPACE  start / stop a take by hand (if your resting hands stay in view)
      u      undo the last saved take
      n      skip to the next word
      q, ESC quit
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import HANDS_SLICE, RECORDINGS_DIR, SPLITS_DIR, save_csv  # noqa: E402
from app import (  # noqa: E402
    create_landmarker, draw_landmarks, extract_landmarks_holistic, resample_window,
)

REST_SECONDS = 0.5      # hands down this long -> GO
END_SECONDS = 0.5       # hands down this long after signing -> take ends
PREROLL_SECONDS = 0.5   # rest kept before the hands appear (dataset clips start at rest)
MIN_SIGN_SECONDS = 0.4  # shorter hand activity is a glitch, not a sign
MAX_TAKE_SECONDS = 8.0


def hands_visible(landmarks: np.ndarray) -> bool:
    return bool(np.abs(landmarks[HANDS_SLICE, :3]).sum() > 0)


class TakeRecorder:
    """rest -> go -> recording state machine over timestamped (75, 5) frames.

    update() returns None, ("saved", (frames, 75, 5) at 25 fps) or ("discarded", reason).
    """

    def __init__(self):
        self.state = "rest"
        self.last_hand_ms = -1e12
        self.preroll: deque = deque()
        self.take: list = []
        self.start_ms = 0.0
        self.manual = False

    def update(self, t_ms: float, landmarks: np.ndarray):
        hands = hands_visible(landmarks)
        if hands:
            self.last_hand_ms = t_ms
        if self.state == "rest":
            if t_ms - self.last_hand_ms >= REST_SECONDS * 1000:
                self.state = "go"
                self.preroll.clear()
            return None
        if self.state == "go":
            self.preroll.append((t_ms, landmarks))
            while t_ms - self.preroll[0][0] > PREROLL_SECONDS * 1000:
                self.preroll.popleft()
            if hands:
                self._start(t_ms)
            return None
        self.take.append((t_ms, landmarks))
        if t_ms - self.start_ms > MAX_TAKE_SECONDS * 1000:
            return self._end(f"longer than {MAX_TAKE_SECONDS:.0f} s")
        if not self.manual and not hands and t_ms - self.last_hand_ms >= END_SECONDS * 1000:
            return self._end()
        return None

    def toggle(self, t_ms: float):
        """SPACE: start a take now, or end a take that SPACE started."""
        if self.state == "recording":
            return self._end() if self.manual else None
        self._start(t_ms, manual=True)
        return None

    def _start(self, t_ms: float, manual: bool = False):
        self.state, self.take, self.start_ms, self.manual = "recording", list(self.preroll), t_ms, manual

    def _end(self, discard_reason: str | None = None):
        take, self.take, self.state = self.take, [], "rest"
        if discard_reason:
            return ("discarded", discard_reason)
        hand_ms = [t for t, lm in take if hands_visible(lm)]
        if not hand_ms or hand_ms[-1] - hand_ms[0] < MIN_SIGN_SECONDS * 1000:
            return ("discarded", "no sign seen")
        times, frames = zip(*take)
        return ("saved", resample_window(times, frames))


def resolve_words(names):
    """CLI names (airplane or pk-hfad-1_airplane) -> [(label, english)]."""
    with open(SPLITS_DIR / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    words = []
    for name in names:
        label = name if name.startswith("pk-hfad-1_") else f"pk-hfad-1_{name}"
        if label not in label_map:
            close = [l.replace("pk-hfad-1_", "") for l in label_map if name.lower() in l.lower()][:8]
            sys.exit(f"unknown sign {name!r}" + (f"; did you mean: {', '.join(close)}" if close else ""))
        words.append((label, (label_map[label]["en"] or [label])[0]))
    return words


def next_take_number(word_dir: Path) -> int:
    numbers = [int(p.stem.split("-")[1]) for p in word_dir.glob("take-*.csv")]
    return max(numbers, default=0) + 1


def put(frame, text, y, scale, colour, thickness=2):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("words", nargs="+", help="signs to record, e.g. airplane name")
    parser.add_argument("--takes", type=int, default=10, help="new takes per word (default 10)")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    args = parser.parse_args()
    words = resolve_words(args.words)

    landmarker = create_landmarker()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {args.camera}")
    t0, last_ts = time.perf_counter(), -1
    print("SPACE start/stop by hand | u undo | n next word | q quit")

    try:
        for label, english in words:
            word_dir = RECORDINGS_DIR / label
            first = next_take_number(word_dir)
            saved: list[Path] = []
            recorder = TakeRecorder()
            message, message_until = f"next word: {english}", time.perf_counter() + 2.0
            print(f"\n{english} ({label}): recording takes {first}..{first + args.takes - 1}")
            while len(saved) < args.takes:
                ok, frame = cap.read()
                if not ok:
                    sys.exit("frame capture failed")
                ts = max(int((time.perf_counter() - t0) * 1000), last_ts + 1)
                last_ts = ts
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
                landmarks = extract_landmarks_holistic(result)
                event = recorder.update(ts, landmarks)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
                if key == ord("n"):
                    break
                if key == ord(" "):
                    event = recorder.toggle(ts) or event
                if key == ord("u") and saved:
                    saved.pop().unlink()
                    message = f"take {first + len(saved)} deleted"
                    message_until = time.perf_counter() + 1.5

                if event and event[0] == "saved":
                    path = word_dir / f"take-{first + len(saved):02d}.csv"
                    save_csv(path, event[1])
                    saved.append(path)
                    message = f"saved take {len(saved)}/{args.takes}"
                    print(f"  {message}: {path.name} ({len(event[1])} frames)")
                    message_until = time.perf_counter() + 1.5
                elif event:
                    message = f"{event[1]} - try again"
                    message_until = time.perf_counter() + 2.0

                # mirrored selfie view; detection ran on the unmirrored frame
                frame = cv2.flip(frame, 1)
                draw_landmarks(frame, result, mirrored=True)
                h = frame.shape[0]
                put(frame, f"{english}   take {len(saved) + 1}/{args.takes}", 40, 1.0, (255, 255, 255))
                status = {"rest": ("lower your hands", (0, 165, 255)),
                          "go": (f"GO - sign '{english}'", (0, 255, 0)),
                          "recording": ("recording... lower hands when done"
                                        if not recorder.manual else "recording... SPACE to stop",
                                        (0, 0, 255))}[recorder.state]
                put(frame, status[0], h - 60, 1.0, status[1])
                if time.perf_counter() < message_until:
                    put(frame, message, h - 20, 0.7, (255, 255, 0), 1)
                cv2.imshow("PSL recorder", frame)
            print(f"  {english}: {len(saved)} takes saved in {word_dir}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
    print("\nnext: train with --recordings data/recordings (see python src/record.py --help)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
