"""Replay the source videos through the live app's landmark extraction.

The dataset CSVs were extracted with an older MediaPipe release; src/app.py
uses the MediaPipe Tasks HolisticLandmarker.  This script runs every source
video through the exact live code path (app.create_landmarker +
app.extract_landmarks_holistic) and writes the result as a "replay" landmark
set: same signs and signer, but landmarks produced the way the webcam app
produces them.  `python src/evaluate.py --split replay` then measures
train/serve parity (extractor differences, preprocessing mismatches), which
the augmented val/test splits cannot.  It is still the training signer, so it
is an upper bound on accuracy for new signers, not an estimate of it.

Needs the source videos (330 MB, gitignored), PowerShell:
    Invoke-WebRequest "https://github.com/sign-language-translator/sign-language-datasets/releases/download/v0.0.3/pk-hfad-1.videos-mp4.zip" -OutFile "data\\videos\\pk-hfad-1.videos-mp4.zip"
    Expand-Archive "data\\videos\\pk-hfad-1.videos-mp4.zip" -DestinationPath "data\\videos\\pk-hfad-1.videos-mp4"

Output: data/landmarks/pk-hfad-1.landmarks-replay-csv/ — same file names and
375-column layout as the dataset CSVs, raw (un-normalized) world landmarks.

Run:  python src/extract_replay.py              # all videos (resumable; ~40 min on a
      python src/extract_replay.py --workers 2  #  4-thread laptop CPU, ~50 ms/frame)
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import FILENAME_SUFFIX, REPLAY_DIR, REPO_ROOT, save_csv  # noqa: E402

VIDEOS_DIR = REPO_ROOT / "data" / "videos" / "pk-hfad-1.videos-mp4"


def extract_video(video_path: Path, out_path: Path) -> int:
    """Run one video through the live extraction; returns the frame count."""
    import cv2
    import mediapipe as mp
    import app  # heavy import (TensorFlow) — only inside workers

    landmarker = app.create_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(round(len(frames) * 1000.0 / fps)))
        frames.append(app.extract_landmarks_holistic(result))
    cap.release()
    landmarker.close()

    save_csv(out_path, frames)
    return len(frames)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Re-extract landmarks with the live pipeline")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        sys.exit(f"no videos in {VIDEOS_DIR} — download them first (see this file's docstring)")
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(v, REPLAY_DIR / f"{v.stem}{FILENAME_SUFFIX}.csv") for v in videos]
    todo = [(v, out) for v, out in todo if not out.is_file()]
    print(f"{len(videos)} videos, {len(videos) - len(todo)} already extracted, "
          f"{len(todo)} to go ({args.workers} workers)", flush=True)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(extract_video, v, out) for v, out in todo]
        for done, (future, (video, _)) in enumerate(zip(futures, todo), 1):
            n_frames = future.result()
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}  {video.stem} ({n_frames} frames)  "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"done: {REPLAY_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
