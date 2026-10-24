"""Check the browser port (docs/js) against the Python pipeline and model.

Takes replay clips (the source videos as the live MediaPipe sees them; falls
back to the dataset CSVs), simulates webcam timing at 25 and 15 fps, and runs
the live app's path in Python — resample to 25 fps, cut the latest sign,
preprocess, BiLSTM — then the same through docs/js/pipeline.js and
docs/js/model.js in Node, and compares every stage.

Needs Node.js (https://nodejs.org) and an export: python src/export_web.py --run <run>
Run:  python src/check_web.py --run demo20-me
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402
import data.dataset as D  # noqa: E402
from train import load_trained_model  # noqa: E402

TOLERANCE = {"window": 1e-6, "features": 1e-4, "logits": 1e-3}


def make_fixture(run: str, n_extra: int, seed: int = 0) -> dict:
    model, cfg = load_trained_model(run)
    vocab = app.load_vocab(run) or []
    labels = list(vocab)
    rng = np.random.default_rng(seed)
    others = sorted(p.name.replace(D.FILENAME_SUFFIX + ".csv", "") for p in D.LANDMARKS_DIR.glob("*.csv"))
    labels += [l for l in rng.permutation(others) if l not in labels][:n_extra]
    clips = []
    for label in labels:
        name = f"{label}{D.FILENAME_SUFFIX}.csv"
        source = D.REPLAY_DIR / name if (D.REPLAY_DIR / name).is_file() else D.LANDMARKS_DIR / name
        seq = D.parse_csv(source)
        t25 = np.arange(len(seq)) * 40.0
        for cam_fps in (25, 15):
            if cam_fps == 25:
                times, frames = t25, seq
            else:  # a slower, jittery webcam
                step = 1000.0 / cam_fps
                times = np.cumsum(step * rng.uniform(0.85, 1.15, int(t25[-1] / step) + 1))
                times = times[times <= t25[-1]]
                frames = seq[np.clip(np.round(times / 40.0).astype(int), 0, len(seq) - 1)]
            window = app.resample_window(times, frames)
            sign, idle = app.last_sign(window)
            if sign is None:
                continue
            features = D.preprocess(sign, cfg.max_frames, cfg.velocity, cfg.include_confidence)
            logits = model.predict_on_batch(features[np.newaxis])[0]
            clips.append({"label": label, "fps": cam_fps, "times": [float(t) for t in times],
                          "frames": np.asarray(frames, np.float32).reshape(len(frames), -1).tolist(),
                          "window": window.reshape(-1).tolist(), "idle": int(idle),
                          "sign_length": len(sign), "features": features.reshape(-1).tolist(),
                          "logits": logits.tolist()})
    return {"run": run, "clips": clips}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="the run exported to docs/")
    parser.add_argument("--extra", type=int, default=10, help="clips of signs outside the vocabulary")
    args = parser.parse_args()
    node = shutil.which("node")
    if node is None:
        sys.exit("Node.js not found — install it from https://nodejs.org to run this check")
    docs = D.REPO_ROOT / "docs"
    manifest = json.loads((docs / "model" / "manifest.json").read_text(encoding="utf-8"))
    if manifest["run"] != args.run:
        sys.exit(f"docs/ holds an export of {manifest['run']!r}; run src/export_web.py --run {args.run}")

    fixture = make_fixture(args.run, args.extra)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixture.json"
        path.write_text(json.dumps(fixture), encoding="utf-8")
        result = subprocess.run([node, str(Path(__file__).with_name("check_web.mjs")), str(path),
                                 str(docs)], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"node failed:\n{result.stderr}")
    report = json.loads(result.stdout.strip().splitlines()[-1])
    print(f"{report['clips']} clips (live path at 25 and 15 fps) through Python and the browser code:")
    ok = True
    for key, tol in TOLERANCE.items():
        good = report[key] <= tol
        ok &= good
        print(f"  {key:9s} max |python - js| = {report[key]:.2e}  (tolerance {tol:g})  "
              f"{'OK' if good else 'MISMATCH'}")
    for key in ("idle_mismatch", "sign_length_mismatch", "argmax_mismatch"):
        good = report[key] == 0
        ok &= good
        print(f"  {key:21s} {report[key]}  {'OK' if good else 'MISMATCH'}")
    print(f"  browser model: {report['ms_per_inference']:.1f} ms per inference (Node)")
    print("PASS — the browser version matches Python" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
