"""Export a trained checkpoint for the browser version (docs/, GitHub Pages).

Writes into docs/:
  model/weights.bin     float32 little-endian weights of the BiLSTM
  model/manifest.json   tensor offsets and shapes, the class list (English /
                        Urdu words, Urdu audio clip), and every pipeline and
                        live-app constant, read from data/dataset.py and app.py
                        so the JavaScript port cannot drift from training
  audio/ur/<label>.mp3  the spoken Urdu clips (src/make_urdu_audio.py)

The browser re-implements the feature pipeline and the BiLSTM in plain
JavaScript (docs/js/pipeline.js, docs/js/model.js); `python src/check_web.py`
verifies it against this Python code on real clips.

Run:  python src/export_web.py --run demo20-me
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402
import data.dataset as D  # noqa: E402
from train import MODELS_DIR, load_trained_model  # noqa: E402

DOCS_DIR = D.REPO_ROOT / "docs"
MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm"


def lstm_tensors(layer):
    """Bidirectional LSTM -> {direction: {kernel, recurrent_kernel, bias}} as float32 arrays."""
    out = {}
    for direction, sub in (("forward", layer.forward_layer), ("backward", layer.backward_layer)):
        cell = {w.name: np.asarray(w.numpy(), np.float32) for w in sub.weights}
        out[direction] = {k: cell[k] for k in ("kernel", "recurrent_kernel", "bias")}
    return out


def export(run: str, out_dir: Path) -> Path:
    model, cfg = load_trained_model(run)
    if cfg.model != "bilstm" or cfg.include_confidence or not cfg.velocity:
        sys.exit("the web version supports BiLSTM checkpoints with the default features only")

    blobs, tensors = [], {}

    def add(name, array):
        array = np.ascontiguousarray(array, np.float32)
        tensors[name] = {"offset": sum(b.size for b in blobs), "shape": list(array.shape)}
        blobs.append(array.ravel())

    lstm_layers = [l for l in model.layers if type(l).__name__ == "Bidirectional"]
    for i, layer in enumerate(lstm_layers):
        for direction, cell in lstm_tensors(layer).items():
            for key, array in cell.items():
                add(f"lstm{i}.{direction}.{key}", array)
    dense = model.get_layer("logits")
    add("dense.kernel", dense.kernel.numpy())
    add("dense.bias", dense.bias.numpy())

    model_dir, audio_dir = out_dir / "model", out_dir / "audio" / "ur"
    model_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    np.concatenate(blobs).astype("<f4").tofile(model_dir / "weights.bin")

    vocab = app.load_vocab(run)
    entries = {label: (cid, en, ur) for cid, label, en, ur in app.load_label_map()}
    labels = vocab or sorted(entries, key=lambda label: entries[label][0])[:cfg.num_classes]
    classes, missing_audio = [], []
    for label in labels:
        _, en, ur = entries[label]
        clip = D.URDU_AUDIO_DIR / f"{label}.mp3"
        audio = None
        if clip.is_file():
            shutil.copyfile(clip, audio_dir / clip.name)
            audio = f"audio/ur/{clip.name}"
        else:
            missing_audio.append(label)
        classes.append({"label": label, "en": D.display_text(label, "en", en),
                        "ur": D.display_text(label, "ur", ur), "audio": audio})

    manifest = {
        "run": run,
        "model": {"type": "bilstm", "hidden": cfg.hidden_size, "layers": len(lstm_layers),
                  "feature_dim": cfg.feature_dim, "num_classes": cfg.num_classes,
                  "weights": "weights.bin", "tensors": tensors},
        "pipeline": {
            "fps": D.FPS, "max_frames": cfg.max_frames, "idle_margin": D.IDLE_MARGIN,
            "n_landmarks": D.N_LANDMARKS, "n_values": D.N_VALUES,
            "pose_slice": [0, 33], "left_hand_slice": [33, 54], "right_hand_slice": [54, 75],
            "left_shoulder": D.LEFT_SHOULDER, "right_shoulder": D.RIGHT_SHOULDER,
            "palm": D.PALM, "n_pose_features": D.N_POSE_FEATURES,
            "feature_landmarks": [int(i) for i in D.FEATURE_LANDMARKS],
            "hand_feature_scale": D.HAND_FEATURE_SCALE, "velocity_scale": D.VELOCITY_SCALE,
        },
        "app": {
            "window_seconds": app.WINDOW_SECONDS, "sign_gap_frames": app.SIGN_GAP_FRAMES,
            "hold_seconds": app.HOLD_SECONDS, "min_sign_frames": app.MIN_SIGN_FRAMES,
            "stable_confirm": app.STABLE_CONFIRM_FRAMES, "threshold": app.CONFIDENCE_THRESHOLD,
            "interval": app.INFERENCE_INTERVAL,
        },
        "mediapipe": {"model": app.BUNDLE_URL, "wasm": MEDIAPIPE_WASM},
        "classes": classes,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    size = (model_dir / "weights.bin").stat().st_size / 1e6
    print(f"exported {run}: {cfg.num_classes} classes, {size:.1f} MB weights -> {model_dir}")
    if missing_audio:
        print(f"no Urdu clip for {len(missing_audio)} signs (run src/make_urdu_audio.py --run {run}): "
              + ", ".join(l.replace("pk-hfad-1_", "") for l in missing_audio[:8]))
    return model_dir


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="checkpoint under models/, e.g. demo20-me")
    parser.add_argument("--out", type=Path, default=DOCS_DIR, help="site folder (default docs/)")
    args = parser.parse_args()
    if not (MODELS_DIR / f"{args.run}.weights.h5").is_file():
        sys.exit(f"no checkpoint models/{args.run}.weights.h5")
    export(args.run, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
