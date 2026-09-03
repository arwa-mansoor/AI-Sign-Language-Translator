"""Phase 4 — evaluate a trained PSL checkpoint on a held-out split.

Loads models/{run}.config.json + models/{run}.weights.h5 (rebuilt via
build_model), materializes data/splits/{split}.csv through the same feature
pipeline as training, and reports:

  - top-1 / top-5 accuracy
  - macro and weighted precision / recall / F1
  - worst classes by F1 with per-class precision / recall / support
  - top confused pairs
  - confidence-threshold sweep: accuracy among accepted predictions vs
    coverage — how reliable a live demo would feel if it only commits to
    high-confidence predictions

Runs trained with a vocabulary subset (--classes-file in train.py) have a
models/{run}.vocab.json; the split is restricted and re-indexed to match.

Outputs (regenerable, gitignored):
  models/{run}.evaluation.txt   the report
  models/{run}.per-class.csv    label,en,support,precision,recall,f1

Run:
  python src/evaluate.py                          # newest checkpoint in models/
  python src/evaluate.py --run bilstm             # specific run, test split
  python src/evaluate.py --run bilstm-vocab100 --worst 30
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import PSLDataset, SPLITS_DIR  # noqa: E402
from train import MODELS_DIR, load_trained_model  # noqa: E402

CONFIDENCE_THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


# ------------------------------------------------------------- loading ----

def available_runs(models_dir=MODELS_DIR) -> list[str]:
    """Run names with both config and weights, newest weights first."""
    runs = []
    for weights in sorted(Path(models_dir).glob("*.weights.h5"),
                          key=lambda p: p.stat().st_mtime, reverse=True):
        run = weights.name[: -len(".weights.h5")]
        if (Path(models_dir) / f"{run}.config.json").is_file():
            runs.append(run)
    return runs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained PSL checkpoint")
    parser.add_argument("--run", help="run name under models/ (default: newest checkpoint)")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="split to evaluate (default: test)")
    parser.add_argument("--worst", type=int, default=20,
                        help="how many worst classes to list (default: 20)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    runs = available_runs()
    if not runs:
        sys.exit(f"no checkpoints ({{run}}.config.json + {{run}}.weights.h5) in {MODELS_DIR}")
    if args.run is None:
        args.run = runs[0]
        print(f"no --run given; using newest checkpoint: {args.run}")
    elif args.run not in runs:
        sys.exit(f"unknown run {args.run!r}; available: {', '.join(runs)}")
    return args


def load_split_data(cfg, run_name, split, models_dir=MODELS_DIR) -> tuple[PSLDataset, list, list]:
    """(dataset, labels, english tokens) for the split, restricted to the run's vocabulary.

    Subset checkpoints come in two kinds, mirroring train.py: runs trained with
    --classes-file carry a vocab.json (explicit label list, dense ids by list
    position); --dry-run runs use the first num_classes label ids.
    """
    index = pd.read_csv(SPLITS_DIR / f"{split}.csv")
    vocab_file = Path(models_dir) / f"{run_name}.vocab.json"
    if vocab_file.is_file():
        with open(vocab_file, encoding="utf-8") as f:
            vocab = json.load(f)
        index = index[index["label"].isin(vocab)].copy()
        remap = {label: i for i, label in enumerate(vocab)}
        index["label_id"] = index["label"].map(remap)
    elif cfg.num_classes < int(index["label_id"].max()) + 1:
        index = index[index["label_id"] < cfg.num_classes].copy()
    labels = index.drop_duplicates("label_id").sort_values("label_id")["label"].tolist()

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / f"{split}.csv"
        index.to_csv(path, index=False)
        ds = PSLDataset(path, include_confidence=cfg.feature_dim == 375)

    with open(SPLITS_DIR / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    en = [", ".join(t for t in label_map.get(label, {}).get("en", [])[:2] if t)
          for label in labels]
    return ds, labels, en


def history_summary(run_name, models_dir=MODELS_DIR) -> str:
    """Best epoch / val accuracy from the training history, if present."""
    history_file = Path(models_dir) / f"{run_name}.history.csv"
    if not history_file.is_file():
        return "training history not found"
    history = pd.read_csv(history_file)
    best = history[history["is_best"] == 1].iloc[-1]
    return (f"{len(history)} epochs, best epoch {int(best['epoch'])} "
            f"(val acc {best['val_acc']:.4f})")


# ------------------------------------------------------------ metrics ----

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def predict(model, X, batch_size=256) -> np.ndarray:
    return np.concatenate([model(X[i:i + batch_size], training=False).numpy()
                           for i in range(0, len(X), batch_size)])


def worst_table(labels, en, p, r, f1, sup, worst) -> list[str]:
    below = int((f1 < 0.5).sum())
    order = sorted(range(len(labels)), key=lambda i: (f1[i], p[i], r[i], labels[i]))
    rows = [i for i in order if f1[i] < 1.0 or sup[i] == 0][:worst]
    lines = [f"per-class F1: mean {f1.mean():.3f}, median {np.median(f1):.3f}; "
             f"classes below 0.5: {below}/{len(labels)}",
             f"worst {len(rows)} classes by F1 (support = test samples per class):",
             "   f1  prec  rec  sup  label"]
    for i in rows:
        lines.append(f"  {f1[i]:.2f} {p[i]:.2f} {r[i]:.2f} {int(sup[i]):4d}  "
                     f"{labels[i]}   ({en[i]})")
    return lines


def confused_pairs(cm, labels, en, top=15) -> list[str]:
    n = len(labels)
    pairs = sorted(((int(cm[i, j]), i, j) for i in range(n) for j in range(n)
                    if i != j and cm[i, j] > 0), reverse=True)
    lines = [f"top {min(top, len(pairs))} confused pairs (true -> predicted, count):"]
    for count, i, j in pairs[:top]:
        lines.append(f"  {count:4d}  {labels[i]} -> {labels[j]}   ({en[i]} | {en[j]})")
    return lines


def confidence_sweep(probs, y_true, thresholds=CONFIDENCE_THRESHOLDS) -> list[str]:
    """Accuracy among predictions accepted at each max-softmax threshold."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    lines = ["confidence sweep (accept only if max softmax >= threshold):",
             "  thr   coverage  accuracy"]
    for thr in thresholds:
        mask = conf >= thr
        acc = f"{(pred[mask] == y_true[mask]).mean():8.1%}" if mask.any() else "       -"
        lines.append(f"  {thr:4.2f}  {mask.mean():8.1%}  {acc}")
    return lines


# ---------------------------------------------------------------- main ----

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    t0 = time.perf_counter()

    model, cfg = load_trained_model(args.run)
    ds, labels, en = load_split_data(cfg, args.run, args.split)
    if len(labels) != cfg.num_classes or ds.num_classes != cfg.num_classes:
        sys.exit(f"class mismatch: checkpoint has {cfg.num_classes} classes, "
                 f"{args.split} split has {len(labels)}")
    X, y = ds.as_arrays()
    print(f"evaluating {args.run} on {args.split}: {len(y)} samples, "
          f"{cfg.num_classes} classes")

    logits = predict(model, X, args.batch_size)
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    acc = float((pred == y).mean())
    top5_acc = float((top5 == y[:, None]).any(axis=1).mean())

    p, r, f1, sup = precision_recall_fscore_support(
        y, pred, labels=range(cfg.num_classes), zero_division=0)
    weight = sup / sup.sum()
    report = [
        f"PSL evaluation — {args.run} on {args.split}",
        f"model: {cfg.model} ({model.count_params():,} params), "
        f"feature_dim {cfg.feature_dim} | {history_summary(args.run)}",
        f"data: {len(y)} samples, {cfg.num_classes} classes "
        f"(random-guess top-1 = {1 / cfg.num_classes:.2%})",
        "",
        f"accuracy:  top-1 {acc:.4f}   top-5 {top5_acc:.4f}",
        f"macro:     precision {p.mean():.4f}  recall {r.mean():.4f}  f1 {f1.mean():.4f}",
        f"weighted:  precision {(p * weight).sum():.4f}  "
        f"recall {(r * weight).sum():.4f}  f1 {(f1 * weight).sum():.4f}",
        "",
        *worst_table(labels, en, p, r, f1, sup, args.worst),
        "",
        *confused_pairs(np.bincount(y * cfg.num_classes + pred,
                                    minlength=cfg.num_classes ** 2
                                    ).reshape(cfg.num_classes, cfg.num_classes),
                        labels, en),
        "",
        *confidence_sweep(probs, y),
        "",
        "caveat: samples are augmented variants of the same videos as training "
        "(one recording per sign), so this measures perturbation robustness, "
        "not person-generalization; live-webcam accuracy will be lower.",
        "",
        f"artifacts: models/{args.run}.evaluation.txt, models/{args.run}.per-class.csv",
    ]

    text = "\n".join(report)
    print()
    print(text)
    print(f"\n(evaluated {len(y)} samples in {time.perf_counter() - t0:.1f}s)")
    (MODELS_DIR / f"{args.run}.evaluation.txt").write_text(text + "\n", encoding="utf-8")
    pd.DataFrame({"label": labels, "en": en, "support": sup,
                  "precision": p, "recall": r, "f1": f1}
                 ).to_csv(MODELS_DIR / f"{args.run}.per-class.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
