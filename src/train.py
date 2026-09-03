"""Phase 3 — train the PSL sign classifier.

Standard training loop over the Phase 1 splits with the Phase 2 models:
SparseCategoricalCrossentropy(from_logits=True), Adam/AdamW, cosine LR decay,
early stopping on val accuracy with best-weight restore.

Artifacts written to models/ (run_name defaults to the model type):
  {run}.weights.h5    best checkpoint, by val accuracy
  {run}.config.json   exact ModelConfig used (rebuild via build_model)
  {run}.history.csv   per-epoch train/val loss + accuracy, lr, seconds
  {run}.report.txt    final metrics, top confused pairs, hardest classes, advice
  {run}.confusion.npz dense (num_classes, num_classes) matrix over val+test

Reload a checkpoint for inference (e.g. the Phase 4 webcam app):
    from train import load_trained_model
    model, cfg = load_trained_model("bilstm")

Run:
    python src/train.py --dry-run          # 24 classes, 2 epochs, smoke test
    python src/train.py                    # full 775-class run
    python src/train.py --config configs/transformer.json --epochs 80
Config precedence: defaults < --config JSON < explicit CLI flags.
"""

import argparse
import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import LANDMARKS_DIR, PSLDataset, SPLITS_DIR  # noqa: E402
from model import ModelConfig, build_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


# ---------------------------------------------------------------- config ----

@dataclasses.dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4      # AdamW only
    optimizer: str = "adamw"        # "adam" | "adamw"
    lr_schedule: str = "cosine"     # "cosine" | "none"
    patience: int = 8               # early stopping on val accuracy
    seed: int = 42
    run_name: str = ""              # default: model type ("dry-run-<model>" with --dry-run)
    dry_run: bool = False
    dry_run_classes: int = 24


def parse_args(argv=None) -> tuple[ModelConfig, TrainConfig]:
    """defaults < --config JSON < explicit CLI flags (same rule as src/model.py)."""
    parser = argparse.ArgumentParser(description="Train the PSL sign classifier")
    parser.add_argument("--config", type=Path, help="JSON file with config overrides")
    for field in dataclasses.fields(ModelConfig) + dataclasses.fields(TrainConfig):
        flag = f"--{field.name.replace('_', '-')}"
        if field.type is bool:
            parser.add_argument(flag, action="store_true", default=None)
        else:
            parser.add_argument(flag, type=field.type, default=None)
    args = parser.parse_args(argv)

    train_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    train_values, model_values = {}, {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            for key, value in json.load(f).items():
                if key in train_fields:
                    train_values[key] = value
                elif key in model_fields:
                    model_values[key] = value
                else:
                    parser.error(f"unknown config key: {key!r}")
    for name in train_fields | model_fields:
        if getattr(args, name) is not None:
            target = train_values if name in train_fields else model_values
            target[name] = getattr(args, name)

    model_cfg = ModelConfig(**model_values)
    if model_cfg.model not in ("bilstm", "transformer"):
        parser.error(f"unknown model type: {model_cfg.model}")
    cfg = TrainConfig(**train_values)
    if cfg.optimizer not in ("adam", "adamw"):
        parser.error(f"unknown optimizer: {cfg.optimizer}")
    if cfg.lr_schedule not in ("cosine", "none"):
        parser.error(f"unknown lr_schedule: {cfg.lr_schedule}")

    if cfg.dry_run:
        if "epochs" not in train_values:
            cfg.epochs = 2
        if "patience" not in train_values:
            cfg.patience = 2
        if not cfg.run_name:
            cfg.run_name = f"dry-run-{model_cfg.model}"
    if not cfg.run_name:
        cfg.run_name = model_cfg.model
    return model_cfg, cfg


# ------------------------------------------------------------------ data ----

def load_split(index_file, n_classes, include_confidence, tmp_dir) -> PSLDataset:
    """PSLDataset for one split; with n_classes set, keep only the first n_classes ids."""
    if n_classes is None:
        return PSLDataset(index_file, include_confidence=include_confidence)
    index = pd.read_csv(index_file)
    index = index[index["label_id"] < n_classes]
    path = Path(tmp_dir) / Path(index_file).name
    index.to_csv(path, index=False)
    return PSLDataset(path, include_confidence=include_confidence)


def class_info(train_ds) -> tuple[list, list]:
    """(labels, english-token strings) indexed by class id, from the split index."""
    idx = train_ds.index.drop_duplicates("label_id").sort_values("label_id")
    labels = idx["label"].tolist()
    en = [""] * len(labels)
    label_map_file = SPLITS_DIR / "label_map.json"
    if label_map_file.is_file():
        with open(label_map_file, encoding="utf-8") as f:
            label_map = json.load(f)
        en = [", ".join(t for t in label_map.get(label, {}).get("en", [])[:2] if t)
              for label in labels]
    return labels, en


# ------------------------------------------------------------- optimizer ----

def make_optimizer(cfg: TrainConfig, total_steps: int) -> keras.Optimizer:
    lr = cfg.learning_rate
    if cfg.lr_schedule == "cosine":
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.learning_rate,
            decay_steps=max(total_steps, 1), alpha=0.1)
    if cfg.optimizer == "adamw":
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=cfg.weight_decay)
    return keras.optimizers.Adam(learning_rate=lr)


def current_lr(optimizer) -> float:
    lr = optimizer.learning_rate
    if callable(lr):  # LR schedule: evaluate at the current step
        return float(lr(tf.cast(optimizer.iterations, tf.float32)))
    return float(lr)


# --------------------------------------------------------------- training ----

def evaluate(model, loss_fn, X, y, batch_size):
    """(mean loss, top-1 acc, top-5 acc, predictions) over materialized arrays."""
    n = len(y)
    total_loss = correct = top5 = 0
    preds = np.empty(n, dtype=np.int64)
    for start in range(0, n, batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model(xb, training=False).numpy()
        total_loss += float(loss_fn(yb, logits)) * len(yb)
        top5_idx = np.argsort(-logits, axis=1)[:, :5]
        correct += int((top5_idx[:, 0] == yb).sum())
        top5 += int((top5_idx == yb[:, None]).any(axis=1).sum())
        preds[start:start + len(yb)] = top5_idx[:, 0]
    return total_loss / n, correct / n, top5 / n, preds


def train(model, cfg: TrainConfig, loss_fn, optimizer,
          X_train, y_train, X_val, y_val, models_dir, run_name):
    """Per-epoch train/val logging, best-checkpoint saving, early stopping on
    val accuracy with best-weight restore. Returns (rows, best_val_acc, best_epoch)."""
    rng = np.random.default_rng(cfg.seed)

    @tf.function(reduce_retracing=True)
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            loss = loss_fn(yb, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, logits

    history_file = models_dir / f"{run_name}.history.csv"
    with open(history_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,seconds,is_best\n")

    best_val_acc, best_epoch, best_weights = -1.0, 0, None
    rows = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(len(y_train))
        tr_loss = tr_correct = 0
        for start in range(0, len(y_train), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            loss, logits = train_step(X_train[idx], y_train[idx])
            tr_loss += float(loss) * len(idx)
            tr_correct += int((logits.numpy().argmax(-1) == y_train[idx]).sum())
        val_loss, val_acc, _, _ = evaluate(model, loss_fn, X_val, y_val, cfg.batch_size)
        seconds = time.perf_counter() - t0

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc, best_epoch = val_acc, epoch
            best_weights = model.get_weights()
            model.save_weights(models_dir / f"{run_name}.weights.h5")

        row = (epoch, tr_loss / len(y_train), tr_correct / len(y_train),
               val_loss, val_acc, current_lr(optimizer), seconds, int(is_best))
        rows.append(row)
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")
        print(f"epoch {epoch:3d}/{cfg.epochs}  train loss {row[1]:.4f} acc {row[2]:.4f}  "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}  lr {row[5]:.2e}  {seconds:5.1f}s"
              + ("  *" if is_best else ""), flush=True)

        if epoch - best_epoch >= cfg.patience:
            print(f"early stopping: no val accuracy improvement for {cfg.patience} epochs")
            break

    if best_weights is not None:
        model.set_weights(best_weights)
    return rows, best_val_acc, best_epoch


# --------------------------------------------------------------- analysis ----

def confusion_report(y_true, y_pred, labels, en_tokens, top_pairs=25, hardest=20):
    """(report lines, dense confusion matrix) for val+test predictions."""
    n = len(labels)
    cm = np.bincount(y_true.astype(np.int64) * n + y_pred.astype(np.int64),
                     minlength=n * n).reshape(n, n)
    name = lambda i: en_tokens[i] or labels[i]  # noqa: E731

    pairs = sorted(((int(cm[i, j]), i, j) for i in range(n) for j in range(n)
                    if i != j and cm[i, j] > 0), reverse=True)
    lines = [f"confusion over {len(y_true)} val+test samples; "
             f"per-class counts min {int(cm.sum(1).min())} max {int(cm.sum(1).max())}",
             "",
             f"top {min(top_pairs, len(pairs))} confused pairs (true -> predicted, count):"]
    for count, i, j in pairs[:top_pairs]:
        lines.append(f"  {count:4d}  {labels[i]} -> {labels[j]}   ({name(i)} | {name(j)})")

    totals = cm.sum(axis=1)
    accs = np.divide(cm.diagonal(), totals, out=np.zeros(n, dtype=float), where=totals > 0)
    lines.append("")
    lines.append(f"per-class accuracy: mean {accs.mean():.3f}, median {np.median(accs):.3f}, "
                 f"min {accs.min():.3f}; classes below 50%: {int((accs < 0.5).sum())}/{n}")
    worst = [i for i in np.argsort(accs, kind="stable") if accs[i] < 1.0][:hardest]
    if worst:
        lines.append("")
        lines.append(f"hardest {len(worst)} classes (per-class accuracy):")
        for i in worst:
            lines.append(f"  {accs[i]:.2f}  {labels[i]}   ({name(i)})")
    return lines, cm


def accuracy_advice(test_acc, top5_acc, num_classes):
    """Heuristic next-step recommendations — ordered levers for a one-video-per-sign dataset."""
    lines = ["", "advice:"]
    if test_acc >= 0.85:
        lines.append(f"  accuracy {test_acc:.1%} is strong for {num_classes} one-shot classes — "
                     "good enough to proceed to the real-time webcam phase.")
    elif test_acc >= 0.60:
        lines.append(f"  accuracy {test_acc:.1%} is workable but improvable; levers by expected payoff:")
        lines.append("    1. more variants: raise VARIANTS_PER_CLASS 12 -> 24 in src/data/split.py, "
                     "rerun it, retrain")
        lines.append("    2. stronger augmentation: widen rotation/scale/jitter ranges in "
                     "src/data/dataset.py augment()")
        lines.append("    3. other architecture: python src/train.py --config configs/transformer.json")
    else:
        lines.append(f"  accuracy {test_acc:.1%} is POOR for {num_classes} classes; levers, cheapest first:")
        lines.append("    1. more augmented variants per class (VARIANTS_PER_CLASS 12 -> 24+ in "
                     "src/data/split.py) — with one video per sign, variant count is the main data lever")
        lines.append("    2. bigger/different model: --config configs/transformer.json, or "
                     "--hidden-size 256 --num-layers 3")
        lines.append("    3. fewer classes: 775 one-shot classes is inherently hard; restrict to the "
                     "signs you need, or merge visually identical classes (see confused pairs above)")
    if top5_acc - test_acc > 0.15:
        lines.append(f"  top-5 {top5_acc:.1%} >> top-1 {test_acc:.1%}: errors cluster in confusable "
                     "sign families — merging those classes or adding context beats more training.")
    lines.append("  caveat: val/test are augmented variants of the same videos as train (one recording "
                 "per sign), so these numbers measure perturbation robustness, not person-generalization.")
    return lines


# ------------------------------------------------------------ checkpoint ----

def load_trained_model(run_name, models_dir=None):
    """Rebuild a trained model from its saved config + weights (for Phase 4)."""
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    with open(models_dir / f"{run_name}.config.json", encoding="utf-8") as f:
        cfg = ModelConfig(**json.load(f))
    model = build_model(cfg)
    model.load_weights(models_dir / f"{run_name}.weights.h5")
    return model, cfg


# ------------------------------------------------------------------ main ----

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model_cfg, cfg = parse_args()
    keras.utils.set_random_seed(cfg.seed)

    if model_cfg.feature_dim not in (225, 375):
        sys.exit(f"unsupported feature_dim {model_cfg.feature_dim} (training expects 225 or 375)")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"run: {cfg.run_name} | model: {model_cfg.model} | {cfg.optimizer} "
          f"(lr {cfg.learning_rate:g}, {cfg.lr_schedule}) | devices: {len(gpus)} GPU, "
          f"{len(tf.config.list_physical_devices('CPU'))} CPU | tensorflow {tf.__version__}")
    if cfg.dry_run:
        print(f"DRY RUN: {cfg.dry_run_classes} classes, {cfg.epochs} epochs — "
              "verifying the full loop end-to-end")

    n_classes = cfg.dry_run_classes if cfg.dry_run else None
    include_confidence = model_cfg.feature_dim == 375
    if not any(LANDMARKS_DIR.glob("*.csv")):
        sys.exit(f"no landmark CSVs in {LANDMARKS_DIR}\n"
                 "the dataset is gitignored — download it first, see README "
                 "'To reproduce the downloads'")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_dir:
        data = {}
        for name in ("train", "val", "test"):
            ds = load_split(SPLITS_DIR / f"{name}.csv", n_classes, include_confidence, tmp_dir)
            print(f"materializing {name:5s} ({len(ds)} samples)...", flush=True)
            data[name] = (ds, *ds.as_arrays())
    train_ds, X_train, y_train = data["train"]
    _, X_val, y_val = data["val"]
    _, X_test, y_test = data["test"]

    if model_cfg.num_classes <= 0:
        model_cfg.num_classes = train_ds.num_classes
    labels, en_tokens = class_info(train_ds)
    if len(labels) != model_cfg.num_classes:
        sys.exit(f"class count mismatch: {len(labels)} labels vs {model_cfg.num_classes} ids")
    print(f"data ready in {time.perf_counter() - t0:.1f}s: {len(y_train)}/{len(y_val)}/{len(y_test)} "
          f"train/val/test samples, {model_cfg.num_classes} classes, "
          f"feature_dim {model_cfg.feature_dim}")

    model = build_model(model_cfg)
    print(f"model: {model_cfg.model}, {model.count_params():,} params")

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    total_steps = ((len(y_train) + cfg.batch_size - 1) // cfg.batch_size) * cfg.epochs
    optimizer = make_optimizer(cfg, total_steps)
    optimizer.build(model.trainable_variables)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / f"{cfg.run_name}.config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(model_cfg), f, indent=2)

    rows, best_val_acc, best_epoch = train(model, cfg, loss_fn, optimizer,
                                           X_train, y_train, X_val, y_val,
                                           MODELS_DIR, cfg.run_name)

    val_loss, val_acc, _, val_preds = evaluate(model, loss_fn, X_val, y_val, cfg.batch_size)
    test_loss, test_acc, test_top5, test_preds = evaluate(model, loss_fn, X_test, y_test,
                                                          cfg.batch_size)
    conf_lines, cm = confusion_report(np.concatenate([y_val, y_test]),
                                      np.concatenate([val_preds, test_preds]),
                                      labels, en_tokens)
    report = [
        f"PSL training report — {cfg.run_name}",
        f"model: {model_cfg.model} ({model.count_params():,} params) | {len(rows)} epochs, "
        f"best epoch {best_epoch} (val acc {best_val_acc:.4f})",
        f"data: {len(y_train)}/{len(y_val)}/{len(y_test)} train/val/test, "
        f"{model_cfg.num_classes} classes, feature_dim {model_cfg.feature_dim}",
        f"final (best weights): val acc {val_acc:.4f} (loss {val_loss:.4f}) | "
        f"test acc {test_acc:.4f} (loss {test_loss:.4f}), top-5 {test_top5:.4f}",
        "",
        *conf_lines,
        *accuracy_advice(test_acc, test_top5, model_cfg.num_classes),
        "",
        f"artifacts: models/{cfg.run_name}.weights.h5, .config.json, "
        f".history.csv, .report.txt, .confusion.npz",
    ]
    text = "\n".join(report)
    print()
    print(text)
    (MODELS_DIR / f"{cfg.run_name}.report.txt").write_text(text + "\n", encoding="utf-8")
    np.savez_compressed(MODELS_DIR / f"{cfg.run_name}.confusion.npz",
                        confusion=cm, labels=np.array(labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
