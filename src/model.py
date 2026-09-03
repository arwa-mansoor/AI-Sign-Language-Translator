"""Phase 2 — sequence classifiers for PSL sign recognition.

Two architectures behind one config-driven factory:
  bilstm       Bidirectional LSTM baseline (default; simpler, faster to debug)
  transformer  small pre-LN Transformer encoder (swap-in alternative)

Input:  (batch, frames, feature_dim) float32 landmark features from PSLDataset
        (225 = xyz only, the pipeline default; 375 with include_confidence=True).
Output: (batch, num_classes) raw logits — train with
        SparseCategoricalCrossentropy(from_logits=True).

Padding handling: padded / missing-detection frames are all-zero, so a boolean
frame mask is derived from "any feature != 0". The BiLSTM uses keras Masking
(the recurrence skips masked steps); the Transformer uses the mask both as an
attention mask and for masked mean-pooling. Padding therefore never influences
the logits (verified by the invariance check in __main__).

Run the smoke test:  python src/model.py --model bilstm
                     python src/model.py --model transformer
Config precedence: defaults < --config JSON < explicit CLI flags.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

REPO_ROOT = Path(__file__).resolve().parent.parent
LABEL_MAP_FILE = REPO_ROOT / "data" / "splits" / "label_map.json"


# ---------------------------------------------------------------- config ----

@dataclasses.dataclass
class ModelConfig:
    model: str = "bilstm"          # "bilstm" | "transformer"
    feature_dim: int = 225         # 225 xyz-only, 375 with confidence channels
    num_classes: int = 0           # 0 = auto-detect from label_map.json
    hidden_size: int = 128         # LSTM units per direction / transformer d_model
    num_layers: int = 2            # stacked BiLSTM layers / encoder blocks
    dropout: float = 0.3
    num_heads: int = 4             # transformer only
    ff_dim: int = 256              # transformer feed-forward width

    @classmethod
    def from_args(cls, argv=None) -> "ModelConfig":
        """defaults < --config JSON < explicit CLI flags."""
        parser = argparse.ArgumentParser(description="PSL sequence classifier")
        parser.add_argument("--config", type=Path, help="JSON file with config overrides")
        for field in dataclasses.fields(cls):
            parser.add_argument(f"--{field.name.replace('_', '-')}",
                                type=field.type, default=None)
        args = parser.parse_args(argv)

        values = {}
        if args.config:
            with open(args.config, encoding="utf-8") as f:
                values.update(json.load(f))
        for field in dataclasses.fields(cls):
            cli = getattr(args, field.name)
            if cli is not None:
                values[field.name] = cli
        cfg = cls(**values)
        if cfg.model not in ("bilstm", "transformer"):
            parser.error(f"unknown model type: {cfg.model}")
        if cfg.num_classes <= 0:  # auto-detect from Phase 1 label map
            cfg.num_classes = num_classes_from_label_map() or 775
        return cfg


def num_classes_from_label_map(path=LABEL_MAP_FILE) -> int | None:
    if not Path(path).is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return len(json.load(f))


# ------------------------------------------------------------ components ----

class FrameMask(layers.Layer):
    """(B, T, F) -> (B, T) bool, True where the frame has any nonzero feature."""

    def call(self, x):
        return tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)


class PositionalEncoding(layers.Layer):
    """Add sinusoidal positional encodings (computed for the dynamic length T)."""

    def call(self, x):
        t = tf.shape(x)[1]
        d = x.shape[-1]
        pos = tf.cast(tf.range(t)[:, None], tf.float32)
        i = tf.cast(tf.range(d)[None, :], tf.float32)
        angle = pos / tf.pow(10000.0, (2.0 * (i // 2)) / float(d))
        pe = tf.where(tf.range(d)[None, :] % 2 == 0, tf.sin(angle), tf.cos(angle))
        return x + pe[None]


class TransformerBlock(layers.Layer):
    """Pre-LN encoder block: MHA (with padding mask) + feed-forward, residual."""

    def __init__(self, d_model, num_heads, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(num_heads, d_model // num_heads,
                                             dropout=dropout)
        self.drop1 = layers.Dropout(dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, mask):
        attn_mask = mask[:, None, :] & mask[:, :, None]  # (B, T, T)
        h = self.ln1(x)
        x = x + self.drop1(self.mha(h, h, attention_mask=attn_mask))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x


class MaskedMeanPool(layers.Layer):
    """Mean over time using only unmasked (real) frames."""

    def call(self, x, mask):
        m = tf.cast(mask, x.dtype)[:, :, None]           # (B, T, 1)
        return tf.reduce_sum(x * m, axis=1) / tf.maximum(tf.reduce_sum(m, axis=1), 1.0)


# -------------------------------------------------------------- builders ----

def build_bilstm(cfg: ModelConfig) -> keras.Model:
    inp = layers.Input(shape=(None, cfg.feature_dim), name="landmarks")
    x = layers.Masking(mask_value=0.0)(inp)  # recurrence skips all-zero frames
    for i in range(cfg.num_layers):
        last = i == cfg.num_layers - 1
        x = layers.Bidirectional(
            layers.LSTM(cfg.hidden_size, return_sequences=not last))(x)
        x = layers.Dropout(cfg.dropout)(x)
    logits = layers.Dense(cfg.num_classes, name="logits")(x)
    return keras.Model(inp, logits, name="bilstm")


def build_transformer(cfg: ModelConfig) -> keras.Model:
    inp = layers.Input(shape=(None, cfg.feature_dim), name="landmarks")
    mask = FrameMask()(inp)
    x = layers.Dense(cfg.hidden_size, name="proj")(inp)
    x = PositionalEncoding()(x)
    x = layers.Dropout(cfg.dropout)(x)
    for i in range(cfg.num_layers):
        x = TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.ff_dim,
                             cfg.dropout, name=f"encoder_{i}")(x, mask)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = MaskedMeanPool()(x, mask)
    x = layers.Dropout(cfg.dropout)(x)
    logits = layers.Dense(cfg.num_classes, name="logits")(x)
    return keras.Model(inp, logits, name="transformer")


def build_model(cfg: ModelConfig) -> keras.Model:
    builder = {"bilstm": build_bilstm, "transformer": build_transformer}[cfg.model]
    return builder(cfg)


# ------------------------------------------------------------ smoke test ----

def _smoke_test(cfg: ModelConfig) -> int:
    keras.utils.set_random_seed(42)
    model = build_model(cfg)
    model.summary()

    # 1. forward pass: batch of 4, last 30 of 80 frames zero-padded
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 80, cfg.feature_dim)).astype(np.float32)
    x[:, 50:] = 0.0
    logits = model(x, training=False).numpy()
    assert logits.shape == (4, cfg.num_classes), logits.shape
    assert np.isfinite(logits).all(), "non-finite logits"
    print(f"\nforward pass:      x {x.shape} -> logits {logits.shape}  [OK]")

    # 2. padding invariance: same real frames, with vs without trailing padding
    trimmed = model(x[:, :50], training=False).numpy()
    max_diff = float(np.abs(logits - trimmed).max())
    assert max_diff < 1e-4, f"padding changed logits by {max_diff}"
    print(f"padding invariance: max |logits(padded) - logits(trimmed)| = {max_diff:.2e}  [OK]")

    # 3. real data, if Phase 1 splits exist
    train_index = REPO_ROOT / "data" / "splits" / "train.csv"
    if train_index.is_file() and cfg.feature_dim in (225, 375):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from data.dataset import PSLDataset
        ds = PSLDataset(train_index, include_confidence=cfg.feature_dim == 375)
        xb = np.stack([ds[i][0] for i in range(4)])
        yb = np.array([ds[i][1] for i in range(4)])
        real_logits = model(xb, training=False).numpy()
        assert real_logits.shape == (4, cfg.num_classes)
        print(f"real batch:        x {xb.shape}, labels {yb.tolist()} -> "
              f"logits {real_logits.shape}  [OK]")
    else:
        print("real batch:        skipped (no data/splits/train.csv or custom feature_dim)")

    print(f"\n{cfg.model} smoke test passed. params: {model.count_params():,}")
    return 0


if __name__ == "__main__":
    config = ModelConfig.from_args()
    print(f"config: {config}\n")
    sys.exit(_smoke_test(config))
