"""PSL landmark dataset: parsing, normalization, augmentation and sequence loading.

CSV layout (per file = one sign video, per row = one frame, 375 columns):
    75 landmarks x 5 values (x, y, z, visibility, presence), ordered
    pose 0-32, left hand 33-53, right hand 54-74.
World coordinates in meters. MediaPipe centers pose at the hip midpoint and
each hand at its own geometric center; missing pose/hands are zero-filled.

Feature output per sample: (MAX_FRAMES, 225) float32 = 75 landmarks * xyz,
normalized (pose centered on shoulder midpoint, everything scaled by shoulder
width). Set include_confidence=True for (MAX_FRAMES, 375) with visibility and
presence appended.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LANDMARKS_DIR = REPO_ROOT / "data" / "landmarks" / "pk-hfad-1.landmarks-mediapipe-world-csv"
MAPPING_FILE = REPO_ROOT / "data" / "mappings" / "pk-dictionary-mapping.json"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

FILENAME_SUFFIX = ".landmarks-mediapipe-world"

N_LANDMARKS = 75
N_VALUES = 5  # x, y, z, visibility, presence
POSE_SLICE = slice(0, 33)
LEFT_HAND_SLICE = slice(33, 54)
RIGHT_HAND_SLICE = slice(54, 75)
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12  # pose landmark indices

# Covers ~95% of clips fully (frame counts: median 51, p90 69, p95 78, max 169).
# Longer clips are uniformly resampled down, shorter ones zero-padded.
MAX_FRAMES = 80


def label_from_path(csv_path) -> str:
    """pk-hfad-1_book.landmarks-mediapipe-world.csv -> pk-hfad-1_book"""
    return Path(csv_path).stem.replace(FILENAME_SUFFIX, "")


def load_label_to_tokens(mapping_file=MAPPING_FILE) -> dict:
    """{label: {"en": [...], "ur": [...], ...}} from all dictionaries in the mapping."""
    with open(mapping_file, encoding="utf-8") as f:
        dictionaries = json.load(f)
    label_to_tokens = {}
    for dictionary in dictionaries:
        for entry in dictionary["mapping"]:
            if "label" in entry:  # sign-phrase entries have "components" but no "label"
                label_to_tokens[entry["label"]] = entry["token"]
    return label_to_tokens


def parse_csv(csv_path) -> np.ndarray:
    """Load one landmark CSV as (frames, 75, 5) float32."""
    df = pd.read_csv(csv_path)
    if df.shape[1] != N_LANDMARKS * N_VALUES:
        raise ValueError(f"{csv_path}: expected {N_LANDMARKS * N_VALUES} columns, got {df.shape[1]}")
    return df.to_numpy(dtype=np.float32).reshape(-1, N_LANDMARKS, N_VALUES)


def split_groups(seq: np.ndarray):
    """(frames, 75, 5) -> (pose (F,33,5), left_hand (F,21,5), right_hand (F,21,5)) views."""
    return seq[:, POSE_SLICE], seq[:, LEFT_HAND_SLICE], seq[:, RIGHT_HAND_SLICE]


def normalize(seq: np.ndarray) -> np.ndarray:
    """Center pose on the shoulder midpoint and scale everything by shoulder width.

    - Pose xyz: per-frame shift to shoulder midpoint, then / mean shoulder distance.
    - Hand xyz: already hand-centered by MediaPipe (shape only), just / shoulder distance.
    - Zero-filled (missing) pose/hand frames stay exactly zero.
    - visibility/presence channels are left untouched.
    """
    seq = seq.copy()
    pose, left, right = split_groups(seq)

    pose_valid = np.abs(pose[:, :, :3]).sum(axis=(1, 2)) > 0  # (F,) frames with a detected pose
    shoulders = pose[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :3]  # (F, 2, 3)
    center = shoulders.mean(axis=1)  # (F, 3) shoulder midpoint
    dist = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)  # (F,)
    scale = float(dist[pose_valid].mean()) if pose_valid.any() else 1.0
    if scale < 1e-6:
        scale = 1.0

    pose[pose_valid, :, :3] = (pose[pose_valid, :, :3] - center[pose_valid, None, :]) / scale
    for hand in (left, right):
        hand_valid = np.abs(hand[:, :, :3]).sum(axis=(1, 2)) > 0
        hand[hand_valid, :, :3] /= scale
    return seq


def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministic augmentation of a raw (frames, 75, 5) sequence.

    Temporal crop (85-100%), rotation about the vertical axis (+-15 deg),
    uniform scale (0.9-1.1) and Gaussian jitter on detected landmarks only.
    Apply BEFORE normalize() so scaling is re-normalized away except its
    effect on relative proportions.
    """
    n_frames = len(seq)

    # temporal crop
    keep = max(2, int(round(n_frames * rng.uniform(0.85, 1.0))))
    start = rng.integers(0, n_frames - keep + 1)
    seq = seq[start:start + keep].copy()

    xyz = seq[:, :, :3]
    detected = np.abs(xyz).sum(axis=2) > 0  # (F, 75)

    # rotation about the vertical axis (x-z plane)
    theta = np.deg2rad(rng.uniform(-15.0, 15.0))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x, z = xyz[:, :, 0].copy(), xyz[:, :, 2].copy()
    xyz[:, :, 0] = cos_t * x + sin_t * z
    xyz[:, :, 2] = -sin_t * x + cos_t * z

    # uniform scale + jitter (sigma = 1 cm in world meters)
    xyz *= rng.uniform(0.9, 1.1)
    xyz += rng.normal(0.0, 0.01, size=xyz.shape).astype(np.float32)

    xyz[~detected] = 0.0  # keep missing landmarks exactly zero
    seq[:, :, :3] = xyz
    return seq


def fit_length(seq: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Uniformly resample longer sequences, zero-pad shorter ones -> (max_frames, 75, 5)."""
    n_frames = len(seq)
    if n_frames > max_frames:
        idx = np.linspace(0, n_frames - 1, max_frames).round().astype(int)
        return seq[idx]
    if n_frames < max_frames:
        pad = np.zeros((max_frames - n_frames, *seq.shape[1:]), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)
    return seq


def to_features(seq: np.ndarray, include_confidence: bool = False) -> np.ndarray:
    """(frames, 75, 5) -> flat (frames, 225) xyz features, or (frames, 375) with confidences."""
    if include_confidence:
        return seq.reshape(len(seq), -1)
    return seq[:, :, :3].reshape(len(seq), -1)


class PSLDataset:
    """Sequence dataset over a split index file produced by src/data/split.py.

    Index rows: file,label,label_id,aug_seed (aug_seed -1 = original, no augmentation).
    Framework-agnostic: __getitem__ returns (features, label_id) numpy pairs and
    as_arrays() materializes (X, y) for e.g. keras Model.fit.
    """

    def __init__(self, index_file, max_frames: int = MAX_FRAMES,
                 include_confidence: bool = False, landmarks_dir=LANDMARKS_DIR):
        self.index = pd.read_csv(index_file)
        self.max_frames = max_frames
        self.include_confidence = include_confidence
        self.landmarks_dir = Path(landmarks_dir)
        self.num_classes = int(self.index["label_id"].max()) + 1
        self._cache: dict = {}  # raw parsed CSVs, keyed by filename

    def __len__(self) -> int:
        return len(self.index)

    @property
    def feature_dim(self) -> int:
        return N_LANDMARKS * (N_VALUES if self.include_confidence else 3)

    def _raw(self, filename: str) -> np.ndarray:
        if filename not in self._cache:
            self._cache[filename] = parse_csv(self.landmarks_dir / filename)
        return self._cache[filename]

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        seq = self._raw(row["file"])
        aug_seed = int(row["aug_seed"])
        if aug_seed >= 0:
            seq = augment(seq, np.random.default_rng(aug_seed))
        seq = fit_length(normalize(seq), self.max_frames)
        return to_features(seq, self.include_confidence), int(row["label_id"])

    def as_arrays(self):
        """Materialize the whole split: X (N, max_frames, feat), y (N,)."""
        xs, ys = zip(*(self[i] for i in range(len(self))))
        return np.stack(xs), np.array(ys, dtype=np.int64)
