"""PSL landmark dataset: parsing, normalization, augmentation and sequence loading.

CSV layout (per file = one sign video, per row = one frame, 375 columns):
    75 landmarks x 5 values (x, y, z, visibility, presence), ordered
    pose 0-32, left hand 33-53, right hand 54-74.
World coordinates in meters; source videos are 25 fps. Missing pose/hands are
zero-filled. The dataset CSVs center each hand on itself, but the live app's
MediaPipe Tasks extractor reports hands in the pose frame — normalize()
re-centers every hand on its palm, so both give identical features.

Feature pipeline, shared by training, evaluation and src/app.py via preprocess():
    trim_idle    drop rest frames before the first / after the last hand detection
    [augment]    training only, on raw world coordinates
    normalize    pose centered on the shoulder midpoint, hands on their palm,
                 everything scaled by mean shoulder width
    to_features  65 landmarks: pose 0-22 as x/y + both hands as x/y/z = 172
                 values, plus their frame-to-frame velocity (velocity=True) = 344,
                 plus visibility/presence (include_confidence=True)
    fit_length   uniformly resample / zero-pad to MAX_FRAMES
Lower-body pose landmarks 23-32 are not used: they are out of frame in
pk-hfad-1 (mean visibility 0.00-0.32), so MediaPipe only guesses them — and
the live app's MediaPipe guesses them ~29 cm differently. Pose depth (z) is not
used either: it is the least consistent coordinate across MediaPipe versions.
"""

import dataclasses
import functools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LANDMARKS_DIR = REPO_ROOT / "data" / "landmarks" / "pk-hfad-1.landmarks-mediapipe-world-csv"
# same videos re-extracted through the live app's MediaPipe (src/extract_replay.py)
REPLAY_DIR = REPO_ROOT / "data" / "landmarks" / "pk-hfad-1.landmarks-replay-csv"
# your own webcam takes (src/record.py): <label>/take-NN.csv
RECORDINGS_DIR = REPO_ROOT / "data" / "recordings"
# spoken Urdu for the live app, one <label>.mp3 per sign (src/make_urdu_audio.py)
URDU_AUDIO_DIR = REPO_ROOT / "data" / "audio" / "ur"
MAPPING_FILE = REPO_ROOT / "data" / "mappings" / "pk-dictionary-mapping.json"
# per-sign corrections of the displayed / spoken word, e.g. car -> گاڑی
DISPLAY_OVERRIDES_FILE = REPO_ROOT / "data" / "mappings" / "display-overrides.json"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

FILENAME_SUFFIX = ".landmarks-mediapipe-world"
FPS = 25  # pk-hfad-1 video frame rate; the live app resamples the webcam to it

N_LANDMARKS = 75
N_VALUES = 5  # x, y, z, visibility, presence
CSV_HEADER = ",".join(f"{v}{i}" for i in range(N_LANDMARKS) for v in "xyzab")
POSE_SLICE = slice(0, 33)
LEFT_HAND_SLICE = slice(33, 54)
RIGHT_HAND_SLICE = slice(54, 75)
HANDS_SLICE = slice(33, 75)
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12  # pose landmark indices
PALM = [0, 5, 9, 13, 17]  # hand landmarks on the rigid palm (wrist + finger MCPs)

N_POSE_FEATURES = 23  # pose 0-22: face, shoulders, elbows, wrists, pose hand points
FEATURE_LANDMARKS = np.r_[0:N_POSE_FEATURES, HANDS_SLICE.start:HANDS_SLICE.stop]  # 65
FEATURE_GROUPS = (slice(0, 23), slice(23, 44), slice(44, 65))  # pose, left, right
_GROUP_STARTS = [g.start for g in FEATURE_GROUPS]
_GROUP_OF_LANDMARK = np.repeat([0, 1, 2], [g.stop - g.start for g in FEATURE_GROUPS])
# Feature conditioning: normalized hand shapes spread ~0.1 vs ~0.55 for the pose,
# and per-frame motion ~0.03; these bring every block to a similar scale.
HAND_FEATURE_SCALE = 5.0
VELOCITY_SCALE = 10.0
_POSITION_SCALE = np.where(_GROUP_OF_LANDMARK == 0, 1.0, HAND_FEATURE_SCALE
                           ).astype(np.float32)[:, None]  # (65, 1)

# After idle trimming, frame counts: median 41, p90 60, p95 70, max 161.
# Longer clips are uniformly resampled down, shorter ones zero-padded.
MAX_FRAMES = 64
IDLE_MARGIN = 3  # rest frames kept on each side of the hand-active span
# Kernel width of AugmentConfig.pose_drift: the dataset-vs-live pose difference has
# autocorrelation 0.92 / 0.69 / 0.17 at 1 / 3 / 8 frames, matched by a 2.5-frame kernel.
DRIFT_FRAMES = 2.5


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


def save_csv(csv_path, seq) -> None:
    """Write a raw (frames, 75, 5) sequence in the dataset's 375-column CSV layout."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".tmp")
    np.savetxt(tmp, np.asarray(seq, dtype=np.float32).reshape(len(seq), -1), fmt="%.4f",
               delimiter=",", header=CSV_HEADER, comments="")
    tmp.replace(csv_path)  # atomic: an interrupted write never leaves a truncated CSV


def display_word(tokens, fallback: str = "") -> str:
    """First dictionary token without its sense note: doctor(physician) -> doctor."""
    text = tokens[0] if tokens else fallback
    return re.sub(r"\s*\([^)]*\)", "", text).strip() or text


@functools.lru_cache(maxsize=1)
def _display_overrides() -> dict:
    if not DISPLAY_OVERRIDES_FILE.is_file():
        return {}
    with open(DISPLAY_OVERRIDES_FILE, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def display_text(label: str, lang: str, tokens) -> str:
    """The word shown and spoken for a sign in "en" or "ur": the entry in
    data/mappings/display-overrides.json if any, else the dictionary's first token."""
    override = _display_overrides().get(label, {}).get(lang)
    if override:
        return override
    return display_word(tokens, label.replace("pk-hfad-1_", "") if lang == "en" else "")


def parse_csv(csv_path) -> np.ndarray:
    """Load one landmark CSV as (frames, 75, 5) float32."""
    df = pd.read_csv(csv_path)
    if df.shape[1] != N_LANDMARKS * N_VALUES:
        raise ValueError(f"{csv_path}: expected {N_LANDMARKS * N_VALUES} columns, got {df.shape[1]}")
    return df.to_numpy(dtype=np.float32).reshape(-1, N_LANDMARKS, N_VALUES)


def detected(xyz: np.ndarray) -> np.ndarray:
    """(frames, L, 3) landmark group -> (frames,) True where it was detected (not zero-filled)."""
    return np.abs(xyz).sum(axis=(1, 2)) > 0


def trim_idle(seq: np.ndarray, margin: int = IDLE_MARGIN) -> np.ndarray:
    """Drop leading/trailing frames with no hand detected, keeping `margin` frames.

    Training clips and the live app's sliding window then both cover just the
    signing span, however long the signer rested before or after it.
    """
    hands = np.flatnonzero(detected(seq[:, HANDS_SLICE, :3]))
    if len(hands) == 0:
        return seq
    return seq[max(hands[0] - margin, 0): hands[-1] + margin + 1]


def normalize(seq: np.ndarray) -> np.ndarray:
    """Center pose on the shoulder midpoint, hands on their palm; scale by shoulder width.

    - Pose xyz: per-frame shift to shoulder midpoint, then / mean shoulder distance.
    - Hand xyz: per-frame shift to the palm center (hand shape only — hand position
      is carried by pose wrist/hand points 15-22), then / mean shoulder distance.
    - Zero-filled (missing) pose/hand frames stay exactly zero.
    - visibility/presence channels are left untouched.
    """
    seq = seq.copy()
    xyz = seq[:, :, :3]

    pose_valid = detected(xyz[:, POSE_SLICE])
    shoulders = xyz[:, [LEFT_SHOULDER, RIGHT_SHOULDER]]  # (F, 2, 3)
    dist = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)  # (F,)
    scale = float(dist[pose_valid].mean()) if pose_valid.any() else 1.0
    if scale < 1e-6:
        scale = 1.0

    # A missing group is all zeros, so its origin is zero and it stays zero.
    origin = np.empty_like(xyz)
    origin[:, POSE_SLICE] = shoulders.mean(axis=1, keepdims=True)  # shoulder midpoint
    for hand in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        origin[:, hand] = xyz[:, hand][:, PALM].mean(axis=1, keepdims=True)  # palm center
    seq[:, :, :3] = (xyz - origin) / scale
    return seq


@dataclasses.dataclass
class AugmentConfig:
    """Train-time augmentation ranges, applied to raw world coordinates."""
    crop: float = 0.15          # keep a random (1 - crop)..100% sub-window of the frames
    speed: float = 0.3          # signing speed factor, log-uniform in [1/(1+s), 1+s]
    yaw: float = 20.0           # camera rotation about the vertical axis, degrees
    pitch: float = 10.0         # ... about the left-right axis
    roll: float = 8.0           # ... in the image plane
    body_scale: float = 0.08    # independent pose x / y scale in [1-s, 1+s] (proportions)
    pose_depth: float = 0.3     # pose z scale in [1-s, 1+s]: monocular depth is unreliable
    hand_scale: float = 0.12    # hand size relative to the body
    pose_bias: float = 0.015    # per-clip offset of each pose landmark, meters
    pose_jitter: float = 0.008  # per-frame pose noise, meters
    pose_drift: float = 0.0     # smooth (~0.1-0.2 s correlated) pose noise, meters: how two
                                # MediaPipe versions disagree within a clip
    hand_jitter: float = 0.003  # per-frame hand-shape noise, meters
    hand_dropout: float = 0.4   # chance per hand of a missed-detection gap (10-40% of frames)
    hand_edge: float = 0.0      # move each hand-detection span's start/end by up to this
                                # many frames (trackers pick up / drop a moving hand at
                                # different moments; extensions hold the edge frame's hand)

    def scaled(self, k: float) -> "AugmentConfig":
        """Every range multiplied by k (k=0 disables augmentation)."""
        return AugmentConfig(**{f.name: getattr(self, f.name) * k
                                for f in dataclasses.fields(self)})


def _rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """3x3 rotation (radians) about the x (left-right), y (vertical) and z (depth) axes."""
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cz, sz = np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float32)


def _smooth_noise(rng: np.random.Generator, shape, sigma_frames: float) -> np.ndarray:
    """Unit-variance noise of `shape` (frames first), Gaussian-correlated over time."""
    t = np.arange(shape[0])
    kernel = np.exp(-0.5 * ((t[:, None] - t[None, :]) / sigma_frames) ** 2)
    kernel /= np.sqrt((kernel ** 2).sum(axis=1, keepdims=True))
    white = rng.normal(size=(shape[0], int(np.prod(shape[1:]))))
    return (kernel @ white).reshape(shape).astype(np.float32)


def _jitter_span_edges(seq: np.ndarray, hand: slice, max_shift: int,
                       rng: np.random.Generator) -> None:
    """In place: move each detection span's start and end of one hand by up to
    max_shift frames, never into a neighbouring span. Growing repeats the edge
    frame's hand; shrinking zero-fills."""
    on = detected(seq[:, hand, :3]).astype(np.int8)
    edges = np.diff(np.r_[0, on, 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    original = seq[:, hand].copy()
    for k, (s, e) in enumerate(zip(starts, ends)):
        lo = ends[k - 1] + 2 if k > 0 else 0  # keep >= 1 empty frame between spans
        hi = starts[k + 1] - 2 if k + 1 < len(starts) else len(seq) - 1
        new_s = int(np.clip(s + rng.integers(-max_shift, max_shift + 1), lo, e))
        new_e = int(np.clip(e + rng.integers(-max_shift, max_shift + 1), new_s, hi))
        seq[new_s:s, hand] = original[s]
        seq[s:new_s, hand] = 0.0
        seq[e + 1:new_e + 1, hand] = original[e]
        seq[new_e + 1:e + 1, hand] = 0.0


def augment(seq: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig | None = None) -> np.ndarray:
    """Random signer / camera / extractor variation of a raw (frames, 75, 5) sequence.

    Temporal crop and speed change, camera rotation, body and hand proportions,
    unreliable pose depth, per-clip pose bias plus per-frame noise, and gaps in
    hand detection. Apply BEFORE normalize(), which removes global position and
    scale, so only the relative changes survive.
    """
    cfg = cfg or AugmentConfig()
    n_frames = len(seq)

    # temporal crop, then resample to a new speed (nearest frame keeps zero-fills exact)
    keep = max(2, int(round(n_frames * rng.uniform(1.0 - cfg.crop, 1.0))))
    start = rng.integers(0, n_frames - keep + 1)
    speed = np.exp(rng.uniform(-1.0, 1.0) * np.log1p(cfg.speed))
    n_out = max(2, int(round(keep / speed)))
    seq = seq[start + np.linspace(0, keep - 1, n_out).round().astype(int)]  # copy

    xyz = np.ascontiguousarray(seq[:, :, :3])
    present = np.abs(xyz).sum(axis=2, keepdims=True) > 0  # (F, 75, 1)
    pose, hands = xyz[:, POSE_SLICE], xyz[:, HANDS_SLICE]

    pose *= np.array([rng.uniform(1 - cfg.body_scale, 1 + cfg.body_scale),
                      rng.uniform(1 - cfg.body_scale, 1 + cfg.body_scale),
                      rng.uniform(1 - cfg.pose_depth, 1 + cfg.pose_depth)], dtype=np.float32)
    hands *= rng.uniform(1 - cfg.hand_scale, 1 + cfg.hand_scale)
    angles = np.deg2rad([rng.uniform(-cfg.pitch, cfg.pitch), rng.uniform(-cfg.yaw, cfg.yaw),
                         rng.uniform(-cfg.roll, cfg.roll)])
    xyz = (xyz.reshape(-1, 3) @ _rotation(*angles).T).reshape(xyz.shape)
    pose, hands = xyz[:, POSE_SLICE], xyz[:, HANDS_SLICE]

    pose += rng.normal(0.0, cfg.pose_bias, size=(1, *pose.shape[1:])).astype(np.float32)
    pose += rng.normal(0.0, cfg.pose_jitter, size=pose.shape).astype(np.float32)
    if cfg.pose_drift > 0:
        pose += cfg.pose_drift * _smooth_noise(rng, pose.shape, DRIFT_FRAMES)
    hands += rng.normal(0.0, cfg.hand_jitter, size=hands.shape).astype(np.float32)
    seq[:, :, :3] = xyz * present  # keep missing landmarks exactly zero

    max_shift = int(round(cfg.hand_edge))
    for hand in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        if max_shift > 0:
            _jitter_span_edges(seq, hand, max_shift, rng)
        if rng.random() < cfg.hand_dropout:
            gap = max(1, int(round(n_out * rng.uniform(0.1, 0.4))))
            gap_start = rng.integers(0, n_out - gap + 1)
            seq[gap_start:gap_start + gap, hand] = 0.0
    return seq


def fit_length(seq: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Uniformly resample longer sequences, zero-pad shorter ones -> (max_frames, ...)."""
    n_frames = len(seq)
    if n_frames > max_frames:
        idx = np.linspace(0, n_frames - 1, max_frames).round().astype(int)
        return seq[idx]
    if n_frames < max_frames:
        pad = np.zeros((max_frames - n_frames, *seq.shape[1:]), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)
    return seq


def feature_dim(velocity: bool = True, include_confidence: bool = False) -> int:
    n_positions = N_POSE_FEATURES * 2 + (len(FEATURE_LANDMARKS) - N_POSE_FEATURES) * 3  # 172
    return n_positions * (2 if velocity else 1) + (len(FEATURE_LANDMARKS) * 2 if include_confidence else 0)


def _without_pose_depth(block: np.ndarray) -> np.ndarray:
    """(frames, 65, 3) -> (frames, 172): pose x/y + hand x/y/z.

    Monocular pose depth is model-specific: the dataset's MediaPipe and the live
    app's put e.g. the face ~9 cm apart in z (9x its natural motion) while x/y
    agree within a few cm. Dropping pose z raised live-pipeline (replay)
    accuracy from 85% to 93%; hand z (a hand-shape model) agrees within ~4 mm.
    """
    return np.concatenate([block[:, :N_POSE_FEATURES, :2].reshape(len(block), -1),
                           block[:, N_POSE_FEATURES:].reshape(len(block), -1)], axis=1)


def to_features(seq: np.ndarray, velocity: bool = True,
                include_confidence: bool = False) -> np.ndarray:
    """Normalized (frames, 75, 5) -> (frames, feature_dim) over the 65 feature landmarks.

    Pose contributes x/y only (see _without_pose_depth), hands x/y/z. Hand
    shapes are scaled up by HAND_FEATURE_SCALE to the same spread as the pose
    (conditioning only). Velocity is the per-frame displacement of each landmark
    group, zero where the group is missing in either frame (a hand appearing is
    not motion).
    """
    kept = seq[:, FEATURE_LANDMARKS]
    xyz = kept[:, :, :3]
    parts = [_without_pose_depth(xyz * _POSITION_SCALE)]
    if velocity:
        group_sums = np.add.reduceat(np.abs(xyz).sum(axis=2), _GROUP_STARTS, axis=1)
        present = (group_sums > 0)[:, _GROUP_OF_LANDMARK]  # (F, 65)
        vel = np.zeros_like(xyz)
        vel[1:] = (xyz[1:] - xyz[:-1]) * (present[1:] & present[:-1])[:, :, None]
        parts.append(_without_pose_depth(vel * VELOCITY_SCALE))
    if include_confidence:
        parts.append(kept[:, :, 3:].reshape(len(seq), -1))
    return np.concatenate(parts, axis=1)


def preprocess(seq: np.ndarray, max_frames: int = MAX_FRAMES, velocity: bool = True,
               include_confidence: bool = False, rng: np.random.Generator | None = None,
               augment_cfg: AugmentConfig | None = None) -> np.ndarray:
    """Raw (frames, 75, 5) landmarks -> (max_frames, feature_dim) model input.

    The one feature pipeline used by training, evaluation and the live app;
    pass rng to augment (training only).
    """
    seq = trim_idle(seq)
    if rng is not None:
        seq = augment(seq, rng, augment_cfg)
    features = to_features(normalize(seq), velocity, include_confidence)
    return fit_length(features, max_frames).astype(np.float32)


RECORDING_HOLDOUT = 2   # last takes of each word kept out of training (--split recordings)
RECORDING_VARIANTS = 4  # training rows per take per epoch: 1 original + 3 augmented


def recording_index(recordings_dir=RECORDINGS_DIR, holdout: int = RECORDING_HOLDOUT,
                    variants: int = RECORDING_VARIANTS):
    """Your own takes (src/record.py) as split-index rows -> (train, held_out) DataFrames.

    Layout <recordings_dir>/<label>/take-NN.csv; `file` holds absolute paths, so
    PSLDataset reads them wherever its landmarks_dir points. Words with at least
    holdout + 3 takes keep their last `holdout` takes out of training, for an
    honest "does it recognise me" check. Unknown labels are skipped.
    """
    with open(SPLITS_DIR / "label_map.json", encoding="utf-8") as f:
        label_ids = {label: info["id"] for label, info in json.load(f).items()}
    train, held = [], []
    for word_dir in sorted(p for p in Path(recordings_dir).glob("*") if p.is_dir()):
        if word_dir.name not in label_ids:
            continue
        cid = label_ids[word_dir.name]
        takes = sorted(word_dir.glob("take-*.csv"))
        n_held = holdout if len(takes) >= holdout + 3 else 0
        for t, take in enumerate(takes):
            row = (str(take.resolve()), word_dir.name, cid)
            if t >= len(takes) - n_held:
                held.append((*row, -1))
                continue
            train.append((*row, -1))
            train.extend((*row, 10 ** 8 + cid * 10 ** 4 + t * 10 + v) for v in range(1, variants))
    columns = ["file", "label", "label_id", "aug_seed"]
    return pd.DataFrame(train, columns=columns), pd.DataFrame(held, columns=columns)


class PSLDataset:
    """Sequence dataset over a split index file produced by src/data/split.py.

    Index rows: file,label,label_id,aug_seed (aug_seed -1 = original, no augmentation).
    Augmented rows use their fixed aug_seed (val/test), or a fresh draw per
    epoch when an epoch is passed (training: every epoch sees new variants).
    Framework-agnostic: __getitem__ returns (features, label_id) numpy pairs and
    as_arrays() materializes (X, y) for e.g. keras Model.fit.
    """

    def __init__(self, index_file, max_frames: int = MAX_FRAMES, velocity: bool = True,
                 include_confidence: bool = False, landmarks_dir=LANDMARKS_DIR,
                 augment_cfg: AugmentConfig | None = None):
        self.index = pd.read_csv(index_file)
        self.max_frames = max_frames
        self.velocity = velocity
        self.include_confidence = include_confidence
        self.landmarks_dir = Path(landmarks_dir)
        self.augment_cfg = augment_cfg or AugmentConfig()
        self.num_classes = int(self.index["label_id"].max()) + 1
        self._rows = list(zip(self.index["file"], self.index["label_id"].astype(int),
                              self.index["aug_seed"].astype(int)))
        self._cache: dict = {}  # raw parsed CSVs, keyed by filename

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def feature_dim(self) -> int:
        return feature_dim(self.velocity, self.include_confidence)

    def _raw(self, filename: str) -> np.ndarray:
        if filename not in self._cache:
            self._cache[filename] = parse_csv(self.landmarks_dir / filename)
        return self._cache[filename]

    def item(self, i: int, epoch: int | None = None):
        filename, label_id, aug_seed = self._rows[i]
        rng = None
        if aug_seed >= 0:
            rng = np.random.default_rng(aug_seed if epoch is None else (aug_seed, epoch))
        features = preprocess(self._raw(filename), self.max_frames, self.velocity,
                              self.include_confidence, rng, self.augment_cfg)
        return features, label_id

    def __getitem__(self, i: int):
        return self.item(i)

    def as_arrays(self, epoch: int | None = None):
        """Materialize the whole split: X (N, max_frames, feat), y (N,)."""
        xs, ys = zip(*(self.item(i, epoch) for i in range(len(self))))
        return np.stack(xs), np.array(ys, dtype=np.int64)
