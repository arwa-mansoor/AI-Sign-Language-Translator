"""Visual sanity checks for the Phase 1 data pipeline.

Saves PNGs into notebooks/figures/:
  frame_count_histogram.png   distribution of clip lengths + MAX_FRAMES line
  trajectories_<label>.png    raw vs normalized wrist trajectories (3 samples)
  skeleton_<label>.png        pose + hand skeletons at 4 timesteps (1 sample)

Run:  python src/visualize_samples.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save files, no window
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import (  # noqa: E402
    LANDMARKS_DIR, MAX_FRAMES, REPO_ROOT, RIGHT_HAND_SLICE,
    label_from_path, normalize, parse_csv,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIGURES_DIR = REPO_ROOT / "notebooks" / "figures"
LEFT_WRIST, RIGHT_WRIST = 15, 16  # pose landmark indices

# minimal upper-body pose skeleton for plotting
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

SAMPLE_FILES = [
    "pk-hfad-1_book.landmarks-mediapipe-world.csv",
    "pk-hfad-1_airplane.landmarks-mediapipe-world.csv",
    "pk-hfad-1_apple.landmarks-mediapipe-world.csv",
]


def plot_frame_histogram() -> Path:
    counts = []
    for f in sorted(LANDMARKS_DIR.glob("*.csv")):
        with open(f, "rb") as fh:
            counts.append(sum(1 for _ in fh) - 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(MAX_FRAMES, color="red", linestyle="--", label=f"MAX_FRAMES = {MAX_FRAMES}")
    ax.set_xlabel("frames per clip")
    ax.set_ylabel("number of clips")
    ax.set_title(f"Clip length distribution ({len(counts)} files, "
                 f"median {int(np.median(counts))}, p95 {int(np.percentile(counts, 95))})")
    ax.legend()
    out = FIGURES_DIR / "frame_count_histogram.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_trajectories(csv_path: Path) -> Path:
    label = label_from_path(csv_path)
    raw = parse_csv(csv_path)
    norm = normalize(raw)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for col, (seq, title) in enumerate([(raw, "raw (meters)"), (norm, "normalized")]):
        for row, wrist, name in [(0, LEFT_WRIST, "left wrist"), (1, RIGHT_WRIST, "right wrist")]:
            ax = axes[row][col]
            for axis, axis_name in enumerate("xyz"):
                ax.plot(seq[:, wrist, axis], label=axis_name)
            ax.set_title(f"{name} — {title}")
            ax.set_xlabel("frame")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{label}  ({len(raw)} frames)")
    fig.tight_layout()
    out = FIGURES_DIR / f"trajectories_{label.replace('pk-hfad-1_', '')}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_skeleton(csv_path: Path) -> Path:
    label = label_from_path(csv_path)
    norm = normalize(parse_csv(csv_path))
    timesteps = np.linspace(0, len(norm) - 1, 4).round().astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharex=True, sharey=True)
    for ax, t in zip(axes, timesteps):
        frame = norm[t]
        pose_xy = frame[:33, :2]
        # pose: world y grows downward -> plot -y so up is up
        for a, b in POSE_CONNECTIONS:
            ax.plot(pose_xy[[a, b], 0], -pose_xy[[a, b], 1], "b-", lw=1.5)
        ax.plot(pose_xy[:, 0], -pose_xy[:, 1], "bo", ms=2)
        # right hand (hand-centered coords): attach at the pose right wrist for display
        hand = frame[RIGHT_HAND_SLICE, :2]
        if np.abs(hand).sum() > 0:
            offset = pose_xy[RIGHT_WRIST] - hand[0]
            hand = hand + offset
            for a, b in HAND_CONNECTIONS:
                ax.plot(hand[[a, b], 0], -hand[[a, b], 1], "g-", lw=1)
            ax.plot(hand[:, 0], -hand[:, 1], "g.", ms=2)
        ax.set_title(f"frame {t}")
        ax.set_aspect("equal")
    fig.suptitle(f"{label} — normalized pose (blue) + right hand (green)")
    fig.tight_layout()
    out = FIGURES_DIR / f"skeleton_{label.replace('pk-hfad-1_', '')}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [plot_frame_histogram()]

    samples = [LANDMARKS_DIR / name for name in SAMPLE_FILES]
    samples = [p for p in samples if p.is_file()]
    if len(samples) < 3:  # fall back to first files if any named sample is missing
        samples = sorted(LANDMARKS_DIR.glob("*.csv"))[:3]

    for path in samples:
        outputs.append(plot_trajectories(path))
    outputs.append(plot_skeleton(samples[0]))

    print("Saved figures:")
    for out in outputs:
        print(f"  {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
