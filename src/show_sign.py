"""Render a sign's landmark CSV back into an animated stick figure.

Use it to see exactly what motion the model was trained on, then mimic it in
front of the webcam.  Writes GIFs to notebooks/figures/.

Run:  python src/show_sign.py house book milk
      python src/show_sign.py --list          # signs in the demo20 vocabulary
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import (  # noqa: E402
    LANDMARKS_DIR, REPO_ROOT, FILENAME_SUFFIX,
    LEFT_HAND_SLICE, RIGHT_HAND_SLICE, normalize, parse_csv,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIGURES_DIR = REPO_ROOT / "notebooks" / "figures"
VOCAB_FILE = REPO_ROOT / "models" / "vocab-demo20.txt"

POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 23), (12, 24), (23, 24), (9, 10)]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
LEFT_WRIST, RIGHT_WRIST = 15, 16


def render(label: str) -> Path | None:
    csv_path = LANDMARKS_DIR / f"{label}{FILENAME_SUFFIX}.csv"
    if not csv_path.is_file():
        print(f"  no CSV for {label}")
        return None

    seq = normalize(parse_csv(csv_path))
    fig, ax = plt.subplots(figsize=(4.5, 5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.2, 2.2)
    title = ax.set_title("")

    pose_lines = [ax.plot([], [], "-", color="#1f77b4", lw=2.5)[0] for _ in POSE_CONNECTIONS]
    hand_lines, hand_slices = [], []
    for slc, colour in ((LEFT_HAND_SLICE, "#ff7f0e"), (RIGHT_HAND_SLICE, "#2ca02c")):
        hand_slices.append(slc)
        hand_lines.append([ax.plot([], [], "-", color=colour, lw=1.4)[0]
                           for _ in HAND_CONNECTIONS])

    def update(t):
        frame = seq[t]
        pose = frame[:33, :2]
        for line, (a, b) in zip(pose_lines, POSE_CONNECTIONS):
            line.set_data(pose[[a, b], 0], -pose[[a, b], 1])
        for lines, slc, wrist in zip(hand_lines, hand_slices, (LEFT_WRIST, RIGHT_WRIST)):
            hand = frame[slc, :2]
            visible = np.abs(hand).sum() > 0
            if visible:  # hands are hand-centred; pin them to the pose wrist
                hand = hand - hand[0] + pose[wrist]
            for line, (a, b) in zip(lines, HAND_CONNECTIONS):
                line.set_data(hand[[a, b], 0], -hand[[a, b], 1]) if visible \
                    else line.set_data([], [])
        title.set_text(f"{label.replace('pk-hfad-1_', '')}   frame {t + 1}/{len(seq)}")
        return []

    anim = FuncAnimation(fig, update, frames=len(seq), interval=50, blit=False)
    out = FIGURES_DIR / f"sign_{label.replace('pk-hfad-1_', '')}.gif"
    anim.save(out, writer=PillowWriter(fps=20))
    plt.close(fig)
    return out


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    vocab = [l.strip() for l in open(VOCAB_FILE, encoding="utf-8") if l.strip()]

    args = sys.argv[1:]
    if not args or args[0] == "--list":
        print("demo20 vocabulary:")
        for label in vocab:
            print(f"  {label.replace('pk-hfad-1_', '')}")
        print("\nRender one:  python src/show_sign.py house book milk")
        return 0

    for name in args:
        label = name if name.startswith("pk-hfad-1_") else f"pk-hfad-1_{name}"
        out = render(label)
        if out:
            print(f"  {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
