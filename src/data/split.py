"""Build reproducible stratified train/val/test splits for the PSL dataset.

Reality of pk-hfad-1: exactly ONE video per sign (775 usable classes). A plain
stratified split is impossible (a val/test class would never appear in
training). Instead each class is expanded into VARIANTS_PER_CLASS deterministic
variants (the original + augmented copies, identified by aug_seed) and the
variants are stratified 8/2/2 per class => ~67/17/17, close to 70/15/15.
The un-augmented original always stays in train.

Caveat (v1): val/test contain augmentations of the same source videos as
train, so they measure robustness to perturbation rather than
person-generalization. True generalization needs new recordings (Phase 5+).

Exclusions (reported at runtime):
  - 13 compound sign-phrase CSVs with no dictionary entry -> excluded from v1.
  - dictionary label pk-hfad-2_hour has no CSV -> ignored.

Outputs (data/splits/):
  train.csv / val.csv / test.csv   columns: file,label,label_id,aug_seed
  label_map.json                   {label: {"id", "en", "ur"}}
  excluded.txt                     labels excluded from v1

Run:  python src/data/split.py
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from data.dataset import (  # noqa: E402
    LANDMARKS_DIR, SPLITS_DIR, label_from_path, load_label_to_tokens,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 42
VARIANTS_PER_CLASS = 12  # 1 original + 11 augmented
N_TRAIN, N_VAL, N_TEST = 8, 2, 2  # per class (train includes the original)


def main() -> int:
    csv_files = sorted(LANDMARKS_DIR.glob("*.csv"))
    label_to_tokens = load_label_to_tokens()

    included, excluded = [], []
    for path in csv_files:
        label = label_from_path(path)
        (included if label in label_to_tokens else excluded).append((path.name, label))

    print(f"CSV files:                {len(csv_files)}")
    print(f"Usable classes (v1):      {len(included)}")
    print(f"Excluded compound files:  {len(excluded)}")

    labels = sorted(label for _, label in included)
    label_map = {
        label: {
            "id": i,
            "en": label_to_tokens[label].get("en", []),
            "ur": label_to_tokens[label].get("ur", []),
        }
        for i, label in enumerate(labels)
    }

    rng = np.random.default_rng(SEED)
    rows = {"train": [], "val": [], "test": []}
    for filename, label in included:
        label_id = label_map[label]["id"]
        # globally unique deterministic augmentation seeds for this class
        seeds = [label_id * 1000 + v for v in range(1, VARIANTS_PER_CLASS)]
        rng.shuffle(seeds)
        rows["train"].append((filename, label, label_id, -1))  # original, unaugmented
        for seed in seeds[: N_TRAIN - 1]:
            rows["train"].append((filename, label, label_id, seed))
        for seed in seeds[N_TRAIN - 1: N_TRAIN - 1 + N_VAL]:
            rows["val"].append((filename, label, label_id, seed))
        for seed in seeds[N_TRAIN - 1 + N_VAL: N_TRAIN - 1 + N_VAL + N_TEST]:
            rows["test"].append((filename, label, label_id, seed))

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["file", "label", "label_id", "aug_seed"]
    for split, split_rows in rows.items():
        df = pd.DataFrame(split_rows, columns=columns)
        df.to_csv(SPLITS_DIR / f"{split}.csv", index=False)
        print(f"{split:5s}: {len(df):5d} samples ({len(df) / sum(map(len, rows.values())):.1%})")

    with open(SPLITS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=1)
    with open(SPLITS_DIR / "excluded.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{label}\n" for _, label in excluded)

    print(f"\nWrote {SPLITS_DIR}\\train.csv, val.csv, test.csv, label_map.json, excluded.txt")
    print(f"Classes: {len(labels)}, variants/class: {VARIANTS_PER_CLASS} "
          f"(train {N_TRAIN} / val {N_VAL} / test {N_TEST})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
