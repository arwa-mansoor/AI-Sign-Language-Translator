"""Sanity check for the pk-hfad-1 landmark dataset.

Loads 3 random landmark CSVs, confirms each has exactly 375 columns
(75 landmarks x 5 values: pose 0-32, left hand 33-53, right hand 54-74),
prints the shape, and confirms the label exists in the dictionary mapping.

Run from the repo root:
    python src/sanity_check.py
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252 which cannot print Urdu tokens.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = REPO_ROOT / "data" / "landmarks" / "pk-hfad-1.landmarks-mediapipe-world-csv"
MAPPING_FILE = REPO_ROOT / "data" / "mappings" / "pk-dictionary-mapping.json"

EXPECTED_COLUMNS = 375  # 75 landmarks * 5 values (x, y, z, visibility, presence)
FILENAME_SUFFIX = ".landmarks-mediapipe-world"

N_SAMPLES = 3


def load_label_to_tokens(mapping_file: Path) -> dict:
    """Return {label: {"en": [...], "ur": [...], ...}} from all dictionaries in the mapping json."""
    with open(mapping_file, encoding="utf-8") as f:
        dictionaries = json.load(f)

    label_to_tokens = {}
    for dictionary in dictionaries:
        for entry in dictionary["mapping"]:
            # Constructable sign-phrase entries have "components" but no "label".
            if "label" in entry:
                label_to_tokens[entry["label"]] = entry["token"]
    return label_to_tokens


def check_file(csv_path: Path, label_to_tokens: dict) -> bool:
    """Validate one landmark CSV. Returns True if all checks pass."""
    label = csv_path.stem.replace(FILENAME_SUFFIX, "")
    print(f"\nFile:  {csv_path.name}")
    print(f"Label: {label}")

    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}  ({df.shape[0]} frames x {df.shape[1]} columns)")

    ok = True
    if df.shape[1] == EXPECTED_COLUMNS:
        print(f"[OK]   Column count is {EXPECTED_COLUMNS}")
    else:
        print(f"[FAIL] Expected {EXPECTED_COLUMNS} columns, found {df.shape[1]}")
        ok = False

    if df.isna().any().any():
        print("[FAIL] CSV contains NaN values")
        ok = False
    else:
        print("[OK]   No NaN values")

    tokens = label_to_tokens.get(label)
    if tokens is not None:
        en = ", ".join(tokens.get("en", [])[:3]) or "-"
        ur = ", ".join(tokens.get("ur", [])[:3]) or "-"
        print(f"[OK]   Label found in dictionary mapping  (en: {en} | ur: {ur})")
    else:
        # 13 files are compound sign-phrases readable only from the filename gloss.
        print("[WARN] Label not in dictionary mapping (compound sign-phrase, expected for 13 files)")

    return ok


def main() -> int:
    if not LANDMARKS_DIR.is_dir():
        print(f"[FAIL] Landmarks directory not found: {LANDMARKS_DIR}")
        return 1
    if not MAPPING_FILE.is_file():
        print(f"[FAIL] Mapping file not found: {MAPPING_FILE}")
        return 1

    csv_files = sorted(LANDMARKS_DIR.glob("*.csv"))
    label_to_tokens = load_label_to_tokens(MAPPING_FILE)

    print(f"Landmark CSVs found:        {len(csv_files)}")
    print(f"Dictionary mapping labels:  {len(label_to_tokens)}")
    print(f"Checking {N_SAMPLES} random files...")

    samples = random.sample(csv_files, N_SAMPLES)
    all_ok = all([check_file(path, label_to_tokens) for path in samples])

    print("\n" + ("Sanity check PASSED" if all_ok else "Sanity check FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
