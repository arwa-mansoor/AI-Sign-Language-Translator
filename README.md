# AI Sign Language Translator — Pakistan Sign Language (PSL)

Real-time sign language translation: **Webcam → PSL → English / Urdu text**.

Webcam frames are converted by MediaPipe into the same 3D landmark format used in the
`pk-hfad-1` dataset, a sequence model (LSTM/Transformer) predicts the PSL sign, and the
predicted label is mapped to English and Urdu words using the dictionary mapping.

## Pipeline

**Phase 1 — train the recognition model (offline, from the pre-extracted landmark dataset):**

```
pk-hfad-1 landmark CSVs (788 sign videos)
        +
data/mappings/pk-dictionary-mapping.json
        ↓
preprocess (normalize, pad/trim sequences, encode labels)
        ↓
train sequence model
        ↓
[landmark sequence] → ML model → PSL sign → English / Urdu text
```

**Phase 2 — real-time webcam translation:**

```
Webcam
   ↓
OpenCV captures frames
   ↓
MediaPipe extracts pose + hand landmarks (same format as training data)
   ↓
trained sequence model
   ↓
predicted PSL sign
   ↓
dictionary mapping → English + Urdu text
```

## Project structure

```
AI-Sign-Language-Translator/
│
├── data/
│   ├── landmarks/
│   │   └── pk-hfad-1.landmarks-mediapipe-world-csv/   # 788 landmark CSVs (~110 MB, gitignored)
│   └── mappings/
│       └── pk-dictionary-mapping.json                 # label → English/Urdu word mapping
│
├── notebooks/      # exploration & training notebooks
├── src/            # preprocessing, model, webcam app
├── models/         # saved trained models (weights gitignored)
├── requirements.txt
└── README.md
```

## Dataset

All data comes from
[sign-language-translator/sign-language-datasets](https://github.com/sign-language-translator/sign-language-datasets):
Pakistan Sign Language videos taught at the **Hamza Foundation Academy for the Deaf,
Lahore, Pakistan**, pre-extracted into MediaPipe landmark CSVs. The raw videos are **not**
needed — the dataset repository publishes the landmark archives separately.

### What is included here

| Path | Contents | Size | Source |
|---|---|---|---|
| `data/landmarks/pk-hfad-1.landmarks-mediapipe-world-csv/` | 788 CSV files, one per sign video | ~110 MB | [release v0.0.4](https://github.com/sign-language-translator/sign-language-datasets/releases/tag/v0.0.4) |
| `data/mappings/pk-dictionary-mapping.json` | sign label → word translations (en, ur, hi, ...) | ~161 KB | [`parallel_texts/`](https://github.com/sign-language-translator/sign-language-datasets/tree/main/parallel_texts) |

To reproduce the downloads (PowerShell):

```powershell
# 1) landmark dataset (23 MB zip → extracts to 788 CSVs)
Invoke-WebRequest "https://github.com/sign-language-translator/sign-language-datasets/releases/download/v0.0.4/pk-hfad-1.landmarks-mediapipe-world-csv.zip" -OutFile "data\landmarks\pk-hfad-1.landmarks-mediapipe-world-csv.zip"
Expand-Archive "data\landmarks\pk-hfad-1.landmarks-mediapipe-world-csv.zip" -DestinationPath "data\landmarks\pk-hfad-1.landmarks-mediapipe-world-csv"

# 2) dictionary mapping
Invoke-WebRequest "https://raw.githubusercontent.com/sign-language-translator/sign-language-datasets/main/parallel_texts/pk-dictionary-mapping.json" -OutFile "data\mappings\pk-dictionary-mapping.json"
```

### What is intentionally NOT downloaded

| File | Why not |
|---|---|
| `pk-hfad-1.landmarks-mediapipe-image-csv.zip` | Image-space coordinates (fractions of frame width/height) instead of world coordinates (meters). Don't mix coordinate systems — stick to `mediapipe-world`. |
| `pk-hfad-1.videos-mp4.zip` (release v0.0.3) | Raw videos. Landmarks are already extracted, so re-running MediaPipe over thousands of frames is unnecessary. Only needed later if you re-extract with a different model. |

### Landmark CSV format

Each CSV is one sign video; each row is one video frame; every file has exactly
**375 columns** = 75 landmarks × 5 values:

| Landmark group | Indices | Values per landmark |
|---|---|---|
| Pose (MediaPipe Pose) | 0–32 | `x, y, z, a, b` |
| Left hand (MediaPipe Hands) | 33–53 | `x, y, z, a, b` |
| Right hand (MediaPipe Hands) | 54–74 | `x, y, z, a, b` |

- `a` = visibility, `b` = presence (confidence scores)
- World coordinates are 3D joint positions in **meters**, rounded to 4 decimal places
- Missing pose/hands in a frame are zero-filled
- Header: `x0,y0,z0,a0,b0,x1,y1,z1,a1,b1,...,x74,y74,z74,a74,b74`
- Row count = number of frames in the source video

The **label** of a sample is its filename with the format suffix removed:

```
pk-hfad-1_1.landmarks-mediapipe-world.csv  →  label: pk-hfad-1_1
```

### Dictionary mapping format

`data/mappings/pk-dictionary-mapping.json` is a JSON array of two dictionaries — the
pk-hfad-1 standard dictionary (776 entries) and a small list of constructable
sign-phrases (18 entries):

```json
[
  {
    "country": "pk",
    "description": "pakistan sign language videos taught in hamza foundation academy for the deaf,lahore, pakistan.",
    "mapping": [
      {
        "label": "pk-hfad-1_1",
        "token": {
          "en": ["1", "one", "a(article)", "an"],
          "ur": ["1", "۱", "ایک", "اک", "اِک", "یکم"]
        }
      }
    ]
  }
]
```

Compound signs additionally list the component signs:

```json
{
  "components": ["pk-hfad-1_a(double-handed-letter)", "pk-hfad-1_market"],
  "label": "pk-hfad-1_advertisement",
  "token": {
    "en": ["advertisement", "advertising", "advertise"],
    "ur": ["اشتہار", "تشہیر"]
  }
}
```

### Verified dataset statistics

- 788 landmark CSVs in `pk-hfad-1`, all with exactly 375 columns
- 775 CSVs have a direct entry in the dictionary mapping (usable as labeled classes)
- 13 CSVs are compound sign-phrases (e.g. `pk-hfad-1_hm(we)-aaein`) with no individual
  dictionary entry — their meaning is still readable from the filename gloss
- The dictionary's 776 entries include one label (`pk-hfad-2_hour`) that belongs to the
  separate pk-hfad-2 dataset and has no CSV here
- Word tokens per official stats: en 1,591 / ur 2,081

## Setup

Python 3.9–3.12 is recommended (MediaPipe wheel availability).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Next steps

1. **Explore** — notebook: load a few CSVs, check shapes, plot landmarks over frames,
   join labels with the dictionary mapping.
2. **Preprocess** — normalize landmarks (center/scale), pad or trim sequences to a fixed
   length, integer-encode the labels.
3. **Train** — LSTM (or Transformer) classifier: `(frames, 375)` → sign label, with a
   train/val/test split.
4. **Export** — save the trained model into `models/`.
5. **Webcam app** — OpenCV capture + MediaPipe extraction in the same
   pose → left hand → right hand order, sliding-window prediction over frames, and
   label → English/Urdu text output.

## License & attribution

The dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — appropriate credit and
citation of the original creators are required:

- Dataset: [sign-language-translator/sign-language-datasets](https://github.com/sign-language-translator/sign-language-datasets)
- Videos: Hamza Foundation Academy for the Deaf, Lahore, Pakistan
