"""Generate the spoken Urdu the live app plays for each recognised sign.

Windows has no Urdu text-to-speech voice, so the Urdu word of every sign is
synthesised once with a Microsoft neural voice (Pakistani Urdu, via the
edge-tts package — needs internet) and saved as data/audio/ur/<label>.mp3
(gitignored). src/app.py then plays these files offline, after speaking the
English word with the Windows voice.

Run:  python src/make_urdu_audio.py                    # the demo vocabulary
      python src/make_urdu_audio.py --run demo20-me    # a run's vocabulary
      python src/make_urdu_audio.py --all              # all 775 signs (~5 min)
      python src/make_urdu_audio.py --voice ur-PK-UzmaNeural --force   # female voice
Voices: ur-PK-AsadNeural (male, default), ur-PK-UzmaNeural (female).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import REPO_ROOT, SPLITS_DIR, URDU_AUDIO_DIR, display_text  # noqa: E402

DEFAULT_VOCAB = REPO_ROOT / "models" / "vocab-demo20.txt"
MODELS_DIR = REPO_ROOT / "models"


def labels_to_voice(args, label_map) -> list[str]:
    if args.all:
        return sorted(label_map, key=lambda label: label_map[label]["id"])
    if args.run:
        vocab_file = MODELS_DIR / f"{args.run}.vocab.json"
        if not vocab_file.is_file():
            sys.exit(f"{vocab_file} not found (runs without --classes-file use all signs: --all)")
        return json.loads(vocab_file.read_text(encoding="utf-8"))
    vocab = Path(args.vocab)
    if not vocab.is_file():
        sys.exit(f"{vocab} not found; pass --vocab, --run or --all")
    return [line.strip() for line in vocab.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


async def synthesise(items, voice: str) -> None:
    import edge_tts
    for i, (label, text, path) in enumerate(items, 1):
        await edge_tts.Communicate(text, voice).save(str(path))
        print(f"  {i}/{len(items)}  {label.replace('pk-hfad-1_', ''):<16} {text}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB), help="file with one label per line")
    parser.add_argument("--run", help="use this run's vocabulary (models/<run>.vocab.json)")
    parser.add_argument("--all", action="store_true", help="every sign in the dataset")
    parser.add_argument("--voice", default="ur-PK-AsadNeural", help="edge-tts voice name")
    parser.add_argument("--force", action="store_true", help="regenerate existing files")
    args = parser.parse_args()
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        sys.exit("needs the edge-tts package: pip install edge-tts")

    with open(SPLITS_DIR / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    URDU_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    items, no_urdu = [], []
    for label in labels_to_voice(args, label_map):
        text = display_text(label, "ur", label_map.get(label, {}).get("ur", []))
        path = URDU_AUDIO_DIR / f"{label}.mp3"
        if not text:
            no_urdu.append(label)
        elif args.force or not path.is_file():
            items.append((label, text, path))
    print(f"generating {len(items)} Urdu clips with {args.voice} -> {URDU_AUDIO_DIR}")
    try:
        asyncio.run(synthesise(items, args.voice))
    except Exception as e:  # network down, service change, ...
        sys.exit(f"synthesis failed ({e}); check the internet connection and retry — "
                 "finished clips are kept")
    if no_urdu:
        print(f"no Urdu word in the dictionary for: {', '.join(no_urdu)}")
    print("done — src/app.py will speak these offline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
