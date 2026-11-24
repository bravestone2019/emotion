
from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

for split in ["train", "test"]:

    dataset_dir = ROOT / "data" / split

    print(f"\n{'='*45}")
    print(f"{split.upper()} DATASET AUDIT")
    print(f"{'='*45}")

    total = 0
    corrupted = 0

    for emotion_dir in sorted(dataset_dir.iterdir()):

        if not emotion_dir.is_dir():
            continue

        files = list(emotion_dir.glob("*"))

        valid = 0

        for file in files:

            if not file.is_file():
                continue

            total += 1

            try:
                with Image.open(file) as img:
                    img.verify()
                valid += 1

            except Exception:
                corrupted += 1
                print("Corrupted:", file)

        print(f"{emotion_dir.name}: {valid} images")

    print("\nTotal images:", total)
    print("Corrupted images:", corrupted)
    print("Valid images:", total - corrupted)
