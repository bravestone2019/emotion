
from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "data_cleaned" / "train"

hashes = defaultdict(list)

for emotion_dir in TRAIN_DIR.iterdir():

    if not emotion_dir.is_dir():
        continue

    for path in emotion_dir.iterdir():

        if not path.is_file():
            continue

        with Image.open(path) as img:

            gray = img.convert("L")

            digest = hashlib.sha256(
                str(gray.size).encode() + gray.tobytes()
            ).hexdigest()

        hashes[digest].append(emotion_dir.name)

cross_class = 0
duplicate_groups = 0

for labels in hashes.values():

    if len(labels) > 1:
        duplicate_groups += 1

        if len(set(labels)) > 1:
            cross_class += 1

print("\nCLEANED DATASET VERIFICATION")
print("Remaining duplicate groups:", duplicate_groups)
print("Remaining cross-class groups:", cross_class)
print("Verification completed!")
