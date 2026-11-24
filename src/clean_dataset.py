
from pathlib import Path
from collections import defaultdict
from PIL import Image
import shutil
import hashlib
import csv

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"

CLEAN_DIR = ROOT / "data_cleaned"

CLEAN_TRAIN = CLEAN_DIR / "train"
CLEAN_TEST = CLEAN_DIR / "test"

CLEAN_TRAIN.mkdir(parents=True, exist_ok=True)
CLEAN_TEST.mkdir(parents=True, exist_ok=True)


def get_hash(path):
    with Image.open(path) as img:
        gray = img.convert("L")
        return hashlib.sha256(
            str(gray.size).encode() + gray.tobytes()
        ).hexdigest()


# 1. Collect all test hashes
test_hashes = set()

for emotion_dir in TEST_DIR.iterdir():

    if not emotion_dir.is_dir():
        continue

    for path in emotion_dir.iterdir():

        if path.is_file():
            test_hashes.add(get_hash(path))


# 2. Copy test images unchanged
for emotion_dir in TEST_DIR.iterdir():

    if not emotion_dir.is_dir():
        continue

    destination = CLEAN_TEST / emotion_dir.name
    destination.mkdir(parents=True, exist_ok=True)

    for path in emotion_dir.iterdir():

        if path.is_file():
            shutil.copy2(path, destination / path.name)


# 3. Clean training images
removed = []
kept = 0

for emotion_dir in TRAIN_DIR.iterdir():

    if not emotion_dir.is_dir():
        continue

    destination = CLEAN_TRAIN / emotion_dir.name
    destination.mkdir(parents=True, exist_ok=True)

    for path in emotion_dir.iterdir():

        if not path.is_file():
            continue

        digest = get_hash(path)

        if digest in test_hashes:

            removed.append([
                emotion_dir.name,
                str(path),
                digest
            ])

        else:

            shutil.copy2(path, destination / path.name)
            kept += 1


# 4. Save report
report_path = CLEAN_DIR / "cleaning_report.csv"

with open(report_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "emotion",
        "removed_image",
        "hash"
    ])

    writer.writerows(removed)


print("\n========== CLEANING SUMMARY ==========")
print("Training images retained:", kept)
print("Training images removed:", len(removed))
print("Test images retained:", sum(
    1 for p in CLEAN_TEST.rglob("*") if p.is_file()
))
print("Cleaning report:", report_path)
print("\nCLEANING COMPLETED")
