
from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib
import csv

ROOT = Path(__file__).resolve().parent.parent

DATA_DIRS = {
    "train": ROOT / "data" / "train",
    "test": ROOT / "data" / "test"
}

OUTPUT_DIR = ROOT / "outputs" / "duplicate_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

hashes = defaultdict(list)

# Scan both datasets
for split, split_dir in DATA_DIRS.items():

    for emotion_dir in split_dir.iterdir():

        if not emotion_dir.is_dir():
            continue

        for image_path in emotion_dir.iterdir():

            if not image_path.is_file():
                continue

            try:
                with Image.open(image_path) as img:

                    gray = img.convert("L")
                    content = gray.tobytes()

                    # Include dimensions to avoid ambiguous hashes
                    digest = hashlib.sha256(
                        str(gray.size).encode() + content
                    ).hexdigest()

                hashes[digest].append({
                    "split": split,
                    "emotion": emotion_dir.name,
                    "path": str(image_path)
                })

            except Exception as e:
                print("Error:", image_path, e)


# Identify duplicate groups
duplicates = {
    key: items
    for key, items in hashes.items()
    if len(items) > 1
}

cross_class = []
cross_split = []

for items in duplicates.values():

    if len(set(x["emotion"] for x in items)) > 1:
        cross_class.append(items)

    if len(set(x["split"] for x in items)) > 1:
        cross_split.append(items)


# Save detailed report
csv_path = OUTPUT_DIR / "duplicate_report.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "split", "emotion", "image_path", "duplicate_group"
    ])

    for group_id, items in enumerate(duplicates.values(), 1):

        for item in items:

            writer.writerow([
                item["split"],
                item["emotion"],
                item["path"],
                group_id
            ])


print("\n========== DUPLICATE ANALYSIS ==========")

print("Total duplicate groups:", len(duplicates))
print("Cross-class groups:", len(cross_class))
print("Train-test leakage groups:", len(cross_split))

print("\nReport saved:", csv_path)

print("\n========== COMPLETED ==========")

