"""Prepare a training-only dataset with exact duplicates handled safely.

The cleaned test set is never modified or copied by this script. Exact pixel
duplicates with conflicting labels are excluded from training and recorded for
manual review. Within-label duplicates are represented by one deterministic
file.
"""

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import shutil

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data_cleaned" / "train"
DESTINATION = ROOT / "data_prepared" / "train"
REPORT = ROOT / "data_prepared" / "duplicate_decisions.csv"


def pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        gray = image.convert("L")
        return hashlib.sha256(
            str(gray.size).encode() + gray.tobytes()
        ).hexdigest()


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Cleaned training data not found: {SOURCE}")

    grouped = defaultdict(list)
    for class_dir in sorted(p for p in SOURCE.iterdir() if p.is_dir()):
        for path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            grouped[pixel_hash(path)].append((class_dir.name, path))

    decisions = []
    kept = []
    for digest, entries in sorted(grouped.items()):
        labels = {label for label, _ in entries}
        if len(labels) > 1:
            decision = "excluded_conflicting_labels"
        else:
            kept.append(entries[0])
            decision = "kept" if len(entries) == 1 else "kept_one_exact_duplicate"

        for label, path in entries:
            decisions.append([digest, decision, label, str(path)])

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    for label, path in kept:
        target_dir = DESTINATION / label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_dir / path.name)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(["pixel_hash", "decision", "label", "source_path"])
        writer.writerows(decisions)

    excluded_groups = sum(
        1 for entries in grouped.values()
        if len({label for label, _ in entries}) > 1
    )
    print(f"Unique exact-image groups: {len(grouped)}")
    print(f"Training images retained: {len(kept)}")
    print(f"Conflicting-label groups excluded: {excluded_groups}")
    print(f"Prepared training data: {DESTINATION}")
    print(f"Review report: {REPORT}")


if __name__ == "__main__":
    main()
