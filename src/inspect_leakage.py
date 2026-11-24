
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT / "outputs" / "duplicate_analysis"
    / "duplicate_report.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "duplicate_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Find groups appearing in both train and test
split_counts = df.groupby("duplicate_group")["split"].nunique()

leakage_ids = split_counts[split_counts > 1].index

leakage = df[df["duplicate_group"].isin(leakage_ids)].copy()

print("\n========== LEAKAGE INSPECTION ==========")

print("Leakage groups:", len(leakage_ids))
print("Total involved images:", len(leakage))

# Identify groups with conflicting emotion labels
label_counts = leakage.groupby(
    "duplicate_group"
)["emotion"].nunique()

conflicting_ids = label_counts[label_counts > 1].index

conflicting = leakage[
    leakage["duplicate_group"].isin(conflicting_ids)
]

print("Cross-class leakage groups:", len(conflicting_ids))

# Save detailed reports
leakage.to_csv(
    OUTPUT_DIR / "train_test_leakage.csv",
    index=False
)

conflicting.to_csv(
    OUTPUT_DIR / "conflicting_leakage.csv",
    index=False
)

print("\nReports saved:")
print("train_test_leakage.csv")
print("conflicting_leakage.csv")

print("\nSample leakage groups:")

for group_id, group in leakage.groupby("duplicate_group"):
    print(f"\nGroup {group_id}")

    for _, row in group.iterrows():
        print(
            row["split"],
            row["emotion"],
            row["image_path"]
        )

    if group_id == list(leakage_ids[:5])[-1]:
        break

print("\n========== COMPLETED ==========")
