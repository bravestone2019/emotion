
import csv
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import EmotionCNN


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "data" / "test"
MODEL_PATH = ROOT / "models" / "emotion_model_weighted.pth"

OUTPUT_DIR = ROOT / "outputs" / "error_analysis"
MISCLASSIFIED_DIR = OUTPUT_DIR / "misclassified"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
MAX_IMAGES_PER_CATEGORY = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# =====================================================
# 2. LOAD TEST DATASET
# =====================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda"
)

classes = test_dataset.classes
class_to_idx = test_dataset.class_to_idx

print("Classes:", classes)
print("Test images:", len(test_dataset))


# =====================================================
# 3. LOAD WEIGHTED CNN
# =====================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if "classes" in checkpoint:
    assert checkpoint["classes"] == classes, (
        "Checkpoint class order does not match test dataset!"
    )

if "class_to_idx" in checkpoint:
    assert checkpoint["class_to_idx"] == class_to_idx, (
        "Checkpoint class mapping does not match!"
    )

model = EmotionCNN(
    num_classes=len(classes)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Weighted CNN loaded successfully!")


# =====================================================
# 4. PREDICTION AND ERROR COLLECTION
# =====================================================

errors = []
correct_count = 0
total_count = 0

error_counts = Counter()
saved_counts = Counter()

sample_index = 0

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        for i in range(len(labels)):

            actual_idx = labels[i].item()
            predicted_idx = predictions[i].item()

            confidence = confidences[i].item()

            actual_emotion = classes[actual_idx]
            predicted_emotion = classes[predicted_idx]

            image_path, _ = test_dataset.samples[sample_index]

            if actual_idx == predicted_idx:

                correct_count += 1

            else:

                error_key = (
                    actual_emotion,
                    predicted_emotion
                )

                error_counts[error_key] += 1

                errors.append({
                    "image": Path(image_path).name,
                    "image_path": image_path,
                    "actual": actual_emotion,
                    "predicted": predicted_emotion,
                    "confidence": confidence
                })

                # Create emotion-specific folder
                folder_name = (
                    f"{actual_emotion}_as_{predicted_emotion}"
                )

                save_dir = MISCLASSIFIED_DIR / folder_name
                save_dir.mkdir(parents=True, exist_ok=True)

                # Limit saved images per category
                if saved_counts[error_key] < MAX_IMAGES_PER_CATEGORY:

                    save_path = (
                        save_dir /
                        f"{sample_index}_{Path(image_path).stem}.png"
                    )

                    # Convert normalized tensor back to image range
                    save_image(
                        images[i].cpu(),
                        save_path,
                        normalize=True,
                        value_range=(-1, 1)
                    )

                    saved_counts[error_key] += 1

            total_count += 1
            sample_index += 1


# =====================================================
# 5. PRINT SUMMARY
# =====================================================

accuracy = correct_count / total_count * 100

print("\n" + "=" * 55)
print("ERROR ANALYSIS SUMMARY")
print("=" * 55)

print("Total images:", total_count)
print("Correct predictions:", correct_count)
print("Incorrect predictions:", len(errors))
print(f"Accuracy: {accuracy:.2f}%")
print(f"Error rate: {100 - accuracy:.2f}%")

print("\nMost common classification errors:")

for (actual, predicted), count in error_counts.most_common(15):

    print(
        f"{actual:10s} -> {predicted:10s}: {count}"
    )


# =====================================================
# 6. SAVE CSV REPORT
# =====================================================

csv_path = OUTPUT_DIR / "error_summary.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "image",
        "image_path",
        "actual",
        "predicted",
        "confidence"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(errors)

print("\nCSV saved:", csv_path)


# =====================================================
# 7. GENERATE ERROR DISTRIBUTION CHART
# =====================================================

if error_counts:

    sorted_errors = error_counts.most_common()

    labels = [
        f"{actual} → {predicted}"
        for (actual, predicted), count in sorted_errors
    ]

    counts = [
        count
        for pair, count in sorted_errors
    ]

    plt.figure(figsize=(12, 8))

    plt.barh(
        labels[::-1],
        counts[::-1]
    )

    plt.xlabel("Number of Misclassified Images")
    plt.ylabel("Actual → Predicted Emotion")

    plt.title(
        "Weighted CNN - Error Distribution"
    )

    plt.tight_layout()

    chart_path = OUTPUT_DIR / "error_distribution.png"

    plt.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Error chart saved:", chart_path)


# =====================================================
# 8. FINAL OUTPUT
# =====================================================

print("\n" + "=" * 55)
print("ERROR ANALYSIS COMPLETED!")
print("=" * 55)

print("Output directory:", OUTPUT_DIR)
print("Misclassified images:", MISCLASSIFIED_DIR)
print("Total error categories:", len(error_counts))
print("Images saved:", sum(saved_counts.values()))
