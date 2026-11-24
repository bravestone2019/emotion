
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import EmotionCNN


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "data" / "test"

MODEL_PATH = (
    ROOT / "models" / "emotion_model_weighted.pth"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "error_analysis" / "contact_sheets"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
IMAGES_PER_CATEGORY = 25

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TARGET_ERRORS = [
    ("neutral", "sad"),
    ("fear", "sad"),
    ("disgust", "angry"),
    ("angry", "sad")
]

print("Device:", DEVICE)


# =====================================================
# 2. LOAD DATASET
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

print("Classes:", classes)
print("Test images:", len(test_dataset))


# =====================================================
# 3. LOAD MODEL
# =====================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if "classes" in checkpoint:
    assert checkpoint["classes"] == classes

if "class_to_idx" in checkpoint:
    assert checkpoint["class_to_idx"] == test_dataset.class_to_idx

model = EmotionCNN(
    num_classes=len(classes)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Weighted CNN loaded successfully!")


# =====================================================
# 4. COLLECT MISCLASSIFIED IMAGES
# =====================================================

error_samples = defaultdict(list)

with torch.inference_mode():

    for images, labels in test_loader:

        images_device = images.to(DEVICE)

        outputs = model(images_device)

        probabilities = torch.softmax(
            outputs, dim=1
        )

        confidences, predictions = probabilities.max(dim=1)

        for i in range(len(labels)):

            actual_idx = labels[i].item()
            predicted_idx = predictions[i].item()

            if actual_idx == predicted_idx:
                continue

            actual = classes[actual_idx]
            predicted = classes[predicted_idx]

            category = (actual, predicted)

            if category not in TARGET_ERRORS:
                continue

            if len(error_samples[category]) >= IMAGES_PER_CATEGORY:
                continue

            confidence = confidences[i].item() * 100

            # Restore image from normalized tensor
            image = images[i].cpu().squeeze(0)

            image = (image * 0.5 + 0.5).clamp(0, 1)

            error_samples[category].append({
                "image": image.numpy(),
                "actual": actual,
                "predicted": predicted,
                "confidence": confidence
            })


# =====================================================
# 5. GENERATE CONTACT SHEETS
# =====================================================

def generate_contact_sheet(category, samples):

    actual, predicted = category

    fig, axes = plt.subplots(
        5, 5,
        figsize=(13, 13)
    )

    fig.suptitle(
        f"Actual: {actual.upper()} | Predicted: {predicted.upper()}",
        fontsize=17,
        fontweight="bold"
    )

    axes = axes.flatten()

    for i, ax in enumerate(axes):

        if i < len(samples):

            sample = samples[i]

            ax.imshow(
                sample["image"],
                cmap="gray",
                vmin=0,
                vmax=1
            )

            ax.set_title(
                f"Confidence: {sample['confidence']:.1f}%",
                fontsize=9
            )

        else:

            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filename = f"{actual}_as_{predicted}.png"

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {output_path} "
        f"({len(samples)} images)"
    )


# =====================================================
# 6. GENERATE ALL FOUR CONTACT SHEETS
# =====================================================

print("\nGenerating contact sheets...\n")

for category in TARGET_ERRORS:

    samples = error_samples[category]

    print(
        f"{category[0]} -> {category[1]}: "
        f"{len(samples)} images collected"
    )

    if samples:

        generate_contact_sheet(
            category,
            samples
        )

    else:

        print("No matching errors found.")


# =====================================================
# 7. FINAL SUMMARY
# =====================================================

print("\n" + "=" * 55)
print("VISUAL ERROR INSPECTION COMPLETED")
print("=" * 55)

print("Output directory:", OUTPUT_DIR)

print("Total images collected:",
      sum(len(v) for v in error_samples.values()))

print("Generated contact sheets:",
      sum(bool(v) for v in error_samples.values()))
