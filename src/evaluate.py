
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from model import EmotionCNN


# ----------------------------------
# 1. Configuration
# ----------------------------------

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "data_cleaned" / "test"
MODEL_PATH = ROOT / "models" / "emotion_model_weighted.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Evaluation device:", DEVICE)


# ----------------------------------
# 2. Load test dataset
# ----------------------------------

test_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


# ----------------------------------
# 3. Load trained model
# ----------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

classes = checkpoint["classes"]

assert classes == test_dataset.classes, \
    "Emotion class mapping does not match!"

model = EmotionCNN(
    num_classes=len(classes)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully!")
print("Classes:", classes)


# ----------------------------------
# 4. Generate predictions
# ----------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)


# ----------------------------------
# 5. Calculate accuracy
# ----------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")


# ----------------------------------
# 6. Classification report
# ----------------------------------

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )
)


# ----------------------------------
# 7. Confusion matrix
# ----------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(classes)))
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.title("FER-2013 Emotion Recognition Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.tight_layout()

plt.savefig(
    ROOT / "models" / "confusion_matrix.png",
    dpi=300
)

# plt.show()

# Normalized confusion matrix
cm_normalized = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(classes))),
    normalize="true"
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.title("Normalized Emotion Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.tight_layout()

plt.savefig(
    ROOT / "models" / "normalized_confusion_matrix.png",
    dpi=300
)

plt.show()

# ----------------------------------
# 8. Per-class accuracy
# ----------------------------------

print("\nPER-CLASS ACCURACY")

for i, emotion in enumerate(classes):

    mask = all_labels == i

    class_accuracy = (
        (all_predictions[mask] == all_labels[mask]).mean()
        * 100
    )

    print(f"{emotion}: {class_accuracy:.2f}%")


print("\nEvaluation completed!")
