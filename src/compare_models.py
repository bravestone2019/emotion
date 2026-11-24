
import csv
import hashlib
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from model import EmotionCNN


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "data_cleaned" / "test"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR = ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATHS = {
    "Normal CNN": MODEL_DIR / "emotion_model.pth",
    "Weighted CNN": MODEL_DIR / "emotion_model_weighted.pth"
}

print("=" * 55)
print("EMOTION RECOGNITION - MODEL COMPARISON")
print("=" * 55)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================================
# 2. DATA PREPROCESSING
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
num_classes = len(classes)

print("\nClasses:", classes)
print("Test images:", len(test_dataset))


# =====================================================
# 3. LOAD MODEL CHECKPOINT
# =====================================================

def load_model(model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    # Support checkpoint dictionaries and raw state dictionaries
    if isinstance(checkpoint, dict):

        if "classes" in checkpoint:
            if checkpoint["classes"] != classes:
                raise ValueError(
                    f"Class order mismatch in {model_path.name}"
                )

        if "class_to_idx" in checkpoint:
            if checkpoint["class_to_idx"] != test_dataset.class_to_idx:
                raise ValueError(
                    f"Class mapping mismatch in {model_path.name}"
                )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Assume this is a raw state dictionary
            state_dict = checkpoint

    else:
        raise ValueError(
            f"Unsupported checkpoint format: {model_path.name}"
        )

    model = EmotionCNN(
        num_classes=num_classes
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    return model


# =====================================================
# 4. EVALUATE MODEL
# =====================================================

def evaluate_model(model, model_name):

    all_predictions = []
    all_labels = []

    # Warm-up for more reliable GPU timing
    if DEVICE.type == "cuda":
        dummy = torch.randn(
            1, 1, 48, 48,
            device=DEVICE
        )

        with torch.inference_mode():
            for _ in range(10):
                model(dummy)

        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():

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

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Sensitivity analysis: count each exact pixel duplicate group once. Groups
    # carrying conflicting labels are omitted from this adjusted metric, while
    # the standard metrics below continue to include every unchanged test item.
    duplicate_groups = {}
    for index, (image_path, _) in enumerate(test_dataset.samples):
        with Image.open(image_path) as image:
            gray = image.convert("L")
            digest = hashlib.sha256(
                str(gray.size).encode() + gray.tobytes()
            ).hexdigest()
        duplicate_groups.setdefault(digest, []).append(index)

    duplicate_group_count = sum(len(indices) > 1 for indices in duplicate_groups.values())
    duplicate_extra_copies = sum(
        len(indices) - 1 for indices in duplicate_groups.values() if len(indices) > 1
    )
    conflicting_groups = [
        indices for indices in duplicate_groups.values()
        if len(set(all_labels[indices])) > 1
    ]
    unique_indices = [
        indices[0] for indices in duplicate_groups.values()
        if len(set(all_labels[indices])) == 1
    ]
    unique_accuracy = accuracy_score(
        all_labels[unique_indices], all_predictions[unique_indices]
    )
    _, _, unique_macro_f1, _ = precision_recall_fscore_support(
        all_labels[unique_indices],
        all_predictions[unique_indices],
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    # Metrics
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    avg_time = elapsed_time / len(test_dataset)

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    print(f"Accuracy:          {accuracy * 100:.2f}%")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")
    print(f"Exact duplicate groups: {duplicate_group_count} ({duplicate_extra_copies} extra copies)")
    print(f"Conflicting-label duplicate groups: {len(conflicting_groups)}")
    print(f"Unique-group accuracy (conflicting groups excluded): {unique_accuracy * 100:.2f}%")
    print(f"Unique-group macro F1 (conflicting groups excluded): {unique_macro_f1:.4f}")
    print(f"Avg time/image:    {avg_time * 1000:.3f} ms")
    print(f"Evaluation time:   {elapsed_time:.3f} seconds")

    # Classification report
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        target_names=classes,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    report_file = OUTPUT_DIR / (
        model_name.lower().replace(" ", "_") + "_report.txt"
    )

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(num_classes))
    )

    normalized_cm = (
        cm.astype(float) /
        np.maximum(cm.sum(axis=1, keepdims=True), 1)
    )

    # Plot normalized confusion matrix
    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        normalized_cm,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    ax.set_title(f"{model_name} - Normalized Confusion Matrix")
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("Actual Emotion")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))

    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    for i in range(num_classes):
        for j in range(num_classes):

            ax.text(
                j, i,
                f"{normalized_cm[i, j]:.2f}",
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300
    )

    plt.close(fig)

    return {
        "Model": model_name,
        "Accuracy (%)": accuracy * 100,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Exact Duplicate Groups": duplicate_group_count,
        "Duplicate Extra Copies": duplicate_extra_copies,
        "Conflicting Duplicate Groups": len(conflicting_groups),
        "Unique-Group Accuracy (%)": unique_accuracy * 100,
        "Unique-Group Macro F1": unique_macro_f1,
        "Avg Inference (ms/image)": avg_time * 1000,
        "Evaluation Time (seconds)": elapsed_time
    }


# =====================================================
# 5. COMPARE BOTH MODELS
# =====================================================

results = []

for model_name, model_path in MODEL_PATHS.items():

    print(f"\nLoading {model_name}...")

    model = load_model(model_path)

    result = evaluate_model(
        model,
        model_name
    )

    results.append(result)

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# =====================================================
# 6. SAVE COMPARISON CSV
# =====================================================

csv_path = OUTPUT_DIR / "model_comparison.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# =====================================================
# 7. FINAL RESULT
# =====================================================

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

for result in results:

    print(f"\n{result['Model']}")

    print(
        f"Accuracy: {result['Accuracy (%)']:.2f}%"
    )

    print(
        f"Macro F1: {result['Macro F1']:.4f}"
    )

    print(
        f"Weighted F1: {result['Weighted F1']:.4f}"
    )

best_model = max(
    results,
    key=lambda x: x["Macro F1"]
)

print("\nBest model based on Macro F1:")
print(best_model["Model"])

print("\nResults saved to:", OUTPUT_DIR)
print("MODEL COMPARISON COMPLETED!")
