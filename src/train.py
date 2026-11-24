
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from model import EmotionCNN


# 1. Configuration
ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "data_prepared" / "train"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weighted",
    action="store_true",
    help="Train with square-root inverse-frequency class weights.",
)
args = parser.parse_args()
MODEL_NAME = "emotion_model_weighted.pth" if args.weighted else "emotion_model.pth"
MODEL_PATH = MODEL_DIR / MODEL_NAME

if not TRAIN_DIR.is_dir():
    raise FileNotFoundError(
        "Prepared training data is missing. Run "
        "python src/prepare_training_data.py first."
    )

BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# 2. Transforms
train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# 3. Load datasets
full_train = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

full_val = datasets.ImageFolder(
    TRAIN_DIR,
    transform=val_transform
)

# Stratified split
indices = list(range(len(full_train)))
labels = full_train.targets

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

train_dataset = Subset(full_train, train_indices)
val_dataset = Subset(full_val, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda"
)

classes = full_train.classes

print("Classes:", classes)
print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test set is reserved for separate final evaluation.")

# Calculate class weights using training data only
train_labels = [labels[i] for i in train_indices]

class_counts = torch.bincount(
    torch.tensor(train_labels),
    minlength=len(classes)
).float()

# Square-root inverse-frequency weighting
if args.weighted:
    class_weights = torch.sqrt(
        class_counts.sum() / (len(classes) * class_counts)
    )
    class_weights = (class_weights / class_weights.mean()).to(DEVICE)
    print("\nClass weights:")
    for emotion, weight in zip(classes, class_weights):
        print(f"{emotion}: {weight.item():.3f}")
else:
    class_weights = None

# 4. Model
model = EmotionCNN(
    num_classes=len(classes)
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)


# 5. Training function
def run_epoch(loader, training=False):

    model.train() if training else model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.set_grad_enabled(training):

        for images, labels_batch in loader:

            images = images.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels_batch)

            if training:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

            all_predictions.extend(
                predictions.detach().cpu().numpy()
            )

            all_labels.extend(
                labels_batch.detach().cpu().numpy()
            )

    accuracy = 100 * correct / total

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return total_loss / total, accuracy, macro_f1


# 6. Training loop
best_val_macro_f1 = -1

for epoch in range(EPOCHS):

    train_loss, train_acc, train_f1 = run_epoch(
        train_loader, training=True
    )

    val_loss, val_acc, val_f1 = run_epoch(
        val_loader, training=False
    )

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Train Macro F1: {train_f1:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Validation Macro F1: {val_f1:.4f}")

    # Save the model with the best validation macro F1
    if val_f1 > best_val_macro_f1:

        best_val_macro_f1 = val_f1

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "class_to_idx": full_train.class_to_idx,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        }, MODEL_PATH)

        print(f"Best {('weighted' if args.weighted else 'normal')} model saved!")

print(f"\n{'WEIGHTED' if args.weighted else 'NORMAL'} TRAINING COMPLETED")
print("Best validation macro F1:", best_val_macro_f1)
print("Checkpoint:", MODEL_PATH)
