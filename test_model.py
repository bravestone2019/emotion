
import torch
from src.model import EmotionCNN

# Select GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Create model
model = EmotionCNN(num_classes=7).to(device)

# Create dummy batch
dummy_input = torch.randn(
    8, 1, 48, 48
).to(device)

# Forward pass
with torch.no_grad():
    output = model(dummy_input)

print("Device:", device)
print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("CNN working successfully!")
