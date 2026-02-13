import torch
import matplotlib.pyplot as plt
from pathlib import Path


CHECKPOINT_DIR = Path("../../models/main/checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint_files = sorted(
    CHECKPOINT_DIR.glob("checkpoint_epoch_*.pth"),
    key=lambda x: int(x.stem.split("_")[-1])  # sort by epoch number
)

if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")

train_losses = []
val_losses = []
val_accuracies = []
epochs = []

for ckpt_path in checkpoint_files:
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    epochs.append(checkpoint["epoch"])
    train_losses.append(checkpoint["train_loss"])
    val_losses.append(checkpoint["val_loss"])
    val_accuracies.append(checkpoint["val_accuracy"])

print(f"Loaded {len(epochs)} checkpoints.")

print("Epochs:", epochs)
print("Train Losses:", train_losses)
print("Val Losses:", val_losses)

plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train")
plt.plot(epochs, val_losses, label="Val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")

plt.tight_layout()
plt.show()
