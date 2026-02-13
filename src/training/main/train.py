import hashlib

import pandas as pd
from data_pipeline import get_webdataset_loaders, create_datasets, get_dataloaders, split_data, set_random_seeds, get_data_transforms
import torch
import torch.nn as nn
from torchvision.models import resnet18
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from tqdm import tqdm
import os


def build_resnet18(num_classes: int, device: torch.device) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        model = resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model.to(device)


def make_criterion(label_smoothing: float) -> nn.Module:
    try:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:
        return nn.CrossEntropyLoss()

def freeze_all_except_fc(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

IMAGE_DIR = '../../../data/raw/images/'
FAILED_URLS_PATH = 'failed_urls.txt'
DATA_PATH = '../../../data/processed/products_cleaned.csv'

def clean_df(df):
    def _url_to_filename(url):
        return hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg"

    cat_counts = df['merged_category_id'].value_counts()
    valid_cats = cat_counts[cat_counts >= 25000].index.tolist()
    df = df[df['merged_category_id'].isin(valid_cats)]
    df = df.groupby('merged_category_id', group_keys=False).sample(
        n=25000,
        random_state=42
    )
    unique_cats = sorted(df['merged_category_id'].unique())
    old_to_new_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_cats)}
    df['merged_category_id'] = df['merged_category_id'].map(old_to_new_mapping)
    df['local_path'] = df['imgUrl'].fillna('').apply(
        lambda u: os.path.join(IMAGE_DIR, _url_to_filename(u)) if isinstance(u, str) and u else ''
    )
    try:
        with open(FAILED_URLS_PATH, 'r') as f:
            failed_urls  = [line.strip() for line in f if line.strip()]
        df = df[~df['imgUrl'].isin(failed_urls)]
    except FileNotFoundError:
        pass

    df = df[df['local_path'].notna()]
    df = df[df['local_path'] != ""]

    return df


def main():

    set_random_seeds()

    NUM_EPOCHS = 10
    LR_HEAD = 1e-3
    LR_BACKBONE = 1e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    ETA_MIN = 1e-6

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # 🔥 Load loaders
    df = pd.read_csv(DATA_PATH)
    df = clean_df(df)
    train_df, val_df, test_df = split_data(df)
    train_transform, val_test_transform = get_data_transforms()
    train_dataset, val_dataset, test_dataset = create_datasets(train_df, val_df, test_df, train_transform, val_test_transform)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, )
    # train_loader, val_loader, test_loader = get_webdataset_loaders()

    # 🔥 Automatically detect number of classes
    # (assumes labels are 0..N-1)
    sample_batch = next(iter(train_loader))
    _, sample_labels = sample_batch
    num_classes = int(sample_labels.max().item() + 1)

    print(f"Detected {num_classes} classes.")

    model = build_resnet18(num_classes, DEVICE)
    freeze_all_except_fc(model)

    criterion = make_criterion(LABEL_SMOOTHING)

    param_groups = [
        {"params": model.fc.parameters(), "lr": LR_HEAD},
        {"params": model.layer4.parameters(), "lr": LR_BACKBONE},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=ETA_MIN,
    )

    scaler = torch.amp.GradScaler("cuda")

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    best_acc = -1.0

    BEST_MODEL_PATH = "../../models/main/best_model.pth"

    for epoch in range(NUM_EPOCHS):
        # Gradually unfreeze layer4 after first epoch
        if epoch == 1:
            print("Unfreezing layer4...")
            unfreeze_module(model.layer4)
        elif epoch == 3:
            print("Unfreezing layer3...")
            unfreeze_module(model.layer3)

        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # ------------------
        # Train
        # ------------------
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc="Training", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = model(x)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ------------------
        # Validation
        # ------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation", leave=False):
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total

        scheduler.step()
        lrs = [pg["lr"] for pg in optimizer.param_groups]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
            f"LRs: {[f'{lr:.2e}' for lr in lrs]}"
        )

        # -------------------------
        # Save BEST model
        # -------------------------
        if val_acc > best_acc:
            best_acc = val_acc

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_accuracy": best_acc,
                "num_classes": num_classes,
                "history": history,
            }, BEST_MODEL_PATH)

            print(f"Saved BEST model to {BEST_MODEL_PATH} (acc={best_acc:.2f}%)")


        # -------------------------
        # Save per-epoch checkpoint
        # -------------------------
        checkpoint_dir = Path("../../models/main/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "num_classes": num_classes,
        }, checkpoint_path)

        print(f"Saved checkpoint to {checkpoint_path}")


    print("\n--- Training Complete ---")

    # ------------------
    # Final Test Evaluation
    # ------------------
    print("\n--- Final Test Evaluation ---")

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            outputs = model(x)
            loss = criterion(outputs, y)

            test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    test_loss /= len(test_loader)
    test_acc = 100.0 * correct / total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")


    print(f"Best Validation Accuracy: {best_acc:.2f}%")

    # Plot curves
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history["val_accuracy"])
    plt.title("Validation Accuracy")

    plt.show()


if __name__ == "__main__":
    main()
