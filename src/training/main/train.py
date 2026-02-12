import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_utils import ProductImageDataset, _url_to_filename


def find_project_root(start_file: Path) -> Path:
    cur = start_file.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return start_file.resolve().parents[3]


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def freeze_all_except_fc(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


def build_resnet18(num_classes: int, device: torch.device) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        model = resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    freeze_all_except_fc(model)

    return model.to(device)


def make_criterion(label_smoothing: float) -> nn.Module:
    try:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:
        return nn.CrossEntropyLoss()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"

    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(x)
            loss = criterion(outputs, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Validation", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        outputs = model(x)
        loss = criterion(outputs, y)

        total_loss += float(loss.item())
        preds = outputs.argmax(dim=1)
        total += int(y.size(0))
        correct += int((preds == y).sum().item())

    acc = 100.0 * correct / max(1, total)
    return total_loss / max(1, len(loader)), acc


def main() -> None:
    set_seed(42)

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 128
    NUM_EPOCHS = 10

    WARMUP_EPOCHS = 2
    UNFREEZE_LAYER3 = False

    LR_HEAD = 1e-3
    LR_BACKBONE = 1e-4
    WEIGHT_DECAY = 1e-4

    ETA_MIN = 1e-6
    LABEL_SMOOTHING = 0.1

    MIN_PER_CLASS = 25000
    TARGET_PER_CLASS = 25000

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    this_file = Path(__file__).resolve()
    repo_root = find_project_root(this_file)

    data_path = repo_root / "data" / "processed" / "products_cleaned.csv"
    image_dir = repo_root / "data" / "raw" / "images"
    failed_urls_path = this_file.parent / "failed_urls.txt"

    out_dir = repo_root / "models" / "phase2_resnet18"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"DATA_PATH:  {data_path}")
    print(f"IMAGE_DIR:  {image_dir}")
    print(f"OUT_DIR:    {out_dir}")
    print(f"FAILED_URLS:{failed_urls_path if failed_urls_path.exists() else 'not found (skipping)'}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find products_cleaned.csv at: {data_path}\n"
            f"Make sure you ran preprocessing and the file exists in data/processed/."
        )

    full_df = pd.read_csv(data_path)
    print(f"Original dataset size: {len(full_df)} images")

    image_dir.mkdir(parents=True, exist_ok=True)
    full_df["local_path"] = full_df["imgUrl"].fillna("").apply(
        lambda u: str(image_dir / _url_to_filename(u)) if isinstance(u, str) and u else ""
    )

    sample_n = min(2000, len(full_df))
    if sample_n > 0:
        sample_paths = full_df["local_path"].sample(n=sample_n, random_state=42).tolist()
        missing = sum(1 for p in sample_paths if (not p) or (not os.path.exists(p)))
        missing_ratio = missing / sample_n
        if missing_ratio > 0.05:
            print(
                f"Warning: {missing_ratio:.1%} of sampled images are missing on disk. "
                f"Make sure images exist under: {image_dir}"
            )

    if failed_urls_path.exists():
        failed_urls = [line.strip() for line in failed_urls_path.read_text().splitlines() if line.strip()]
        before = len(full_df)
        full_df = full_df[~full_df["imgUrl"].isin(failed_urls)]
        after = len(full_df)
        print(f"Removed {before - after} broken rows. Total valid samples: {after}")

    cat_counts = full_df["merged_category_id"].value_counts()
    valid_cats = cat_counts[cat_counts >= MIN_PER_CLASS].index.tolist()
    print(f"Found {len(valid_cats)} categories with >= {MIN_PER_CLASS} images.")

    full_df = full_df[full_df["merged_category_id"].isin(valid_cats)]

    full_df = full_df.groupby("merged_category_id", group_keys=False).sample(
        n=TARGET_PER_CLASS, random_state=42
    )

    unique_cats = sorted(full_df["merged_category_id"].unique())
    old_to_new_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_cats)}
    full_df["merged_category_id"] = full_df["merged_category_id"].map(old_to_new_mapping)

    num_classes = len(unique_cats)
    print(f"Filtered & Balanced Dataset: {len(full_df)} images ({num_classes} classes x {TARGET_PER_CLASS} images)")

    pd.DataFrame([{"old_id": k, "new_id": v} for k, v in old_to_new_mapping.items()]).to_csv(
        out_dir / "old_to_new_category_mapping.csv", index=False
    )

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(degrees=20, fill=255),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=255),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        stratify=full_df["merged_category_id"],
    )

    train_dataset = ProductImageDataset(df=train_df, transform=train_transform, image_dir=str(image_dir))
    val_dataset = ProductImageDataset(df=val_df, transform=val_transform, image_dir=str(image_dir))

    num_workers = 8
    pin_memory = (DEVICE.type == "cuda")
    persistent_workers = (num_workers > 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    print(f"Training set size:   {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    model = build_resnet18(num_classes=num_classes, device=DEVICE)

    criterion = make_criterion(LABEL_SMOOTHING)

    param_groups = [
        {"params": model.fc.parameters(), "lr": LR_HEAD},
        {"params": model.layer4.parameters(), "lr": LR_BACKBONE},
    ]
    if UNFREEZE_LAYER3:
        param_groups.append({"params": model.layer3.parameters(), "lr": LR_BACKBONE})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=ETA_MIN,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    best_acc = -1.0
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"

    for epoch in range(NUM_EPOCHS):
        if epoch == WARMUP_EPOCHS:
            unfreeze_module(model.layer4)
            if UNFREEZE_LAYER3:
                unfreeze_module(model.layer3)

        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | LRs: {[f'{lr:.2e}' for lr in lrs]}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "num_classes": num_classes,
            },
            last_path,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_acc,
                    "num_classes": num_classes,
                    "config": {
                        "image_size": IMAGE_SIZE,
                        "batch_size": BATCH_SIZE,
                        "num_epochs": NUM_EPOCHS,
                        "warmup_epochs": WARMUP_EPOCHS,
                        "unfreeze_layer3": UNFREEZE_LAYER3,
                        "lr_head": LR_HEAD,
                        "lr_backbone": LR_BACKBONE,
                        "weight_decay": WEIGHT_DECAY,
                        "label_smoothing": LABEL_SMOOTHING,
                        "min_per_class": MIN_PER_CLASS,
                        "target_per_class": TARGET_PER_CLASS,
                    },
                },
                best_path,
            )
            print(f"Saved BEST checkpoint to {best_path} (best_acc={best_acc:.2f}%)")

    print("\n--- Training Complete ---")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(history["train_loss"], label="Training Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["val_accuracy"], label="Validation Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()