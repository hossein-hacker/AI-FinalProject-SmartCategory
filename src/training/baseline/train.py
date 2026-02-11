import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import requests
import hashlib
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset_utils import ProductImageDataset
from dataset_utils import _url_to_filename
from dataset_utils import load_image_from_url
from sklearn.model_selection import train_test_split


def main():
    
    torch.backends.cudnn.benchmark = True


    # CONFIG
    mapping_df = pd.read_csv('../../../data/processed/category_mapping.csv')
    NUM_CLASSES = mapping_df['merged_category_id'].nunique()

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # --- Paths ---
    DATA_PATH = '../../../data/processed/products_cleaned.csv'
    IMAGE_DIR = '../../../data/raw/images/' # Directory where images are stored
    
    PRECACHE_STEP = False

    print(f'Using device: {DEVICE}')

    full_df = pd.read_csv(DATA_PATH)
    print(f"Original dataset size: {len(full_df)} images")

    # ------------------------------------------------------------------------

    # 1. Identify categories with >= 25,000 samples
    cat_counts = full_df['merged_category_id'].value_counts()
    valid_cats = cat_counts[cat_counts >= 25000].index.tolist()
    print(f"Found {len(valid_cats)} categories with >= 25,000 images.")

    # 2. Filter the dataframe to only these categories
    full_df = full_df[full_df['merged_category_id'].isin(valid_cats)]

    # 3. Downsample each category to exactly 25,000
    # group_keys=False keeps the original index structure
    full_df = full_df.groupby('merged_category_id', group_keys=False).apply(lambda x: x.sample(25000, random_state=42))

    # 4. CRITICAL: Remap category IDs to contiguous 0...N-1 range
    # If we keep categories [0, 5, 10], the model will crash if we set NUM_CLASSES=3.
    # We must remap them to [0, 1, 2].
    unique_cats = sorted(full_df['merged_category_id'].unique())
    old_to_new_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_cats)}
    full_df['merged_category_id'] = full_df['merged_category_id'].map(old_to_new_mapping)

    # 5. Update global configuration
    # We overwrite the NUM_CLASSES from Step 2 to match this new filtered dataset
    NUM_CLASSES = len(unique_cats)

    print(f"Filtered & Balanced Dataset: {len(full_df)} images ({NUM_CLASSES} classes x 25,000 images)")

    # ------------------------------------------------------------------------

    
    os.makedirs(IMAGE_DIR, exist_ok=True)

    full_df['local_path'] = full_df['imgUrl'].fillna('').apply(
        lambda u: os.path.join(IMAGE_DIR, _url_to_filename(u)) if isinstance(u, str) and u else ''
    )

    full_df.head()

    # ------------------------------------------------------------------------

    # Remove the failed URLs
    failed_urls_path = 'failed_urls.txt'
    try:
        with open(failed_urls_path, 'r') as f:
            failed_urls  = [line.strip() for line in f if line.strip()]

        initial_count = len(full_df)
        full_df = full_df[~full_df['imgUrl'].isin(failed_urls)]
        final_count = len(full_df)

        print(f"\nDataFrame Cleaned:")
        print(f"- Removed {initial_count - final_count} broken rows.")
        print(f"- Total valid samples: {final_count}")
    except FileNotFoundError:
        print(f"Failed URLs file {failed_urls_path} not found. No rows removed from dataset.")

    # ------------------------------------------------------------------------

    # Define separate transforms for training and validation
    # The training transform includes augmentation from the augmentation_steps notebook
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(degrees=20, fill=255),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=255),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # The validation transform is minimal (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Split the dataframe into training and validation sets, ensuring stratification
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['merged_category_id'])

    # Create separate datasets for training and validation with their respective transforms
    train_dataset = ProductImageDataset(df=train_df, transform=train_transform)
    val_dataset = ProductImageDataset(df=val_df, transform=val_transform)

    # Create DataLoaders
    num_workers = 8
    print("Number of workers for DataLoader:", num_workers)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=False,
        persistent_workers=True  # This keeps workers alive between epochs, saving time
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=False,
        persistent_workers=True
    )

    print(f'Found {len(full_df)} total images.')
    print(f'Training set size: {len(train_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')

    # ------------------------------------------------------------------------

    if PRECACHE_STEP:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def cache_image_only(url):
            return load_image_from_url(url) is not None


        MAX_WORKERS = 50
        PREFETCH = 5000

        urls = full_df['imgUrl'].dropna().unique()[::-1]
        total = len(urls)

        print(f"Found {total} unique image URLs to download.")
        print(f"Using {MAX_WORKERS} workers, prefetch={PREFETCH}")

        success = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = set()

            pbar = tqdm(total=total, desc="Caching images", unit="img")

            for url in urls:
                futures.add(executor.submit(cache_image_only, url))

                if len(futures) >= PREFETCH:
                    done = next(as_completed(futures))
                    futures.remove(done)

                    if done.result():
                        success += 1
                    pbar.update(1)

            # drain remaining
            for future in as_completed(futures):
                if future.result():
                    success += 1
                pbar.update(1)

            pbar.close()

        print(f"--- Image caching complete! Downloaded {success} images. ---")
    else:
        print("Skipping precaching step.")

    # ------------------------------------------------------------------------

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)

    # ------------------------------------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(DEVICE)
    def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(loader, desc='Training'):
            # x, y = x.to(device), y.to(device)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Runs the forward pass in mixed precision (Float16)
            with torch.cuda.amp.autocast(DEVICE):
                outputs = model(x)
                loss = criterion(outputs, y)

            # Scales loss and calls backward() to prevent underflow
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        return total_loss / len(loader)
    def validate_one_epoch(model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(loader, desc='Validation'):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = 100 * correct / total
        return total_loss / len(loader), accuracy

    # ------------------------------------------------------------------------
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(NUM_EPOCHS):
        print(f'\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---')
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        epoch_loss = train_loss / len(train_loader)

        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%')

        # SAVE CHECKPOINT
        checkpoint_path = f"../../models/baseline/checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print('--- Training Complete ---')

    # ------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()