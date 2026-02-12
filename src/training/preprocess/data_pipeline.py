import numpy as np
from pandas import DataFrame
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io.image import read_image
from sklearn.model_selection import train_test_split

BATCH_SIZE = 256
NUM_WORKERS = 12

def set_random_seeds():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

def split_data(df, random_state=42):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['merged_category_id'],
        random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.25,
        stratify=temp_df['merged_category_id'],
        random_state=random_state
    )

    return train_df, val_df, test_df

def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform, val_test_transform

def create_datasets(train_df, val_df, test_df, train_transform, val_test_transform):
    train_dataset = ProductImageDataset(train_df, train_transform)
    val_dataset = ProductImageDataset(val_df, val_test_transform)
    test_dataset = ProductImageDataset(test_df, val_test_transform)
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader, test_loader

class ProductImageDataset(Dataset):
    def __init__(self, df: DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['local_path']
        label = int(row['merged_category_id'])

        try:
            img = read_image(img_path)
            img = img.float() / 255.0
        except:
            img = torch.zeros(3, 224, 224)

        if self.transform:
            img = self.transform(img)

        return img, label