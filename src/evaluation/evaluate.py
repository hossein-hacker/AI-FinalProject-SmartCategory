import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import hashlib
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from data_pipeline import set_random_seeds, split_data, get_data_transforms, create_datasets, get_dataloaders

def build_resnet18(num_classes: int, device: torch.device) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

def clean_df(df):
    IMAGE_DIR = '../../../data/raw/images/'
    FAILED_URLS_PATH = 'failed_urls.txt'
    def _url_to_filename(url):
        return hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg"
    cat_counts = df['merged_category_id'].value_counts()
    valid_cats = cat_counts[cat_counts >= 25000].index.tolist()
    df = df[df['merged_category_id'].isin(valid_cats)]
    df = df.groupby('merged_category_id', group_keys=False).sample(n=25000, random_state=42)
    unique_cats = sorted(df['merged_category_id'].unique())
    old_to_new_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_cats)}
    df['merged_category_id'] = df['merged_category_id'].map(old_to_new_mapping)
    df['local_path'] = df['imgUrl'].fillna('').apply(
        lambda u: os.path.join(IMAGE_DIR, _url_to_filename(u)) if isinstance(u, str) and u else ''
    )
    try:
        with open(FAILED_URLS_PATH, 'r') as f:
            failed_urls = [line.strip() for line in f if line.strip()]
        df = df[~df['imgUrl'].isin(failed_urls)]
    except FileNotFoundError:
        pass
    df = df[df['local_path'].notna()]
    df = df[df['local_path'] != ""]
    return df

class ProductDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['local_path']
        label = row['merged_category_id']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate_model():
    set_random_seeds()

    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    
    CSV_PATH = project_root / "data" / "processed" / "products_cleaned.csv"
    MODEL_PATH = project_root / "models" / "main" / "best_model.pth"
    REPORT_DIR = project_root / "Reports" / "phase 2" / "evaluation"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    df = pd.read_csv(CSV_PATH)
    df = clean_df(df)
    unique_cats = sorted(df['merged_category_id'].unique())
    class_names = [str(cat) for cat in unique_cats]

    _, _, test_df = split_data(df)
    _, test_transform = get_data_transforms()
    _, _, test_dataset = create_datasets(None, None, test_df, None, test_transform)
    None, None, test_loader = get_dataloaders(None, None, test_dataset)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    num_classes = checkpoint['num_classes']
    
    model = build_resnet18(num_classes, DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names)
    accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).mean()

    with open(REPORT_DIR / "classification_report.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n{'='*30}\n{report}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix - Acc: {accuracy:.2f}%')
    plt.savefig(REPORT_DIR / "test_evaluation_result.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model()