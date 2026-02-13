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
from torchvision import models
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

def clean_df(df, project_root):
    IMAGE_DIR = project_root / 'data' / 'raw' / 'images'
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
        lambda u: str(IMAGE_DIR / _url_to_filename(u)) if isinstance(u, str) and u else ''
    )
    
    try:
        if os.path.exists(FAILED_URLS_PATH):
            with open(FAILED_URLS_PATH, 'r') as f:
                failed_urls = [line.strip() for line in f if line.strip()]
            df = df[~df['imgUrl'].isin(failed_urls)]
    except FileNotFoundError:
        pass
        
    df = df[df['local_path'].notna()]
    df = df[df['local_path'] != ""]
    return df

def evaluate_model():
    set_random_seeds()

    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    
    CSV_PATH = project_root / "data" / "processed" / "products_cleaned.csv"
    MAPPING_PATH = project_root / "data" / "processed" / "category_mapping.csv"
    MODEL_PATH = project_root / "src" / "models" / "main" / "best_model.pth"
    REPORT_DIR = project_root / "Reports" / "phase 2" / "evaluation"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    df = clean_df(df, project_root)
    
    unique_original_ids = sorted(pd.read_csv(CSV_PATH)['merged_category_id'].value_counts().pipe(lambda x: x[x >= 25000]).index.tolist())
    class_names = []
    for i, orig_id in enumerate(unique_original_ids):
        name = mapping_df[mapping_df['merged_category_id'] == orig_id]['merged_category_name'].iloc[0]
        class_names.append(f"{i}: {name}")


    print("Getting DataLoaders...")
    train_df, val_df, test_df = split_data(df)
    train_transform, val_test_transform = get_data_transforms()
    train_dataset, valid_dataset, test_dataset = create_datasets(train_df, val_df, test_df, train_transform, val_test_transform)
    _, _, test_loader = get_dataloaders(train_dataset, valid_dataset, test_dataset)

    print("Loading model weights...")
    # Added weights_only=False to silence PyTorch warning
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    num_classes = checkpoint['num_classes']
    
    model = build_resnet18(num_classes, DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Added zero_division=0 to silence the sklearn warning
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).mean()

    print(f"\nTest Accuracy: {accuracy:.2f}%")

    with open(REPORT_DIR / "classification_report.txt", "w") as f:
        f.write(f"MODEL EVALUATION REPORT\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write("-" * 50 + "\n")
        f.write("Class Mapping (ID in Model : Original Name):\n")
        for name in class_names:
            f.write(f"{name}\n")
        f.write("-" * 50 + "\n")
        f.write(report)

    plt.figure(figsize=(15, 12))
    sns.heatmap(confusion_matrix(all_labels, all_preds), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Acc: {accuracy:.2f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "test_evaluation_result.png")
    plt.show()
    print(f"Evaluation complete. Files saved in {REPORT_DIR}")

if __name__ == "__main__":
    evaluate_model()