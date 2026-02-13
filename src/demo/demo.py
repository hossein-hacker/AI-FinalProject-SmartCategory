import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gradio as gr
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
CSV_PATH = project_root / "data" / "processed" / "products_cleaned.csv"
MODEL_PATH = project_root / "src" / "models" / "main" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_resnet18(num_classes: int, device: torch.device) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)

def get_class_names():
    df = pd.read_csv(CSV_PATH)
    cat_counts = df['merged_category_id'].value_counts()
    valid_cats = sorted(cat_counts[cat_counts >= 25000].index.tolist())
    return [f"Category {cat}" for cat in valid_cats]

class_names = get_class_names()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = build_resnet18(checkpoint['num_classes'], DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_product(input_img):
    if input_img is None:
        return None
    
    image = Image.fromarray(input_img.astype('uint8'), 'RGB')
    input_tensor = test_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=classify_product, 
    inputs=gr.Image(), 
    outputs=gr.Label(num_top_classes=3),
    title="🛒 AI Product Category Detector",
    description=f"Project: ResNet18 | Target: {len(class_names)} Categories",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=True)