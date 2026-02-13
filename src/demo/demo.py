import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gradio as gr
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# --- Path Configuration ---
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
CSV_PATH = project_root / "data" / "processed" / "products_cleaned.csv"
MAPPING_PATH = project_root / "data" / "processed" / "category_mapping.csv"
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
    # 1. Load data and mapping
    df = pd.read_csv(CSV_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    
    # 2. Get the exact same valid categories used in training (>= 25000)
    cat_counts = df['merged_category_id'].value_counts()
    valid_cat_ids = sorted(cat_counts[cat_counts >= 25000].index.tolist())
    
    # 3. Create the list of names in the exact index order (0, 1, 2...)
    names = []
    for cat_id in valid_cat_ids:
        # Find the name corresponding to this merged_category_id
        name = mapping_df[mapping_df['merged_category_id'] == cat_id]['merged_category_name'].iloc[0]
        names.append(name)
    
    return names


print("Loading Model and Mapping...")
class_names = get_class_names()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# Validation check: Does model output match our class list length?
if checkpoint['num_classes'] != len(class_names):
    print(f"⚠️ Warning: Model has {checkpoint['num_classes']} classes, but found {len(class_names)} in CSV.")

model = build_resnet18(checkpoint['num_classes'], DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])

def classify_product(input_img):
    if input_img is None:
        return {"No image uploaded": 1.0}

    image = Image.fromarray(input_img.astype('uint8'), 'RGB')
    input_tensor = test_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    results = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

    return results

demo = gr.Interface(
    fn=classify_product, 
    inputs=gr.Image(),
    outputs="text",
    title="Smart Category Detector",
    description=f"Upload a product image to identify its category. This model was trained on {len(class_names)} specific high-volume categories.",
    theme="soft"
)

print("Number of class names:", len(class_names))
print("Checkpoint num_classes:", checkpoint['num_classes'])
print("Checkpoint keys:", checkpoint.keys())

if __name__ == "__main__":
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    demo.launch(
        share=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )