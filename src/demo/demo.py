# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# import pandas as pd
# import gradio as gr

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
#         )
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes)
#         )
#     def forward(self, x):
#         return self.classifier(self.features(x))

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# CHECKPOINT_PATH = "../../models/baseline/checkpoint_epoch_10.pth"
# MAPPING_PATH = '../../../data/processed/category_mapping.csv'

# mapping_df = pd.read_csv(MAPPING_PATH)
# full_df = pd.read_csv('../../../data/processed/products_cleaned.csv')
# valid_cats = sorted(full_df['merged_category_id'].value_counts()[lambda x: x >= 25000].index.tolist())
# id_to_name = dict(zip(mapping_df['merged_category_id'], mapping_df['category_name']))
# class_names = [id_to_name[old_id] for old_id in valid_cats]

# model = SimpleCNN(num_classes=len(class_names)).to(DEVICE)
# checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# def classify_product(confused_image):
#     if confused_image is None:
#         return None
    
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     image = Image.fromarray(confused_image.astype('uint8'), 'RGB')
#     input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
#     with torch.no_grad():
#         output = model(input_tensor)
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
#     return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

# demo = gr.Interface(
#     fn=classify_product, 
#     inputs=gr.Image(), 
#     outputs=gr.Label(num_top_classes=3),
#     title="AI Product Category Detector",
#     description="Please upload your picture",
#     theme="soft"
# )

# if __name__ == "__main__":
#     demo.launch(share=True) # گزینه share یک لینک عمومی موقت به شما می‌دهد

import gradio as gr
import numpy as np
import time

class_names = ["کالای دیجیتال", "مد و پوشاک", "خانه و آشپزخانه", "زیبایی و سلامت", "اسباب بازی"]

def fake_classify_product(input_img):
    if input_img is None:
        return None, "0.000 ثانیه"
    
    start_time = time.time()
    
    time.sleep(0.2) 
    
    preds = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
    results = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    
    end_time = time.time()
    duration = f"{end_time - start_time:.3f} ثانیه"
    
    return results, duration

with gr.Blocks(theme=gr.themes.Soft(), title="دموی تشخیص کالا") as demo:
    gr.Markdown("""
    # 🛒 سامانه هوشمند تشخیص دسته‌بندی کالا
    **وضعیت مدل:** در حال نمایش نسخه دمو (خروجی تصادفی)
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="تصویر محصول را اینجا آپلود کنید")
            btn = gr.Button("شروع پردازش", variant="primary")
        
        with gr.Column():
            output_labels = gr.Label(label="پیش‌بینی مدل", num_top_classes=3)
            output_time = gr.Textbox(label="زمان پاسخگویی")

    btn.click(fn=fake_classify_product, inputs=input_image, outputs=[output_labels, output_time])

    gr.Examples(
        examples=[], 
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch()