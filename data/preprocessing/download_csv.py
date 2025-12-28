import os
import gdown

FILE_ID = "1vJrbYs7zHtvwBptapWaLKC7rkMw77c7K"
OUTPUT_PATH = "data/raw/products.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

url = f"https://drive.google.com/uc?id={FILE_ID}"

if os.path.exists(OUTPUT_PATH):
    print("Dataset already exists. Skipping download.")

print("Downloading dataset from Google Drive:\n")
try:
    gdown.download(
        url,
        OUTPUT_PATH,
        quiet=False,
        fuzzy=True
    )
    print(f"\nDataset downloaded successfully to {OUTPUT_PATH}")
except Exception as e:
    print(f"An error occurred during download: {e}")
    raise
