import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import hashlib
import requests
from io import BytesIO
import time

LOG_FAILED_DOWNLOADS = True

def _url_to_filename(url):
    """
    Create a deterministic filename from a URL using hashing.
    """
    return hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg"

def load_image_from_url(
    url,
    timeout=5,
    retries=3,
    backoff=1.0,
    cache_dir='../data/raw/images/'
):
    """
    Fetch an image from a URL with caching and retries.

    If the image was previously downloaded, it is loaded from disk
    instead of downloading again.
    """
    if not url or not isinstance(url, str):
        return None
    
    filename = _url_to_filename(url)
    filepath = os.path.join(cache_dir, filename)
    # os.makedirs(cache_dir, exist_ok=True)

    # 1 -> Load from cache if exists
    if os.path.exists(filepath):
        try:
            return Image.open(filepath).convert("RGB")
        except Exception:
            # Corrupted cache → remove and re-download
            os.remove(filepath)

    # print("Not Cached, Downloading", url)
    # 2 -> Download with retries
    # for attempt in range(1, retries + 1):
    #     try:
    #         response = requests.get(url, timeout=timeout)
    #         response.raise_for_status()

    #         if "image" not in response.headers.get("Content-Type", ""):
    #             raise ValueError("URL did not return an image")

    #         img = Image.open(BytesIO(response.content)).convert("RGB")
    #         img.save(filepath, format="JPEG", quality=90)
    #         return img

    #     except Exception:
    #         if attempt < retries:
    #             time.sleep(backoff)
    #         else:
    #             if LOG_FAILED_DOWNLOADS:
    #                 print(f"\nFailed to download {url} after {retries} attempts.")
    #             return None

class ProductImageDataset(Dataset):
    def __init__(self, df, transform=None, image_dir=None):
        self.df = df
        self.transform = transform
        self.image_dir = image_dir 

        if self.image_dir:
            os.makedirs(self.image_dir, exist_ok=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        local_path = row['local_path']
        
        img = None
        # Fast path: Try direct load
        if local_path and os.path.exists(local_path):
            try:
                img = Image.open(local_path).convert("RGB")
            except:
                img = None
        
        # Fallback (Safety Only)
        # if img is None:
        #     # Try loading via helper (but helper now disables download)
        #     img = load_image_from_url(row['imgUrl'], cache_dir=self.image_dir)
            
        # Last Resort: Black Image (Prevents crashing)
        if img is None:
            img = Image.new('RGB', (224, 224), color='black')
            
        label = int(row['merged_category_id'])
        
        if self.transform:
            img = self.transform(img)
            
        return img, label