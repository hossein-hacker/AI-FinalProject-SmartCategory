import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataset_utils import load_image_from_url

IMAGE_DIR = '../data/raw/images/'

# 1. Read the failed URLs from the text file
retry_urls_path = 'retry_urls.txt'
output_failed_path = 'failed_urls.txt'

try:
    with open(retry_urls_path, 'r') as f:
        urls_to_retry = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"Retry file {retry_urls_path} not found. No URLs to retry.")
    urls_to_retry = []

print(f"Attempting to re-download {len(urls_to_retry)} images ...")

def download_and_check(url):
    """Worker function to download one image and check success"""
    try:
        # Try download (load_image_from_url uses requests.get internally)
        _ = load_image_from_url(url, retries=5, cache_dir=IMAGE_DIR)
        
        # Verify the file exists on disk
        filename = hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg"
        filepath = os.path.join(IMAGE_DIR, filename)
        
        if os.path.exists(filepath):
            return None # Success
        else:
            return url # Failed
    except Exception:
        return url # Failed

still_failed = []

# 2. Execute with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=16) as executor:
    # Map the function to the list of URLs
    futures = {executor.submit(download_and_check, url): url for url in urls_to_retry}
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Multithreaded Retry"):
        result = future.result()
        if result:
            still_failed.append(result)

with open(output_failed_path, 'a') as f:
    for url in still_failed:
        f.write(f"{url}\n")

print(f"\nRetry finished. {len(urls_to_retry) - len(still_failed)} fixed. {len(still_failed)} permanently failed.")
print(f"Writing the failed urls in {output_failed_path}")