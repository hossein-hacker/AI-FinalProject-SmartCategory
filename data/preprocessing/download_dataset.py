"""
This script downloads all necessary datasets from Google Drive.

It is designed to be used in a cloud environment (like Google Colab or Kaggle)
to fetch the data needed for model training, without needing to run the full
preprocessing pipeline every time.
"""
import os
import sys
import gdown

# --- Configuration ---
# all files to be downloaded here.
CONFIG = {
    "category_mapping": {
        "file_id": "1RuHRDpdu424TWkgMHElSimN-xaPMimMt",
        "output_path": "data/processed/category_mapping.csv"
    },
    "products_cleaned": {
        "file_id": "1ENviTklhLsFjusQ-hDAtXexyDRy14t3z",
        "output_path": "data/processed/products_cleaned.csv"
    },
}

def download_file(file_name: str, file_id: str, destination: str):
    """
    Downloads a single file from Google Drive if it doesn't already exist.

    Args:
        file_name (str): The stored file name
        file_id (str): The Google Drive file ID.
        destination (str): The local path where the file will be saved.
    """

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Check if the file already exists to avoid re-downloading
    if os.path.exists(destination):
        print(f"File '{file_name}' already exists at '{destination}'. Skipping.")
        return

    print(f"Downloading '{file_name}' from Google Drive to '{destination}'...")
    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        gdown.download(url, destination, quiet=False, fuzzy=True)
        print(f"-> Successfully downloaded '{file_name}'.\n")
    except Exception as e:
        print(f"-> An error occurred while downloading '{file_name}': {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to download all datasets specified in the CONFIG.
    """
    print("--- Starting Dataset Download ---\n")
    for name, details in CONFIG.items():
        download_file(
            file_name=name,
            file_id=details["file_id"],
            destination=details["output_path"]
        )
    print("\n--- All datasets are ready. ---")

if __name__ == "__main__":
    main()
