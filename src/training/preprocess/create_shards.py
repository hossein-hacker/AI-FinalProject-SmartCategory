import hashlib
import os
import pandas as pd
import webdataset as wds
from tqdm import tqdm
from data_pipeline import split_data

IMAGE_DIR = '../../../data/raw/images/'
CSV_PATH   = "../../../data/processed/products_cleaned.csv"
OUTPUT_DIR = "../../../data/raw/webdataset_shards"
FAILED_URLS_PATH = 'failed_urls.txt'
SHARD_SIZE = 5000  # images per shard

# Create split directories
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR   = os.path.join(OUTPUT_DIR, "val")
TEST_DIR  = os.path.join(OUTPUT_DIR, "test")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def clean_df(df):
    def _url_to_filename(url):
        return hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg"

    cat_counts = df['merged_category_id'].value_counts()
    valid_cats = cat_counts[cat_counts >= 25000].index.tolist()
    df = df[df['merged_category_id'].isin(valid_cats)]
    df = df.groupby('merged_category_id', group_keys=False).sample(
        n=25000,
        random_state=42
    )
    unique_cats = sorted(df['merged_category_id'].unique())
    old_to_new_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_cats)}
    df['merged_category_id'] = df['merged_category_id'].map(old_to_new_mapping)
    df['local_path'] = df['imgUrl'].fillna('').apply(
        lambda u: os.path.join(IMAGE_DIR, _url_to_filename(u)) if isinstance(u, str) and u else ''
    )
    try:
        with open(FAILED_URLS_PATH, 'r') as f:
            failed_urls  = [line.strip() for line in f if line.strip()]
        df = df[~df['imgUrl'].isin(failed_urls)]
    except FileNotFoundError:
        pass

    df = df[df['local_path'].notna()]
    df = df[df['local_path'] != ""]

    return df

def write_shards(df, output_dir, prefix):
    print(f"Writing {prefix} shards... Total samples: {len(df)}")
    with wds.ShardWriter(
        os.path.join(output_dir, f"{prefix}-%05d.tar"),
        maxcount=SHARD_SIZE
    ) as sink:

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                with open(row['local_path'], "rb") as f:
                    image_bytes = f.read()

                sample = {
                    "__key__": str(idx),
                    "jpg": image_bytes,
                    "cls": str(int(row['merged_category_id']))
                }

                sink.write(sample)

            except Exception:
                continue

df = pd.read_csv(CSV_PATH)
df = clean_df(df)

print("Total samples:", len(df))

train_df, val_df, test_df = split_data(df)

write_shards(train_df, TRAIN_DIR, "train")
write_shards(val_df, VAL_DIR, "val")
write_shards(test_df, TEST_DIR, "test")

print("All shards created successfully.")