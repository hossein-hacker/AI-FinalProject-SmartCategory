import pandas as pd
import numpy as np

# --- Configuration ---
# The minimum number of products a merged category must have to be kept.
MIN_PRODUCT_COUNT = 5000

# Input files (the output of the previous scripts)
PRODUCTS_FILE = "data/processed/products_cleaned.csv"
MAPPING_FILE = "data/processed/category_mapping.csv"

# --- Main Script ---

print("--- Starting Final Filtering Step ---")

# 1. Load the datasets processed by the previous scripts
try:
    products_df = pd.read_csv(PRODUCTS_FILE)
    mapping_df = pd.read_csv(MAPPING_FILE)
except FileNotFoundError as e:
    print(f"Error: Could not find input file. Make sure you have run the previous scripts.")
    print(f"File not found: {e.filename}")
    exit()

print(f"Loaded {len(products_df)} products across {products_df['merged_category_id'].nunique()} merged categories.")

# 2. Calculate product counts for each merged category
category_counts = products_df['merged_category_id'].value_counts()

# 3. Identify which merged categories to keep
valid_category_ids = category_counts[category_counts >= MIN_PRODUCT_COUNT].index.tolist()

print(f"Found {len(valid_category_ids)} categories with >= {MIN_PRODUCT_COUNT} products.")

# 4. Filter both DataFrames to keep only the valid categories
original_product_count = len(products_df)
original_category_count = products_df['merged_category_id'].nunique()

products_df = products_df[products_df['merged_category_id'].isin(valid_category_ids)]
mapping_df = mapping_df[mapping_df['merged_category_id'].isin(valid_category_ids)]

print(f"Removed {original_product_count - len(products_df)} products.")
print(f"Removed {original_category_count - len(valid_category_ids)} categories.")


# 5. Re-index the final 'merged_category_id's to be sequential (0, 1, 2, ...)
# This is important for many ML frameworks (like PyTorch) which expect class labels from 0 to N-1.
print("Re-indexing final category IDs to be sequential (0 to N-1)...")

# Create a mapping from the old, non-sequential IDs to the new sequential IDs
unique_ids = products_df['merged_category_id'].unique()
old_to_new_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

# Apply the new sequential IDs
products_df['final_category_id'] = products_df['merged_category_id'].map(old_to_new_id_map)
mapping_df['final_category_id'] = mapping_df['merged_category_id'].map(old_to_new_id_map)

# Drop the intermediate 'merged_category_id' column
products_df = products_df.drop(columns=['merged_category_id'])
mapping_df = mapping_df.drop(columns=['merged_category_id'])

# Rename 'final_category_id' to 'category_id' for simplicity in the final files
products_df.rename(columns={'final_category_id': 'category_id'}, inplace=True)
mapping_df.rename(columns={'final_category_id': 'category_id'}, inplace=True)


# 6. Overwrite the original files with the newly filtered and finalized data
products_df.to_csv(PRODUCTS_FILE, index=False)
mapping_df.to_csv(MAPPING_FILE, index=False)

print("\n--- Filtering Complete ---")
print(f"Final number of products: {len(products_df)}")
print(f"Final number of categories: {products_df['category_id'].nunique()}")
print(f"Files '{PRODUCTS_FILE}' and '{MAPPING_FILE}' have been updated.")

print("\nFinal category distribution:")
print(products_df['category_id'].value_counts().sort_index())
print("--------------------------")
