import pandas as pd

# -----------------------------
# File paths
# -----------------------------
merged_categories_file = "merged_categories.csv"
products_file = "amazon_products_filtered.csv"

new_merged_categories_file = "merged_categories_new.csv"
new_products_file = "amazon_products_merged.csv"

# -----------------------------
# Step 1: Read merged categories
# -----------------------------
merged_df = pd.read_csv(merged_categories_file)

# Assign new merged category IDs
merged_df["merged_category_id"] = range(len(merged_df))

# -----------------------------
# Step 2: Build mapping
# original_category_id -> merged_category_id
# -----------------------------
original_to_merged = {}

for _, row in merged_df.iterrows():
    merged_id = row["merged_category_id"]
    original_ids = row["category_ids"]

    for cid in str(original_ids).split(","):
        original_to_merged[int(cid.strip())] = merged_id

# -----------------------------
# Step 3: Save new merged categories CSV
# -----------------------------
merged_categories_out = merged_df[
    ["merged_category_id", "category_name", "category_ids"]
]

merged_categories_out.to_csv(new_merged_categories_file, index=False)

print(f"Saved new merged categories to '{new_merged_categories_file}'")

# -----------------------------
# Step 4: Read products
# -----------------------------
products_df = pd.read_csv(products_file)
products_df["category_id"] = products_df["category_id"].astype(int)

# -----------------------------
# Step 5: Replace category_id with merged_category_id
# -----------------------------
products_df["merged_category_id"] = products_df["category_id"].map(
    original_to_merged
)

# Optional: drop old category_id
products_df = products_df.drop(columns=["category_id"])

# -----------------------------
# Step 6: Save new products dataset
# -----------------------------
products_df.to_csv(new_products_file, index=False)

print(f"Saved migrated products to '{new_products_file}'")
print(f"Total products: {len(products_df)}")
