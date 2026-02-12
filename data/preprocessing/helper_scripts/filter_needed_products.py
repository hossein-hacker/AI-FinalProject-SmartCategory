import pandas as pd

# File paths
merged_categories_file = "merged_categories.csv"
products_file = "amazon_products_minimal.csv"
output_file = "amazon_products_filtered.csv"

# -----------------------------
# Step 1: Read merged categories
# -----------------------------
merged_df = pd.read_csv(merged_categories_file)

# NEW column name
ids_column = "category_ids"

# Collect all allowed category IDs
allowed_category_ids = set()

for ids in merged_df[ids_column]:
    for cid in str(ids).split(","):
        allowed_category_ids.add(int(cid.strip()))

print(f"Allowed category IDs ({len(allowed_category_ids)}): {sorted(allowed_category_ids)}")

# -----------------------------
# Step 2: Read products
# -----------------------------
products_df = pd.read_csv(products_file)

# Make sure category_id is integer
products_df["category_id"] = products_df["category_id"].astype(int)

# -----------------------------
# Step 3: Filter products
# -----------------------------
filtered_products = products_df[
    products_df["category_id"].isin(allowed_category_ids)
]

# -----------------------------
# Step 4: Save filtered products
# -----------------------------
filtered_products.to_csv(output_file, index=False)

print(f"Filtered products saved to '{output_file}'")
print(f"Rows before: {len(products_df)}")
print(f"Rows after:  {len(filtered_products)}")


print(f"---\nSanity check:\n")
print(
    filtered_products["category_id"]
    .value_counts()
    .sort_index()
)
