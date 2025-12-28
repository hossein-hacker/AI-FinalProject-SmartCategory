import pandas as pd

# Input and output files
input_file = "amazon_products.csv"
output_file = "amazon_products_minimal.csv"

# Read the original CSV
df = pd.read_csv(input_file)

# Select only the required columns
selected_columns = ["asin", "title", "imgUrl", "category_id"]
df_filtered = df[selected_columns]

# Save to a new CSV file
df_filtered.to_csv(output_file, index=False)

print(f"Saved {len(df_filtered)} rows to '{output_file}'")
