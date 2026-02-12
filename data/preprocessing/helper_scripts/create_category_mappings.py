import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# --- Configuration ---
# The original file with all 248 categories
CATEGORIES_FILE_PATH = "data/raw/categories.csv"
# The desired number of final, merged categories
NUM_CLUSTERS = 50  # You can tune this number
# The output file that will store the mapping from old to new categories
OUTPUT_MAPPING_FILE = "data/processed/category_mapping.csv"

# --- Main Script ---

print("Starting category merging process...")

# 1. Load the original categories data
try:
    categories_df = pd.read_csv(CATEGORIES_FILE_PATH)
    categories_df.rename(columns={'id': 'category_id'}, inplace=True)
except FileNotFoundError:
    print(f"Error: The file '{CATEGORIES_FILE_PATH}' was not found.")
    print("Please ensure you have the original 'categories.csv' in the 'data/raw/' directory.")
    exit()

print(f"Loaded {len(categories_df)} original categories.")

# 2. Generate sentence embeddings for category names
# This model is lightweight and effective for semantic similarity.
print("Loading sentence transformer model and generating embeddings for category names...")
model = SentenceTransformer('all-MiniLM-L6-v2')
# The embeddings are numerical representations of the category names
category_embeddings = model.encode(categories_df['category_name'].tolist(), show_progress_bar=True)

print(f"Generated {len(category_embeddings)} embeddings with dimension {category_embeddings.shape[1]}.")

# 3. Cluster the embeddings to group similar categories
print(f"Clustering categories into {NUM_CLUSTERS} groups using KMeans...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
# The 'fit_predict' method assigns a cluster ID (from 0 to NUM_CLUSTERS-1) to each category
cluster_labels = kmeans.fit_predict(category_embeddings)

# Add the new 'merged_category_id' to our DataFrame
categories_df['merged_category_id'] = cluster_labels

# 4. Create human-readable names for the new merged categories (Optional but recommended)
# We can create a representative name for each cluster by finding the most common words
# among the original category names within that cluster.
merged_category_names = {}
for i in range(NUM_CLUSTERS):
    # Get all original names belonging to the current cluster
    names_in_cluster = categories_df[categories_df['merged_category_id'] == i]['category_name'].tolist()
    # A simple way to name the cluster is to use the name of the first item,
    # or you could implement a more complex naming logic (e.g., find common words).
    # For simplicity, we'll use the name of the category closest to the cluster center.
    center = kmeans.cluster_centers_[i]
    distances = np.linalg.norm(category_embeddings[categories_df['merged_category_id'] == i] - center, axis=1)
    closest_index = np.argmin(distances)
    representative_name = names_in_cluster[closest_index]
    merged_category_names[i] = representative_name

categories_df['merged_category_name'] = categories_df['merged_category_id'].map(merged_category_names)


# 5. Save the final mapping to a new CSV file
# This file is your new source of truth for category definitions.
output_df = categories_df[['category_id', 'category_name', 'merged_category_id', 'merged_category_name']]
output_df.to_csv(OUTPUT_MAPPING_FILE, index=False)

print(f"\nProcess complete!")
print(f"Successfully created the category mapping file at: '{OUTPUT_MAPPING_FILE}'")

# --- Display a preview of the merged categories ---
print("\n--- Preview of Merged Categories ---")
for i in range(NUM_CLUSTERS):
    cluster_name = merged_category_names[i]
    original_names = categories_df[categories_df['merged_category_id'] == i]['category_name'].tolist()
    print(f"\nMerged Category '{cluster_name}' (ID: {i}) includes:")
    for name in original_names:
        print(f"  - {name}")
print("\n------------------------------------")
