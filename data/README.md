
# Dataset Information

The full dataset CSV (~280MB) is not stored in the GitHub repository due to its size.

To download the full dataset, run in the project root:

```bash
python data/preprocessing/download_dataset.py
```

The dataset will be downloaded to:
***data/processed/***

The products CSV contains the following columns:
- asin
- title
- imgUrl
- merged_category_id

The categories CSV has:
- category_id (The original amazon category id)
- category_name (The original amazon category name)
- merged_category_name (The new merged category name)
- merged_category_id (The new merged category id)