
# Dataset Information

The full dataset CSV (~140MB) is not stored in the GitHub repository due to its size.

To download the full dataset, run in the project root:

```bash
python src/preprocessing/download_csv.py
```

The dataset will be downloaded to:
*data/raw/products.csv*

**Note:** There is a *products_test.csv* dataset just for testing included in *data/raw* directory.

The CSV contains the following columns:
- asin
- title
- imageUrl
- category_id

**Note:** the categories themselves are in the categories.csv in the *data/raw* directory, and doesn't need any download.