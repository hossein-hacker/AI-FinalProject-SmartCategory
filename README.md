# AI-FinalProject-SmartCategory

An AI based category detector, which identifies the best fitting category based on the given image of the product.

# Project Goal
The goal of this project is to design and implement an intelligent system based on deep learning that, upon receiving an image of a product, automatically recognizes its appropriate category. It can be used in online shopping platforms (such as دیوار, Amazon, BestBuy). The model output will include the final category and the probability of each class.

# Project Setup
1. Create a virtual enviornment in the root of the project using
    ```bash
    # 1. Create the environment (only do this once)
    python -m venv venv

    # 2. Activate the environment 
    # Do this every time you open project

    # On Windows:
    venv\Scripts\activate
    ```

2. Install the required libraries
    ```bash
    # Make sure your (venv) is active
    pip install -r requirements.txt
    ```
    **Note:** make sure you are in the virtual enviornment while installing

## Dataset

Due to GitHub file size limitations, the full dataset CSV is hosted externally on Google Drive.
Please read *data/README.md* for exact instructions.

This ensures full reproducibility without storing large files in the repository.

# Folder Structure
```
AI-FinalProject-ShopVision/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/             # original dataset
│   ├── preprocessing/   # scripts and cleaning stuff for dataset
│   ├── processed/       # cleaned dataset
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_dataset_analysis.ipynb
│   ├── 03_baseline_model.ipynb
│
├── src/
│   ├── preprocessing/
│   │   └── image_preprocess.py
│   │
│   ├── models/
│   │   ├── cnn_baseline.py
│   │   ├── resnet_model.py
│   │
│   ├── training/
│   │   ├── train.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │
│   ├── utils/
│       ├── config.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│
└── demo/
    ├── app.py
```
