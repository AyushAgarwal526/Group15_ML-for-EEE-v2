# Group15_ML-for-EEE-v2

This repository contains machine learning models and datasets for analyzing EEG data, potentially related to cognitive performance or neurological disorders. The project leverages Python and various machine learning libraries to perform tasks such as classification, feature importance analysis, and sequence modeling.

## Key Features & Benefits

*   **EEG Data Analysis:** Process and analyze EEG data using machine learning techniques.
*   **Multiple Models:** Includes implementations of CNN-LSTM, CatBoost, LightGBM, and Random Forest models.
*   **Feature Importance:** Identify important features in EEG data using LightGBM.
*   **Binary Classification:** Implements binary classification models for disorder detection.
*   **Data Preprocessing:** Includes preprocessing steps for cleaning and preparing the data.

## Prerequisites & Dependencies

Before running the code in this repository, ensure you have the following installed:

*   **Python:** (>=3.6)
*   **Libraries:**

    ```bash
    pip install pandas numpy scikit-learn tensorflow lightgbm catboost
    ```

    Specifically:

    *   `pandas`
    *   `numpy`
    *   `scikit-learn` (`sklearn`)
    *   `tensorflow`
    *   `lightgbm`
    *   `catboost`
    *   `matplotlib`
    *   `tqdm`

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/AyushAgarwal526/Group15_ML-for-EEE-v2.git
    cd Group15_ML-for-EEE-v2
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt # If you create a requirements.txt, otherwise install manually using the command above
    ```

3.  **Download the datasets:**  The necessary `.csv` files should already be present in the `data/` directory.  If they are missing, ensure they are downloaded and placed in the `data/` folder.

## Usage Examples & API Documentation

### Running the Models

The `model/` directory contains Python scripts for each model.  Each script can be run independently.

*   **CNN-LSTM:**

    ```bash
    python model/CNN-LSTM.py
    ```

*   **CatBoost:**

    ```bash
    python model/CatBoost.py
    ```

*   **LightGBM (Binary Classification):**

    ```bash
    python model/LightGBM-BinaryClassification.py
    ```

*   **LightGBM (Feature Importance):**

    ```bash
    python model/LightGBM-FeatureImportance.py
    ```

*   **RandomForest:**

    ```bash
    python model/RandomForest.py
    ```

### Data Loading

The datasets are loaded using `pandas`:

```python
import pandas as pd

data = pd.read_csv("data/final_dataset.csv")
print(data.head())
```

### Feature Selection

Feature selection often involves selecting columns like `LATENT_FEATURES` as defined in the model scripts.

```python
LATENT_FEATURES = [f'Latent_Dim_{i}' for i in range(1, 129)]
X = data[LATENT_FEATURES] # Example
```

## Configuration Options

Several configuration options can be adjusted within the Python scripts, including:

*   **`N_ITERATIONS` (CatBoost, RandomForest):** Number of iterations for the boosting algorithms.
*   **`RANDOM_STATE`:**  Seed for random number generation.
*   **`FILE_PATH`:** Path to the dataset.
*   **`TARGET_COLUMN`:** Name of the target variable column.
*   **`N_FOLDS`:** Number of folds for cross-validation (LightGBM Feature Importance).
*   **Model-specific hyperparameters:**  Learning rate, number of layers, etc.

These parameters are defined at the beginning of each corresponding `.py` file and can be modified to suit different datasets or improve model performance.
