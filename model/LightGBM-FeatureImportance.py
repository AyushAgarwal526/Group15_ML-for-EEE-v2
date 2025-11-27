import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from tqdm import tqdm # Used for displaying progress

# --- Configuration ---
N_FOLDS = 10
RANDOM_STATE = 42
N_ESTIMATORS = 1000 # Reduced for speed, but can be increased for final model

def load_data():
    """
    Loads the real dataset (final_dataset.csv) and prepares features (IQ + Latent Dims) 
    and target (main.disorder).
    """
    try:
        # Load the provided CSV file
        # ASSUMPTION: The file is accessible at this path for execution
        df = pd.read_csv('../data/final_dataset.csv')
    except FileNotFoundError:
        print("Error: 'final_dataset.csv' not found. Please ensure the file is in the correct directory.")
        # Raise an error to stop execution if the required data is missing
        raise

    # Determine feature and target columns based on the file structure and user request
    latent_features = [col for col in df.columns if col.startswith('Latent_Dim_')]
    
    # Input features: IQ and all Latent Dimensions
    FEATURES = ['IQ'] + latent_features
    # Output target: main.disorder
    TARGET = 'main.disorder'

    if not all(f in df.columns for f in FEATURES):
        missing = [f for f in FEATURES if f not in df.columns]
        print(f"Error: Missing required feature columns in CSV: {missing}")
        raise ValueError("Missing required feature columns.")
    
    if TARGET not in df.columns:
        print(f"Error: Missing required target column '{TARGET}' in CSV.")
        raise ValueError("Missing required target column.")

    X_df = df[FEATURES]
    y_df = df[TARGET]
    
    # Handle potential missing values by imputing with the mean (for continuous features)
    X_df = X_df.fillna(X_df.mean())

    print(f"Loaded real data: {X_df.shape[0]} samples, {X_df.shape[1]} features (IQ + Latent Dims).")
    print(f"Target variable: '{TARGET}' with {y_df.nunique()} unique classes.")
    
    return X_df, y_df

def calculate_cv_feature_importance(X, y, model_params):
    """
    Performs K-Fold Cross-Validation, trains a LightGBM model in each fold,
    and aggregates the feature importances. This mimics the 'cv_fi' function logic.
    """
    # Initialize the Stratified K-Fold cross-validator
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Initialize a list to store feature importances from all folds
    all_fi = []
    feature_names = X.columns.tolist()

    # Label encode the target variable if it's not already numeric (LightGBM requirement for multi-class)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nStarting {N_FOLDS}-Fold Cross-Validation for Feature Importance...")

    # Loop through each fold
    for fold_n, (train_index, test_index) in tqdm(enumerate(cv.split(X, y_encoded)), total=N_FOLDS):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Initialize and train the LightGBM model for the current fold
        model = LGBMClassifier(**model_params, random_state=RANDOM_STATE, n_jobs=-1)

        # Use early stopping for robustness
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        # Extract feature importance from the trained model
        fi = model.feature_importances_
        all_fi.append(fi)

    # Convert list of arrays into a DataFrame for easy aggregation
    fi_df = pd.DataFrame(np.array(all_fi), columns=feature_names)

    # --- Aggregation (Sum, Mean, Survival Rate) ---
    # Calculate the mean importance across all folds
    mean_fi = fi_df.mean(axis=0).sort_values(ascending=False).reset_index()
    mean_fi.columns = ['Feature', 'Mean Importance']

    # Calculate Sum Importance (optional, often correlates highly with mean)
    sum_fi = fi_df.sum(axis=0).sort_values(ascending=False).to_frame('Sum Importance')

    # Calculate Survival Rate (the percentage of folds where importance > 0)
    survival_rate = (fi_df > 0).mean(axis=0).sort_values(ascending=False).to_frame('Survival Rate')

    # Combine all metrics into a single results table
    results_df = mean_fi.set_index('Feature')
    results_df['Sum Importance'] = sum_fi['Sum Importance']
    results_df['Survival Rate'] = survival_rate['Survival Rate']

    # Reset index for final output
    results_df = results_df.reset_index()

    return results_df

def plot_feature_importance(fi_df, top_n=20, title='Aggregated LightGBM Feature Importance'):
    """
    Plots the top N features based on Mean Importance and ANNOTATES 
    each bar with its Mean Importance value.
    """
    df_plot = fi_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Horizontal bar plot for better readability of long feature names
    bars = ax.barh(df_plot['Feature'], df_plot['Mean Importance'], color='#1e88e5') # Blue color for LightGBM/LGBM
    ax.set_xlabel("Mean Feature Importance (Split Count)")
    ax.set_ylabel("Feature Name")
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis() # Display the most important feature at the top
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # --- NEW CODE TO ADD TEXT ANNOTATION ---
    for bar in bars:
        # Get the value of the bar (width in a horizontal bar chart)
        width = bar.get_width() 
        # Add the text label to the right of the bar
        ax.text(
            width + 5, # X position: slightly to the right of the bar end
            bar.get_y() + bar.get_height()/2, # Y position: centered vertically on the bar
            f'{width:.0f}', # The text to display (mean importance value, rounded to 0 decimal places)
            ha='left', # Horizontal alignment
            va='center', # Vertical alignment
            fontsize=9 # Smaller font size for clarity
        )
    # ----------------------------------------
    
    # Adjust xlim to make space for the text labels
    # Find the maximum importance value and add a buffer (e.g., 10%)
    max_val = df_plot['Mean Importance'].max()
    ax.set_xlim(right=max_val * 1.15) 

    plt.tight_layout()
    plt.savefig('../plots/LightGBM-Feature-Importance.png', bbox_inches='tight', dpi=300)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Real Data
    X, y = load_data()

    # 2. Define LightGBM Model Parameters
    lgbm_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'n_estimators': N_ESTIMATORS,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1, # Suppress verbose output during training
        # Ensure 'num_class' is correctly set based on the loaded target data
        'num_class': y.nunique() 
    }

    # 3. Calculate CV Feature Importance
    importance_df = calculate_cv_feature_importance(X, y, lgbm_params)

    # 4. Display Results
    print("\n" + "="*50)
    print("TOP 10 AGGREGATED FEATURE IMPORTANCE RESULTS (IQ + Latent Dims)")
    print("="*50)
    # FIX: Replaced .to_markdown() with .to_string() to remove dependency on 'tabulate'
    print(importance_df.head(10).to_string(index=False, float_format="%.4f"))

    # 5. Plot Results
    plot_feature_importance(
        importance_df,
        top_n=20,
        title=f"Top 20 Features (Mean Importance across {N_FOLDS} Folds)"
    )