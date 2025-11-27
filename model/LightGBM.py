import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
from lightgbm import LGBMClassifier

# --- Configuration ---
# Note: iterations in CatBoost is equivalent to n_estimators in LightGBM.
N_ESTIMATORS = 5000 
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 100
# ---------------------

# Define feature lists
LATENT_FEATURES = [f'Latent_Dim_{i}' for i in range(1, 129)]
EEG_ONLY_FEATURES = LATENT_FEATURES
EEG_IQ_FEATURES = LATENT_FEATURES + ['IQ']

# List to store all results
all_accuracy_results = []

# Load the dataset
try:
    # Assuming the path is correct based on the original script
    df = pd.read_csv('../data/final_dataset.csv')
except FileNotFoundError:
    print("Error: 'final_dataset.csv' not found. Please ensure the path is correct.")
    # Create a mock dataframe for demonstration purposes if file is missing
    print("Using mock data for demonstration.")
    data = {
        **{f'Latent_Dim_{i}': np.random.rand(100) for i in range(1, 129)},
        'IQ': np.random.randint(70, 130, 100),
        'main.disorder': np.random.choice(['ADHD', 'ASD', 'Control'], 100),
        'specific.disorder': np.random.choice(['Type1', 'Type2', 'Type3', 'Control'], 100)
    }
    df = pd.DataFrame(data)


def run_lightgbm_experiment(X_data, y_data, experiment_name, results_list):
    """
    Splits data, trains a LGBMClassifier with early stopping, evaluates performance,
    and updates the results list.
    """
    print(f"\n--- Running Experiment: {experiment_name} on Target: {y_data.name} ---")

    # 1. Split full data into training (80%) and testing (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=RANDOM_STATE, stratify=y_data
    )
    
    # 2. Further split the training data into actual training (90%) and validation (10%) 
    #    The validation set is required for LightGBM's early stopping.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train_full
    )

    # Initialize the LGBM Classifier
    # 'multiclass' objective is used because the target variables are categorical (more than 2 classes).
    # verbose=-1 suppresses the per-iteration training log for a cleaner output.
    model = LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        objective='multiclass', 
        n_jobs=-1, # Use all available cores
        verbose=-1 # Suppress output
    )

    # Train the model with early stopping
    print("Training LGBMClassifier...")
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], # Validation set for early stopping
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        eval_metric='multi_logloss' # The metric to monitor for early stopping
    )

    # Make predictions on the test data
    y_pred = model.predict(X_test) 

    # Evaluate the model's performance
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {overall_accuracy:.4f}")

    # Print Classification Report
    classification_rep = classification_report(y_test, y_pred, zero_division=1)
    print("\nClassification Report:\n", classification_rep)

    # Append results to the master list
    results_list.append({
        'experiment_name': experiment_name,
        'target_variable': y_data.name,
        'final_accuracy': overall_accuracy
    })
    
    print("-" * 60)


# ==============================================================================
# 1. EEG Only -> main.disorder
# ==============================================================================
run_lightgbm_experiment(
    X_data=df[EEG_ONLY_FEATURES], 
    y_data=df['main.disorder'], 
    experiment_name='EEG only', 
    results_list=all_accuracy_results
)

# ==============================================================================
# 2. EEG Only -> specific.disorder
# ==============================================================================
run_lightgbm_experiment(
    X_data=df[EEG_ONLY_FEATURES], 
    y_data=df['specific.disorder'], 
    experiment_name='EEG only', 
    results_list=all_accuracy_results
)

# ==============================================================================
# 3. EEG + IQ -> main.disorder
# ==============================================================================
run_lightgbm_experiment(
    X_data=df[EEG_IQ_FEATURES], 
    y_data=df['main.disorder'], 
    experiment_name='EEG+IQ', 
    results_list=all_accuracy_results
)

# ==============================================================================
# 4. EEG + IQ -> specific.disorder
# ==============================================================================
run_lightgbm_experiment(
    X_data=df[EEG_IQ_FEATURES], 
    y_data=df['specific.disorder'], 
    experiment_name='EEG+IQ', 
    results_list=all_accuracy_results
)

# --- Final Summary ---
print("\n" + "=" * 60)
print("FINAL SUMMARY OF ALL LIGHTGBM EXPERIMENTS")
print("=" * 60)
for result in all_accuracy_results:
    print(f"| {result['experiment_name']:<10} | Target: {result['target_variable']:<20} | Accuracy: {result['final_accuracy']:.4f}")
print("=" * 60)