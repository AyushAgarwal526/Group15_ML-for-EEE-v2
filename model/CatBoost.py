import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Import the CatBoost Classifier
from catboost import CatBoostClassifier

# --- Configuration ---
# Note: n_estimators in RF is equivalent to iterations in CatBoost.
N_ITERATIONS = 1000 
RANDOM_STATE = 42

# Define feature lists
LATENT_FEATURES = [f'Latent_Dim_{i}' for i in range(1, 129)]
EEG_ONLY_FEATURES = LATENT_FEATURES
EEG_IQ_FEATURES = LATENT_FEATURES + ['IQ']

# List to store all results (maintaining the original name for consistency)
all_accuracy_results = []
# ---------------------

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


def run_catboost_experiment(X_data, y_data, experiment_name, results_list):
    """
    Splits data, trains a CatBoostClassifier, evaluates performance,
    and updates the results list.
    """
    print(f"\n--- Running Experiment: {experiment_name} on Target: {y_data.name} ---")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=RANDOM_STATE, stratify=y_data
    )
    
    # Initialize the CatBoost Classifier
    # Since the target is categorical (disorder names), we use MultiClass loss function.
    # We set verbose=False to suppress the training log output for cleaner reports.
    model = CatBoostClassifier(
        iterations=N_ITERATIONS,
        random_state=RANDOM_STATE,
        loss_function='MultiClass', # For multi-class classification
        eval_metric='Accuracy',
        verbose=False, # Suppress training logs
        early_stopping_rounds=100 # Stop early if validation performance doesn't improve
    )

    # Train the model
    # Note: CatBoost handles the multi-output classification automatically.
    print("Training CatBoostClassifier...")
    # Using 'fit' directly. CatBoost automatically handles any non-numeric features
    # but here all features are numeric (Latent_Dim_i and IQ).
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test).flatten() # Predictions are sometimes nested arrays, flatten it.

    # Evaluate the model's performance
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {overall_accuracy:.4f}")

    # Print Classification Report
    # zero_division=1 ensures that if a class has no predictions/true samples,
    # its metric shows 1.0 instead of a warning, matching the original script's behavior.
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
run_catboost_experiment(
    X_data=df[EEG_ONLY_FEATURES], 
    y_data=df['main.disorder'], 
    experiment_name='EEG only', 
    results_list=all_accuracy_results
)

# ==============================================================================
# 2. EEG Only -> specific.disorder
# ==============================================================================
run_catboost_experiment(
    X_data=df[EEG_ONLY_FEATURES], 
    y_data=df['specific.disorder'], 
    experiment_name='EEG only', 
    results_list=all_accuracy_results
)

# ==============================================================================
# 3. EEG + IQ -> main.disorder
# ==============================================================================
run_catboost_experiment(
    X_data=df[EEG_IQ_FEATURES], 
    y_data=df['main.disorder'], 
    experiment_name='EEG+IQ', 
    results_list=all_accuracy_results
)

# ==============================================================================
# 4. EEG + IQ -> specific.disorder
# ==============================================================================
run_catboost_experiment(
    X_data=df[EEG_IQ_FEATURES], 
    y_data=df['specific.disorder'], 
    experiment_name='EEG+IQ', 
    results_list=all_accuracy_results
)

# --- Final Summary ---
print("\n" + "=" * 60)
print("FINAL SUMMARY OF ALL CATBOOST EXPERIMENTS")
print("=" * 60)
for result in all_accuracy_results:
    print(f"| {result['experiment_name']:<10} | Target: {result['target_variable']:<20} | Accuracy: {result['final_accuracy']:.4f}")
print("=" * 60)