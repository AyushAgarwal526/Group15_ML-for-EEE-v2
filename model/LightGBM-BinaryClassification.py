import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
FILE_PATH = '../data/final_dataset.csv'
TARGET_COLUMN = 'main.disorder'
IQ_COLUMN = 'IQ'
LATENT_DIM_PREFIX = 'Latent_Dim_'
RANDOM_SEED = 42

def load_data(file_path):
    """Loads the dataset and converts column names to be Python-friendly."""
    try:
        df = pd.read_csv(file_path)
        # Rename columns to remove dots, making them easier to access
        df.columns = df.columns.str.replace('.', '_', regex=False)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure the CSV is in the correct location.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

def preprocess_data(df):
    """
    Prepares the features and identifies unique disorder IDs for OvR classification.
    """
    if df is None:
        return None, None, None

    print("Data preprocessing steps:")

    # 1. Identify unique disorder IDs (excluding 0, which means no disorder)
    main_disorder_col = TARGET_COLUMN.replace('.', '_')
    unique_disorders = sorted(df[main_disorder_col].unique())
    disorder_ids = [did for did in unique_disorders if did != 0]
    print(f" - Found {len(disorder_ids)} unique disorder IDs: {disorder_ids}")

    # 2. Select input features (IQ + all Latent Dimensions)
    latent_cols = [col for col in df.columns if col.startswith(LATENT_DIM_PREFIX)]
    features = [IQ_COLUMN] + latent_cols
    X = df[features]
    print(f" - Selected {len(features)} features: IQ + {len(latent_cols)} latent dimensions.")

    # 3. Handle missing values (if any)
    if X.isnull().sum().any():
        X = X.fillna(X.mean())
        print(" - Missing values filled using column mean.")

    # 4. Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print(" - Features scaled (StandardScaler).")

    return X_scaled_df, df, disorder_ids, main_disorder_col, features

def train_and_evaluate_ovr_lgbm(X, df_original, disorder_ids, main_disorder_col):
    """
    Trains a separate LightGBM binary classifier for each unique disorder (One-vs-Rest).
    """
    trained_models = {}
    performance_summary = {}
    
    print("\n--- Training One-vs-Rest LightGBM Classifiers ---")

    # Define common LightGBM parameters
    params = {
        'objective': 'binary',        
        'metric': 'auc',              
        'boosting_type': 'gbdt',      
        'n_estimators': 5000,         
        'learning_rate': 0.05,        
        'verbose': -1,                
        'n_jobs': -1,                 
        'seed': RANDOM_SEED           
    }
    
    for disorder_id in disorder_ids:
        # Create binary target for the current disorder (OvR)
        # y = 1 if the current disorder is present, 0 otherwise
        y_disorder = (df_original[main_disorder_col] == disorder_id).astype(int)
        
        positive_count = y_disorder.sum()
        if positive_count < 10: # Skip training if too few positive samples (optional guard)
             print(f" - Disorder ID {disorder_id}: Skipped (Only {positive_count} samples).")
             continue

        print(f"\nTraining Classifier for Disorder ID {disorder_id} (Positive Samples: {positive_count})")
        
        # Stratified split ensures the proportion of the target class is maintained
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_disorder, test_size=0.2, random_state=RANDOM_SEED, stratify=y_disorder
        )

        model = lgb.LGBMClassifier(**params)
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)] 
        )

        # Evaluate performance
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        trained_models[disorder_id] = model
        performance_summary[disorder_id] = {
            'auc': auc, 
            'accuracy': accuracy, 
            'confusion_matrix': conf_matrix
        }

        print(f"   -> AUC: {auc:.4f}")
        print(f"   -> Accuracy: {accuracy:.4f}")
        # TN, FP, FN, TP
        # print(f"   -> Confusion Matrix:\n{conf_matrix}")
        
    print("\n--- Overall Performance Summary ---")
    for did, metrics in performance_summary.items():
        print(f"Disorder ID {did}: AUC = {metrics['auc']:.4f}  |  Accuracy = {metrics['accuracy']:.4f}")

    return trained_models, performance_summary

# --- Main execution block ---
if __name__ == "__main__":
    data = load_data(FILE_PATH)
    
    if data is not None:
        X, df_original, disorder_ids, main_disorder_col, features = preprocess_data(data)
        
        if X is not None and len(disorder_ids) > 0:
            # Train and evaluate the multiple models
            trained_models, summary = train_and_evaluate_ovr_lgbm(
                X, df_original, disorder_ids, main_disorder_col
            )
            
            print("\nModel Training Complete.")
            print(f"Trained {len(trained_models)} separate classifiers.")
            
            