import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from copy import deepcopy

# --- Helper Function for Renaming Columns ---
def reformat_name(name):
    '''
    Reformat column names from:
    - XX.X.band.x.channel to band.channel
    - COH.X.band.x.channel1.x.channel2 to COH.band.channel1.channel2
    '''
    splitted = name.split(sep='.')
    if len(splitted) < 5:
        return name
    if splitted[0] != 'COH':
        # Handles Power Spectral Density (PSD) features
        result = f'{splitted[2]}.{splitted[4]}'
    else:
        # Handles Functional Connectivity (FC/Coherence) features
        result = f'{splitted[0]}.{splitted[2]}.{splitted[4]}.{splitted[6]}'
    return result

# --- Function to Separate Data into Binary Subsets ---
def sep_to_bin(features, target, target_ord, disorders, hc_id=0):
    '''
    Creates binary classification datasets (Disorder vs. Healthy Control) for each disorder.
    '''
    assert len(features) == len(target)
    X_subsets = {}
    Y_subsets = {}

    for disorder in disorders:
        # Find how the disorder was encoded
        disorder_id = np.where(target_ord == disorder)[0][0]

        # Extract targets for Healthy Control (hc_id=0) and the specific disorder (disorder_id)
        y = target[target.isin([hc_id, disorder_id])].copy(deep=True)

        # Convert the target to binary: Healthy Control=0, Disorder=1
        y[y != hc_id] = 1

        # Extract corresponding features
        x = features.loc[y.index].copy(deep=True)

        # Save
        X_subsets[disorder] = x
        Y_subsets[disorder] = y

    return X_subsets, Y_subsets

# --- Main Preprocessing Steps ---
def run_preprocessing(data_path='dataset.csv', save_path=None):
    # 1. Read Data
    df = pd.read_csv(data_path)

    # Store original column names for the final saved file header
    original_df = df.copy(deep=True)

    # Map of simplified names to original names for use in Step 9
    original_to_simplified_names = {col: reformat_name(col) for col in df.columns}
    simplified_to_original_names = {v: k for k, v in original_to_simplified_names.items()}

    # 2. Rename Columns
    df.rename(reformat_name, axis=1, inplace=True)

    # 3. Typo Fix
    typo_ind = df[df['specific.disorder'] == 'Obsessive compulsive disorder'].index
    df.loc[typo_ind, 'specific.disorder'] = 'Obsessive compulsive disorder'

    # Create a working copy of the dataframe with renamed columns
    df_processed = df.copy(deep=True)

    # Identify separation column for dropping (contains all NaNs)
    missing = df_processed.isna().sum()
    try:
        sep_col = missing[missing == df_processed.shape[0]].index[0] 
        drop_cols_base = ['no.', 'eeg.date', sep_col] 
    except IndexError:
        print("Warning: No fully NaN column found to drop as separator.")
        drop_cols_base = ['no.', 'eeg.date']
    
    # Define columns
    target_col = ['main.disorder', 'specific.disorder']
    cat_col = ['sex', 'main.disorder','specific.disorder']
    mv_cols = ['education', 'IQ']
    drop_cols = drop_cols_base

    # 4. Handle Categorical Columns and Targets (Ordinal Encoding)
    hc = 'Healthy control'
    md_unique = df_processed['main.disorder'].unique()
    sd_unique = df_processed['specific.disorder'].unique()

    # Order categories with 'Healthy control' as the first category (ID=0)
    md = md_unique[md_unique != hc]
    sd = sd_unique[sd_unique != hc]
    md_ord = np.insert(md, 0, hc)
    sd_ord = np.insert(sd, 0, hc)
    sex_ord = df_processed['sex'].unique()

    # Encoder setup
    enc = OrdinalEncoder(categories=[sex_ord, md_ord, sd_ord])
    df_processed[cat_col] = enc.fit_transform(df_processed[cat_col])

    # Save encoded targets
    md_target = df_processed['main.disorder'].rename('main.disorder')
    sd_target = df_processed['specific.disorder'].rename('specific.disorder')

    # Create the final feature set X (by dropping targets and other unnecessary columns)
    X = df_processed.drop(drop_cols + target_col, axis=1)

    # 6. Impute Missing Values (Median strategy)
    imputer = SimpleImputer(strategy='median')
    X[mv_cols] = imputer.fit_transform(X[mv_cols])

    # 7. Log-Transform Numerical Features (excluding demographics)
    logtrans_cols = X.drop(mv_cols + ['sex'], axis=1).columns
    X[logtrans_cols] = np.log(X[logtrans_cols])

    # 8. Create Binary Subsets (using the final X and encoded targets)
    Xmd, Ymd = sep_to_bin(X, df_processed['main.disorder'], md_ord, md)
    Xsd, Ysd = sep_to_bin(X, df_processed['specific.disorder'], sd_ord, sd)

    # 9. Save the full preprocessed dataset if a path is provided
    if save_path:
        # Get columns to keep from the original, un-renamed DataFrame
        original_drop_cols = [simplified_to_original_names.get(col, col) for col in drop_cols_base]
        cols_to_keep = [col for col in original_df.columns if col not in original_drop_cols]
        final_df = original_df[cols_to_keep].copy()

        # 9a. FIX: Ensure X has original column names for assignment
        eeg_feature_cols_simplified = X.drop(mv_cols + ['sex'], axis=1).columns
        reverse_map = {simplified: original for original, simplified in original_to_simplified_names.items() 
                       if simplified in eeg_feature_cols_simplified}
        X_final = X.rename(columns=reverse_map)

        # 9b. Replace the feature columns with the imputed, log-transformed features (X_final)
        X_cols_final = X_final.columns
        final_df[X_cols_final] = X_final[X_cols_final] 

        # 9c. FIX: Overwrite the original string target columns with the encoded numeric values
        # The columns are overwritten in place, so no duplicates are created.
        final_df['main.disorder'] = md_target.values
        final_df['specific.disorder'] = sd_target.values

        # 9d. Rename the *encoded* target columns to reflect the encoding
        final_df.rename(columns={
            'main.disorder': 'main.disorder',
            'specific.disorder': 'specific.disorder'
        }, inplace=True)

        final_df.to_csv(save_path, index=False)
        print(f"Full preprocessed data saved to: {save_path}")

    return {
        'X_full': X, 
        'Y_main_disorder_encoded': md_target,
        'Y_specific_disorder_encoded': sd_target,
        'X_main_disorder_subsets': Xmd,
        'Y_main_disorder_subsets': Ymd,
        'X_specific_disorder_subsets': Xsd,
        'Y_specific_disorder_subsets': Ysd,
        'main_disorder_order': md_ord,
        'specific_disorder_order': sd_ord
    }

if __name__ == '__main__':
    # This block is for example usage and will not run without the actual data file.
    try:
        # Example usage:
        preprocessed_data = run_preprocessing(data_path='../data/raw_dataset.csv', save_path='../data/preprocessed_data.csv')
        print("Preprocessing complete. Available keys:")
        print(preprocessed_data.keys())
        print("\nFirst 5 rows of the full preprocessed feature set (X_full has simplified names):")
        print(preprocessed_data['X_full'].head())
        print(f"\nExample subset (Schizophrenia vs. Healthy Control) features shape: {preprocessed_data['X_main_disorder_subsets']['Schizophrenia'].shape}")
        print(f"Example subset (Schizophrenia vs. Healthy Control) targets shape: {preprocessed_data['Y_main_disorder_subsets']['Schizophrenia'].shape}")
    except FileNotFoundError:
        print("\n*** File Not Found Error ***")
        print("Please ensure 'dataset.csv' or the path specified in run_preprocessing (e.g., '../data/raw_dataset.csv') exists to run the main block.")