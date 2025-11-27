import pandas as pd
import re

# Define the input and output file names
PREPROCESSED_FILE = '../data/preprocessed_data.csv'
LATENT_FEATURES_FILE = '../data/eeg_latent_features.csv'
OUTPUT_FILE = '../data/final_dataset.csv'

def process_and_merge_data(preprocessed_path, latent_path, output_path):
    """
    Loads preprocessed data, removes existing EEG power (AB.) and coherence (COH.) 
    columns, merges with latent features, and saves the result to a new CSV file.
    """
    try:
        # 1. Load the datasets
        print(f"Loading primary data from {preprocessed_path}...")
        df_preprocessed = pd.read_csv(preprocessed_path)
        
        print(f"Loading latent features from {latent_path}...")
        df_latent = pd.read_csv(latent_path)

        # 2. Identify and remove EEG columns from the preprocessed data
        # EEG columns are identified by the prefixes 'AB.' (Absolute Band Power) 
        # and 'COH.' (Coherence).
        
        # We use a regular expression to match columns starting with either 'AB.' or 'COH.'
        pattern_to_remove = r'^(AB\.|COH\.)'
        
        eeg_cols_to_remove = [
            col for col in df_preprocessed.columns 
            if re.match(pattern_to_remove, col)
        ]
        
        print(f"Found {len(eeg_cols_to_remove)} EEG/Coherence columns to remove.")
        
        # Keep columns that are NOT in the list of columns to remove
        df_non_eeg = df_preprocessed.drop(columns=eeg_cols_to_remove, errors='ignore')
        
        print(f"Primary data shape after removing old EEG features: {df_non_eeg.shape}")
        
        # 3. Perform the merge operation
        # Assuming both datasets have the same number of rows and are in the
        # correct corresponding order (a simple column-wise concatenation/join).
        # We use reset_index() for a clean join on the implicit row index.
        if len(df_non_eeg) != len(df_latent):
            print("Warning: The number of rows in the two files do not match. Proceeding with simple index-based merge.")
        
        # Reset indices to ensure a proper column-wise concatenation based on row order
        df_non_eeg = df_non_eeg.reset_index(drop=True)
        df_latent = df_latent.reset_index(drop=True)
        
        # Concatenate the non-EEG data and the latent features horizontally (axis=1)
        df_combined = pd.concat([df_non_eeg, df_latent], axis=1)
        
        print(f"Combined data shape: {df_combined.shape}")

        # 4. Save the combined DataFrame to a new CSV file
        df_combined.to_csv(output_path, index=False)
        
        print(f"\nSuccessfully created and saved the new file: {output_path}")
        print(f"The new file contains {df_combined.shape[0]} rows and {df_combined.shape[1]} columns.")
        
    except FileNotFoundError as e:
        print(f"Error: One of the required files was not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Run the main function
    process_and_merge_data(PREPROCESSED_FILE, LATENT_FEATURES_FILE, OUTPUT_FILE)