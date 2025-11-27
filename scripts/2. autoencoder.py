import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau 

# --- Configuration ---
DATA_FILE = "../data/preprocessed_data.csv" 
ENCODED_OUTPUT_FILE = "../data/eeg_latent_features.csv"

LATENT_DIM = 128 
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 50
MIN_DELTA = 1e-5

# ReduceLROnPlateau parameters
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 20
LR_SCHEDULER_MIN_LR = 1e-6

# --- 1. Data Loading and Preprocessing (Revised for PSD Features) ---

def load_and_prepare_data(filepath):
    """
    Loads the numerical EEG PSD features from the CSV file, starting from 
    'AB.A.delta.a.FP1' to the end of the columns, handles NaNs, and standardizes the data.
    
    Returns: data_tensor (torch.Tensor), scaler (StandardScaler), input_dim (int)
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please ensure the file is accessible.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return None, None, None

    # --- Identify and Select Relevant Columns (from 'I' / AB.A.delta.a.FP1 to 'ARE' / end) ---
    # We assume 'AB.A.delta.a.FP1' is the first feature column.
    start_col = 'AB.A.delta.a.FP1'
    
    if start_col not in df.columns:
        print(f"Error: Starting column '{start_col}' not found. Please verify column names.")
        return None, None, None
        
    # Select all columns from the starting column onwards, assuming they are the features
    start_index = df.columns.get_loc(start_col)
    eeg_df = df.iloc[:, start_index:].copy()
    
    # Check for non-numeric columns and attempt conversion/removal
    numeric_cols = eeg_df.select_dtypes(include=np.number).columns
    if len(numeric_cols) != eeg_df.shape[1]:
        print("Warning: Non-numeric columns found and dropped.")
        eeg_df = eeg_df[numeric_cols]
        
    # --- Data Cleaning ---
    # 1. Handle missing values by simple imputation (mean of the column)
    eeg_df = eeg_df.fillna(eeg_df.mean())

    # 2. FIX: Drop columns that remain all NaN or are constant (zero variance)
    # This prevents the 'ValueError: Input contains NaN' during scikit-learn's checks.
    
    # Drop columns that are still all NaN (mean was NaN because the entire column was NaN)
    cols_to_drop_nan = eeg_df.columns[eeg_df.isna().all()].tolist()
    if cols_to_drop_nan:
        print(f"INFO: Dropping {len(cols_to_drop_nan)} feature column(s) that were entirely NaN and could not be imputed.")
        eeg_df = eeg_df.drop(columns=cols_to_drop_nan)
    
    # Drop constant columns (zero variance), which are useless for modeling and can cause scaler issues
    cols_to_drop_constant = eeg_df.columns[eeg_df.nunique() <= 1].tolist()
    if cols_to_drop_constant:
        print(f"INFO: Dropping {len(cols_to_drop_constant)} feature column(s) that are constant (zero variance).")
        eeg_df = eeg_df.drop(columns=cols_to_drop_constant)

    # Re-check for any remaining NaNs/Infs (final safety net)
    if eeg_df.isna().any().any() or np.isinf(eeg_df).any().any():
        print("CRITICAL WARNING: NaNs or Infs still exist after cleaning. Filling with 0.")
        eeg_df = eeg_df.fillna(0)
        eeg_df = eeg_df.replace([np.inf, -np.inf], 0)

    # --- Standardization ---
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(eeg_df)
    
    input_dim = data_scaled.shape[1]
    
    print(f"Successfully loaded and prepared {data_scaled.shape[0]} samples.")
    print(f"New Feature Vector Dimension (INPUT_DIM): {input_dim}")
    
    # Convert numpy array to PyTorch Tensor (float32 is standard for PyTorch)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    
    return data_tensor, scaler, input_dim

# --- 2. Autoencoder Model Definition (Simplified) ---

class SimplifiedEEGAutoencoder(nn.Module):
    """
    A simple three-layer fully connected autoencoder for dimensionality reduction
    of EEG PSD features.
    
    Encoder: INPUT_DIM -> LATENT_DIM * 2 -> LATENT_DIM
    Decoder: LATENT_DIM -> LATENT_DIM * 2 -> INPUT_DIM
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # --- Encoder ---
        self.encoder = nn.Sequential(
            # First layer: Compresses to a wider dimension
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            # Bottleneck layer
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        # --- Decoder ---
        self.decoder = nn.Sequential(
            # Expands from bottleneck
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            # Output layer: Must match the original input dimension
            nn.Linear(latent_dim * 2, input_dim),
            # No activation on the final layer for reconstruction loss on scaled data.
        )

    def forward(self, x):
        # x is the input features
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed
    
    def encode(self, x):
        """Returns the latent representation."""
        return self.encoder(x)

# --- 3. Early Stopping Class ---

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    For an autoencoder on a single dataset, we monitor the training loss itself.
    """
    def __init__(self, patience=7, min_delta=0, verbose=True, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.path = path
        
    def __call__(self, val_loss, model):
        # NOTE: For this autoencoder setup without a separate validation set,
        # we are using the training loss as the monitored value (val_loss).
        
        if val_loss < self.best_loss - self.min_delta:
            # Improvement detected
            self.best_loss = val_loss
            self.counter = 0
            # Save the model
            self.save_checkpoint(val_loss, model)
        else:
            # No significant improvement
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        # Note: We save a checkpoint model, but for the final output, we'll use the model object directly.
        # This is a standard practice for robust training, though simplified here.
        torch.save(model.state_dict(), self.path)

# --- 4. Training Loop (Updated) ---

def train_model(model, train_loader, criterion, optimizer, scheduler, early_stopper, num_epochs):
    """
    Trains the autoencoder model with Early Stopping and ReduceLROnPlateau.
    """
    print("\nStarting model training...")
    
    history = {'train_loss': []}
    
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_loss = 0
        
        for batch_data in train_loader:
            x = batch_data[0]
            
            # Forward pass:
            outputs = model(x)
            loss = criterion(outputs, x) # Calculate loss
            
            # Backward and optimize:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        # --- Scheduler Step ---
        # Reduce LR if the epoch loss (monitored metric) hasn't improved
        scheduler.step(epoch_loss)

        # --- Early Stopping Check ---
        # Monitor the training loss
        early_stopper(epoch_loss, model)
        
        if (epoch + 1) % 50 == 0 or epoch == 0 or early_stopper.early_stop:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, Current LR: {current_lr:.2e}')

        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Loading best model state...")
            # Load the best model found so far
            model.load_state_dict(torch.load(early_stopper.path))
            break

    print(f"Training complete. Ran for {epoch+1} epochs.")
    return history

# --- 5. Evaluation and Visualization ---

def evaluate_and_visualize(model, data_tensor):
    """
    Evaluates the model and visualizes an example sample's reconstruction.
    """
    model.eval() # Set model to evaluation mode
    
    with torch.no_grad():
        # Reconstruct all data
        reconstructed_tensor = model(data_tensor)
        
        # Convert tensors back to numpy for evaluation and plotting
        original_data_scaled = data_tensor.numpy()
        reconstructed_data_scaled = reconstructed_tensor.numpy()
        
        # Calculate Root Mean Squared Error (RMSE) across the entire dataset
        rmse = np.sqrt(mean_squared_error(original_data_scaled, reconstructed_data_scaled))
        print(f"\nModel Evaluation (Scaled Data):")
        print(f"Overall Reconstruction RMSE: {rmse:.4f}")

        # --- Visualize a single example ---
        
        # Pick a random sample index
        sample_idx = np.random.randint(0, original_data_scaled.shape[0])
        original_sample = original_data_scaled[sample_idx, :]
        reconstructed_sample = reconstructed_data_scaled[sample_idx, :]
        
        num_features_to_plot = original_sample.shape[0]
        feature_indices = np.arange(num_features_to_plot)

        plt.figure(figsize=(20, 6)) 
        
        # Plot the entire feature vector
        plt.plot(feature_indices, original_sample, 
                 label='Original (Scaled)', alpha=0.7, linestyle='-')
        plt.plot(feature_indices, reconstructed_sample, 
                 label='Reconstructed (Scaled)', alpha=0.9, linestyle='--')
        
        plt.title(f'Original vs. Reconstructed Feature Vector (Full Vector, {num_features_to_plot} Features)', fontsize=14)
        plt.xlabel('Feature Index')
        plt.ylabel('Standardized Value')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig('../plots/Autoencoder_Reconstruction_Loss.png')
        plt.show()

# --- 6. Main Execution ---

if __name__ == '__main__':
    # Load and prepare data
    data_tensor, scaler, INPUT_DIM = load_and_prepare_data(DATA_FILE)
    
    if data_tensor is None or INPUT_DIM is None:
        exit()

    # Create PyTorch DataLoader
    dataset = TensorDataset(data_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SimplifiedEEGAutoencoder(INPUT_DIM, LATENT_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Initialize Scheduler and Early Stopper ---
    # FIX: Removed 'verbose=True' from ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=LR_SCHEDULER_FACTOR, 
        patience=LR_SCHEDULER_PATIENCE, 
        min_lr=LR_SCHEDULER_MIN_LR,
        # verbose=True # REMOVED: This is the line that caused the error.
    )
    
    early_stopper = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, 
        min_delta=MIN_DELTA, 
        verbose=True, 
        path='autoencoder_best_model.pt'
    )

    # Train the model (pass new objects)
    history = train_model(model, train_loader, criterion, optimizer, scheduler, early_stopper, NUM_EPOCHS)

    # Evaluate and visualize results
    evaluate_and_visualize(model, data_tensor)

    # --- Save Latent Features ---
    model.eval()
    with torch.no_grad():
        latent_features = model.encode(data_tensor).numpy()
    
    latent_df = pd.DataFrame(latent_features, 
                             columns=[f'Latent_Dim_{i+1}' for i in range(LATENT_DIM)])
    latent_df.to_csv(ENCODED_OUTPUT_FILE, index=False)
    print(f"\nLatent features saved to {ENCODED_OUTPUT_FILE} (Dimension: {LATENT_DIM})")