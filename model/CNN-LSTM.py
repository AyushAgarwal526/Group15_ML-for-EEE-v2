# pyright: reportMissingImports=false

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Configuration ---
RANDOM_STATE = 42
EPOCHS = 500 # Set a reasonable number of epochs, EarlyStopping will manage this
BATCH_SIZE = 32
CONV_FILTERS = 64
LSTM_UNITS = 64
DROPOUT_RATE = 0.5
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


def build_cnn_lstm_model(input_shape, num_classes):
    """
    Constructs the CNN-LSTM model architecture.
    """
    model = Sequential([
        # 1. 1D Convolutional Layer (expects data in shape [batch, timesteps, features])
        # Here, the 'time-steps' are the number of features, and the 'features' is 1 (univariate per time step).
        Conv1D(filters=CONV_FILTERS, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        
        # 2. LSTM Layer
        LSTM(units=LSTM_UNITS, return_sequences=False), # return_sequences=False for the final LSTM layer
        Dropout(DROPOUT_RATE),
        
        # 3. Output Layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def run_cnn_lstm_experiment(X_data, y_data, experiment_name, results_list):
    """
    Prepares data, builds, trains, and evaluates the CNN-LSTM model.
    """
    print(f"\n--- Running Experiment: {experiment_name} on Target: {y_data.name} ---")

    # --- 1. Data Preprocessing ---
    # a. Label Encoding for Target Variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_data)
    num_classes = len(le.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    # b. Split data
    X_train_raw, X_test_raw, y_train_cat, y_test_cat = train_test_split(
        X_data, y_categorical, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # c. Scaling Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # d. Reshape for CNN-LSTM (shape: [samples, timesteps, features])
    # For 1D CNN and LSTM, we treat the input features as 'timesteps'
    # with a single 'feature' channel (univariate time-series).
    X_train = X_train_scaled[:, :, np.newaxis]
    X_test = X_test_scaled[:, :, np.newaxis]
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input Shape for Model: {input_shape}")
    
    # --- 2. Model Setup and Callbacks ---
    model = build_cnn_lstm_model(input_shape, num_classes)
    # model.summary() # Uncomment to see the model architecture
    
    # Early Stopping: Stop training when validation loss stops improving
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        verbose=1, 
        restore_best_weights=True
    )
    
    # Reduce LR on Plateau: Reduce learning rate when validation loss stops improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=0.00001, 
        verbose=1
    )

    # --- 3. Training ---
    print("Training CNN-LSTM Model...")
    history = model.fit(
        X_train, 
        y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1, # Use 10% of the training data for validation
        callbacks=[early_stopping, reduce_lr],
        verbose=2 # Set to 1 for progress bar, 2 for one line per epoch
    )

    # --- 4. Evaluation ---
    # Make predictions on the test data (uses the best weights restored by EarlyStopping)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Convert true test labels back from one-hot encoding
    y_test_labels = np.argmax(y_test_cat, axis=1)
    
    # Convert numerical predictions/true labels back to original class names
    y_pred_classes = le.inverse_transform(y_pred)
    y_test_classes = le.inverse_transform(y_test_labels)

    # Evaluate the model's performance
    overall_accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f"Final Test Accuracy: {overall_accuracy:.4f}")

    # Print Classification Report
    classification_rep = classification_report(y_test_classes, y_pred_classes, zero_division=1)
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
run_cnn_lstm_experiment(
    X_data=df[EEG_ONLY_FEATURES], 
    y_data=df['main.disorder'], 
    experiment_name='CNN-LSTM: EEG only', 
    results_list=all_accuracy_results
)

# ==============================================================================
# 2. EEG Only -> specific.disorder
# ==============================================================================
run_cnn_lstm_experiment(
    X_data=df[EEG_ONLY_FEATURES], 
    y_data=df['specific.disorder'], 
    experiment_name='CNN-LSTM: EEG only', 
    results_list=all_accuracy_results
)

# ==============================================================================
# 3. EEG + IQ -> main.disorder
# ==============================================================================
run_cnn_lstm_experiment(
    X_data=df[EEG_IQ_FEATURES], 
    y_data=df['main.disorder'], 
    experiment_name='CNN-LSTM: EEG+IQ', 
    results_list=all_accuracy_results
)

# ==============================================================================
# 4. EEG + IQ -> specific.disorder
# ==============================================================================
run_cnn_lstm_experiment(
    X_data=df[EEG_IQ_FEATURES], 
    y_data=df['specific.disorder'], 
    experiment_name='CNN-LSTM: EEG+IQ', 
    results_list=all_accuracy_results
)

# --- Final Summary ---
print("\n" + "=" * 60)
print("FINAL SUMMARY OF ALL CNN-LSTM EXPERIMENTS")
print("=" * 60)
for result in all_accuracy_results:
    print(f"| {result['experiment_name']:<20} | Target: {result['target_variable']:<20} | Accuracy: {result['final_accuracy']:.4f}")
print("=" * 60)