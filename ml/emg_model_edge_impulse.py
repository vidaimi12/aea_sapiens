# -*- coding: utf-8 -*-
"""
EMG Angle Prediction with CSV Export
Preprocesses EMG input and targets (moving average + 128-level quantization), saves to CSV
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter
import time
from tqdm import tqdm
from pathlib import Path

# Global configuration
NUM_EPOCHS = 50
WINDOW_SIZE = 125  # 500ms at 250Hz sampling rate
ANGLE_COLUMNS = [
    'thumb_ext_angle',
    'index_ext_angle',
    'middle_ext_angle',
    'ring_ext_angle',
    'pinky_ext_angle',
]

# CSV output configuration
OUTPUT_DIR = "./preprocessed_data"

# ============================================================================
# DATASET CLASS
# ============================================================================

class EMGDataset(Dataset):
    def __init__(self, features, angle_data):
        self.features = features
        self.angle_data = angle_data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.angle_data[idx]


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_emg(data, fs=250):
    """
    EMG preprocessing:
      1. Remove 20-sample moving mean
      2. Take absolute value
      3. Low-pass IIR filter
    """
    win = 20
    b = np.ones(win) / win
    a = 1
    moving_mean = lfilter(b, a, data, axis=0)
    data_demeaned = data - moving_mean
    rectified = np.abs(data_demeaned)
    
    cutoff = 8.3  # Hz
    b_lp, a_lp = butter(N=2, Wn=cutoff/(fs/2), btype='low')
    smoothed = lfilter(b_lp, a_lp, rectified, axis=0)
    
    return smoothed


def preprocess_targets(angle_data, window_size=50, n_levels=128):
    """
    Preprocess targets with moving average smoothing and quantization
    Args:
        angle_data: Raw angle data (N, 1) or (N,)
        window_size: Size of moving average window
        n_levels: Number of quantization levels (default: 128)
    Returns:
        Smoothed and quantized angle data
    """
    angle_data_flat = angle_data.flatten()
    
    # Apply moving average
    b_ma = np.ones(window_size) / window_size
    a_ma = 1
    smoothed = lfilter(b_ma, a_ma, angle_data_flat)
    
    # Quantize to n_levels
    # 1. Normalize to [0, 1] range
    min_val = np.min(smoothed)
    max_val = np.max(smoothed)
    range_val = (max_val - min_val) + 1e-8
    normalized = (smoothed - min_val) / range_val
    
    # 2. Quantize to discrete levels (0 to n_levels-1)
    max_level = n_levels - 1
    quantized = np.round(np.clip(normalized * max_level, 0, max_level)).astype(np.float32)
    
    return quantized.reshape(-1, 1)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_windowed_statistics(emg_data, window_size=50):
    """
    Extract windowed statistics: mean, std, and last value for each channel.
    Returns features: 4 (means) + 4 (stds) + 4 (last values) = 12 features
    """
    N_samples, N_channels = emg_data.shape
    features = []
    
    for i in range(N_samples):
        start_idx = max(0, i - window_size + 1)
        window = emg_data[start_idx:i+1, :]
        
        means = np.mean(window, axis=0)
        stds = np.std(window, axis=0)
        last_vals = emg_data[i, :]
        
        sample_features = np.concatenate([means, stds, last_vals])
        features.append(sample_features)
    
    return np.array(features)


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_features(train_features, test_features, n_levels=256):
    """Normalize features to discrete levels"""
    # 1. Standardize using train set statistics
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0) + 1e-8
    
    train_std = (train_features - mean) / std
    test_std = (test_features - mean) / std
    
    # 2. Map to [0, 1] range based on train set min/max
    t_min = np.min(train_std, axis=0)
    t_max = np.max(train_std, axis=0)
    range_val = (t_max - t_min) + 1e-8
    
    train_01 = (train_std - t_min) / range_val
    test_01 = (test_std - t_min) / range_val
    
    # 3. Quantize to N levels (0 to N-1)
    max_level = n_levels - 1
    
    train_quant = np.round(np.clip(train_01 * max_level, 0, max_level)).astype(np.float32)
    test_quant = np.round(np.clip(test_01 * max_level, 0, max_level)).astype(np.float32)
    
    return train_quant, test_quant


def compute_rom_bounds(angle_data):
    """Compute per-finger range-of-motion bounds"""
    lower = np.min(angle_data, axis=0)
    upper = np.max(angle_data, axis=0)
    return lower, upper


# ============================================================================
# CSV SAVE FUNCTIONS
# ============================================================================

def save_preprocessed_data(features_train, labels_train, features_test, labels_test, finger_label):
    """Save preprocessed data to CSV files"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Combine features and labels
    train_data = pd.DataFrame(features_train)
    train_data.columns = [f"feature_{i}" for i in range(features_train.shape[1])]
    train_data['angle'] = labels_train.flatten()
    
    test_data = pd.DataFrame(features_test)
    test_data.columns = [f"feature_{i}" for i in range(features_test.shape[1])]
    test_data['angle'] = labels_test.flatten()
    
    # Save to CSV
    train_path = os.path.join(OUTPUT_DIR, f"train_{finger_label.lower()}_preprocessed.csv")
    test_path = os.path.join(OUTPUT_DIR, f"test_{finger_label.lower()}_preprocessed.csv")
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"\n✓ Training data saved to: {train_path}")
    print(f"  Shape: {train_data.shape}")
    print(f"  First row:\n{train_data.iloc[0]}")
    
    print(f"\n✓ Test data saved to: {test_path}")
    print(f"  Shape: {test_data.shape}")
    print(f"  First row:\n{test_data.iloc[0]}")
    
    return train_path, test_path


# ============================================================================
# SIMPLE NEURAL NETWORK MODEL
# ============================================================================

class SimpleEMGModel(nn.Module):
    """Simple feedforward model: 12 inputs -> 1 output"""
    def __init__(self):
        super(SimpleEMGModel, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output in [0, 1] range
        return x


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(model, train_loader, num_epochs=NUM_EPOCHS, learning_rate=1e-3):
    """Train model using simple MSE loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    losses = []
    model.train()
    
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            optimizer.zero_grad()
            
            feature_batch, angle_batch = batch
            feature_batch = torch.as_tensor(feature_batch, dtype=torch.float32)
            angle_batch = torch.as_tensor(angle_batch, dtype=torch.float32)
            
            outputs = model(feature_batch)
            loss = criterion(outputs, angle_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
    
    return losses


def evaluate_model(model, test_loader, test_angles, rom_bounds=None):
    """Evaluate model on test set"""
    model.eval()
    criterion = nn.MSELoss()
    raw_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            feature_batch, _ = batch
            feature_batch = torch.as_tensor(feature_batch, dtype=torch.float32)
            outputs = model(feature_batch)
            raw_predictions.append(outputs.cpu().numpy())

    evaluated_predictions = np.concatenate(raw_predictions, axis=0).flatten()
    test_angles_flat = test_angles.flatten()
    
    if rom_bounds is not None:
        evaluated_predictions = np.clip(evaluated_predictions, rom_bounds[0], rom_bounds[1])

    test_loss = np.mean((evaluated_predictions - test_angles_flat) ** 2)
    mae = np.mean(np.abs(evaluated_predictions - test_angles_flat))
    correlation = np.corrcoef(evaluated_predictions, test_angles_flat)[0, 1]

    return test_loss, mae, correlation, evaluated_predictions.reshape(-1, 1)


# ============================================================================
# CONSOLE INTERFACE
# ============================================================================

def select_finger():
    """Allow user to select which finger to train"""
    print("\n" + "="*70)
    print("EMG ANGLE PREDICTION - PREPROCESSED DATA TO CSV")
    print("="*70)
    print("\nSelect finger to train:")
    for i, finger in enumerate(ANGLE_COLUMNS):
        print(f"  {i+1}. {finger.replace('_ext_angle', '').capitalize()}")
    
    while True:
        try:
            choice = int(input("\nEnter finger number (1-5): "))
            if 1 <= choice <= 5:
                return choice - 1, ANGLE_COLUMNS[choice - 1]
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Select finger
    finger_idx, finger_name = select_finger()
    finger_label = finger_name.replace('_ext_angle', '').capitalize()
    
    print(f"\nTraining for: {finger_label}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("\n[Loading data...]")
    data_train1 = pd.read_csv('./merged_emg_angles_sample_indexed3.csv')
    data_train2 = pd.read_csv('./merged_emg_angles_sample_indexed2.csv')
    data_train3 = pd.read_csv('./merged_emg_angles_sample_indexed.csv')
    data_train4 = pd.read_csv('./SlowMovement1.csv')
    data_train = pd.concat([data_train1, data_train2, data_train3, data_train4], ignore_index=True)
    data_test = pd.read_csv('./merged_emg_angles_sample_indexed3.csv')
    
    data_train_emg = data_train[['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3']].values
    data_train_angles = data_train[[finger_name]].values / 180.0
    
    data_test_emg = data_test[['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3']].values
    data_test_angles = data_test[[finger_name]].values / 180.0
    
    rom_bounds = compute_rom_bounds(data_train_angles)
    
    # ========================================================================
    # PREPROCESS EMG INPUT
    # ========================================================================
    
    print("[Preprocessing EMG input data...]")
    train_emg_processed = preprocess_emg(preprocess_emg(data_train_emg))
    test_emg_processed = preprocess_emg(preprocess_emg(data_test_emg))
    
    # ========================================================================
    # PREPROCESS TARGETS (MOVING AVERAGE + QUANTIZATION)
    # ========================================================================
    
    print("[Preprocessing targets with moving average and quantization to 128 levels...]")
    ma_window = 50
    train_angles_smoothed = preprocess_targets(data_train_angles, window_size=ma_window, n_levels=128)
    test_angles_smoothed = preprocess_targets(data_test_angles, window_size=ma_window, n_levels=128)
    
    print(f"  Original train angles - Min: {data_train_angles.min():.4f}, Max: {data_train_angles.max():.4f}")
    print(f"  Quantized train angles - Min: {train_angles_smoothed.min():.0f}, Max: {train_angles_smoothed.max():.0f} (128 levels)")
    
    # ========================================================================
    # EXTRACT AND NORMALIZE FEATURES
    # ========================================================================
    
    print("[Extracting windowed statistics features...]")
    train_features = extract_windowed_statistics(train_emg_processed, window_size=WINDOW_SIZE)
    test_features = extract_windowed_statistics(test_emg_processed, window_size=WINDOW_SIZE)
    
    print(f"  Train features shape: {train_features.shape}")
    print(f"  Test features shape: {test_features.shape}")
    
    print("[Normalizing features...]")
    train_features_norm, test_features_norm = normalize_features(train_features, test_features)
    
    # ========================================================================
    # SAVE PREPROCESSED DATA TO CSV
    # ========================================================================
    
    print("\n[Saving preprocessed data to CSV...]")
    train_csv_path, test_csv_path = save_preprocessed_data(
        train_features_norm, 
        train_angles_smoothed, 
        test_features_norm, 
        test_angles_smoothed, 
        finger_label
    )
    
    # ========================================================================
    # CREATE DATASETS AND DATALOADERS
    # ========================================================================
    
    train_dataset = EMGDataset(train_features_norm, train_angles_smoothed)
    test_dataset = EMGDataset(test_features_norm, test_angles_smoothed)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # ========================================================================
    # CREATE AND TRAIN MODEL
    # ========================================================================
    
    print("\n[Creating model...]")
    model = SimpleEMGModel()
    
    print("[Training model...]")
    start_time = time.time()
    losses = train_model(model, train_loader, num_epochs=NUM_EPOCHS)
    train_time = time.time() - start_time
    
    # ========================================================================
    # EVALUATE MODEL
    # ========================================================================
    
    print("\n[Evaluating model...]")
    test_loss, mae, correlation, evaluated_predictions = evaluate_model(
        model, test_loader, test_angles_smoothed, rom_bounds=rom_bounds
    )
    
    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Finger: {finger_label}")
    print(f"✓ Test Loss (MSE): {test_loss:.6f}")
    print(f"✓ Mean Absolute Error: {mae:.6f}")
    print(f"✓ Correlation: {correlation:.4f}")
    print(f"✓ Training Time: {train_time:.2f}s")
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    
    model_path = f"emg_model_{finger_label.lower()}_ei.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved as '{model_path}'")
    
    # ========================================================================
    # PLOT RESULTS
    # ========================================================================
    
    print("\n[Generating plots...]")
    
    # Calculate errors
    true_vals = test_angles_smoothed.flatten()
    pred_vals = evaluated_predictions.flatten()
    errors = pred_vals - true_vals
    
    # Figure 1: True vs Predicted
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_axis = np.arange(len(true_vals))
    ax.plot(x_axis, true_vals, label='True (Quantized, 128 levels)', linewidth=2, alpha=0.7)
    ax.plot(x_axis, pred_vals, label='Predicted', linewidth=2, alpha=0.7)
    
    window_size_plot = 125
    if len(true_vals) >= window_size_plot:
        true_ma = np.convolve(true_vals, np.ones(window_size_plot)/window_size_plot, mode='valid')
        pred_ma = np.convolve(pred_vals, np.ones(window_size_plot)/window_size_plot, mode='valid')
        ax.plot(x_axis[window_size_plot-1:], true_ma, label='True (MA)', linewidth=2, color='blue', alpha=0.9)
        ax.plot(x_axis[window_size_plot-1:], pred_ma, label='Predicted (MA)', linewidth=2, color='orange', alpha=0.9)
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Angle Level (0-127)', fontsize=12)
    ax.set_title(f'True vs Predicted - {finger_label} (MA + 128-level Quantization)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_vals, pred_vals, alpha=0.5, color='steelblue', edgecolor='black')
    ax.plot([0, 127], [0, 127], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('True Angle Level (0-127)', fontsize=12)
    ax.set_ylabel('Predicted Angle Level (0-127)', fontsize=12)
    ax.set_title(f'True vs Predicted Scatter - {finger_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Error histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.4f}')
    
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Error Distribution - {finger_label}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Training loss
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(losses, marker='o', linewidth=2, markersize=6, color='steelblue')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
