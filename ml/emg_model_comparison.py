4# -*- coding: utf-8 -*-
"""
EMG Angle Prediction: Multi-Model Training and Comparison
Supports multiple feature extraction methods and allows model selection via console.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter
from itertools import combinations
import time
from tqdm import tqdm

# Global training configuration
NUM_EPOCHS = 15
WINDOW_SIZES_STATS = [10, 50, 100, 250]

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


class ConvEMGDataset(Dataset):
    def __init__(self, windows, extra_features, angle_data):
        self.windows = windows
        self.extra_features = extra_features
        self.angle_data = angle_data

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.extra_features[idx], self.angle_data[idx]


# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def preprocess_emg(data, fs=250):
    """
    EMG preprocessing:
      1. Remove 20-sample moving mean
      2. Take absolute value
      3. Low-pass IIR filter (approx 30-sample moving average)
    """
    # --- 1. Remove 20-sample moving mean ---
    win = 20
    b = np.ones(win) / win
    a = 1
    moving_mean = lfilter(b, a, data, axis=0)
    data_demeaned = data - moving_mean

    # --- 2. Rectify (absolute value) ---
    rectified = np.abs(data_demeaned)

    # --- 3. IIR low-pass filter ---
    cutoff = 8.3  # Hz
    b_lp, a_lp = butter(N=2, Wn=cutoff/(fs/2), btype='low')
    smoothed = lfilter(b_lp, a_lp, rectified, axis=0)

    return smoothed


# ============================================================================
# FEATURE EXTRACTION METHODS
# ============================================================================

def extract_raw_features(emg_data):
    """Raw EMG signals as features (4 channels)"""
    return emg_data


def extract_pairwise_ratios(emg_data, epsilon=1e-8):
    """Extract pairwise ratios from EMG channels (6 features)"""
    N_samples, N_channels = emg_data.shape
    channel_pairs = list(combinations(range(N_channels), 2))
    N_ratios = len(channel_pairs)
    
    ratios = np.zeros((N_samples, N_ratios))
    
    for ratio_idx, (ch_i, ch_j) in enumerate(channel_pairs):
        ratios[:, ratio_idx] = emg_data[:, ch_i] / (emg_data[:, ch_j] + epsilon)
    
    return np.clip(ratios, -10, 10)


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
        
        # Mean for each channel
        means = np.mean(window, axis=0)
        # Std for each channel
        stds = np.std(window, axis=0)
        # Last value for each channel
        last_vals = emg_data[i, :]
        
        # Concatenate all features
        sample_features = np.concatenate([means, stds, last_vals])
        features.append(sample_features)
    
    return np.array(features)


def extract_windowed_stats_with_pairwise_ratios(emg_data, window_size=50, epsilon=1e-8):
    """
    Extract windowed statistics (mean, std) + pairwise ratios of window last values.
    Returns features: 4 (means) + 4 (stds) + 6 (pairwise ratios) = 14 features
    """
    N_samples, N_channels = emg_data.shape
    channel_pairs = list(combinations(range(N_channels), 2))
    N_ratios = len(channel_pairs)
    features = []
    
    for i in range(N_samples):
        start_idx = max(0, i - window_size + 1)
        window = emg_data[start_idx:i+1, :]
        
        # Mean for each channel
        means = np.mean(window, axis=0)
        # Std for each channel
        stds = np.std(window, axis=0)
        
        # Last values from the window
        last_vals = emg_data[i, :]
        
        # Pairwise ratios of last values
        ratios = np.zeros(N_ratios)
        for ratio_idx, (ch_i, ch_j) in enumerate(channel_pairs):
            ratios[ratio_idx] = last_vals[ch_i] / (last_vals[ch_j] + epsilon)
        
        # Clip ratios to avoid extreme values
        ratios = np.clip(ratios, -10, 10)
        
        # Concatenate all features
        sample_features = np.concatenate([means, stds, ratios])
        features.append(sample_features)
    
    return np.array(features)


def build_sliding_windows(emg_data, window_size=50):
    """Build fixed-size causal windows with edge padding: (N, window_size, N_channels)."""
    N_samples, N_channels = emg_data.shape
    padded = np.pad(emg_data, ((window_size - 1, 0), (0, 0)), mode='edge')
    windows = np.zeros((N_samples, window_size, N_channels), dtype=np.float32)

    for i in range(N_samples):
        windows[i] = padded[i:i + window_size]

    return windows


def extract_window_conv_with_last_raw(emg_data, window_size=50):
    """Features for conv model: fixed windows + last raw values (4 features)."""
    windows = build_sliding_windows(emg_data, window_size=window_size)
    last_raw = emg_data.astype(np.float32)
    return windows, last_raw


def extract_window_conv_with_pairwise_ratios(emg_data, window_size=50, epsilon=1e-8):
    """Features for conv model: fixed windows + pairwise ratios of last raw values (6 features)."""
    windows = build_sliding_windows(emg_data, window_size=window_size)
    ratios = extract_pairwise_ratios(emg_data, epsilon=epsilon).astype(np.float32)
    return windows, ratios


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_features(train_features, test_features):
    """Normalize features using train set statistics"""
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0) + 1e-8
    
    train_norm = (train_features - mean) / std
    test_norm = (test_features - mean) / std
    
    return train_norm, test_norm


def normalize_windows(train_windows, test_windows):
    """Normalize windows per channel using train-set statistics."""
    mean = np.mean(train_windows, axis=(0, 1), keepdims=True)
    std = np.std(train_windows, axis=(0, 1), keepdims=True) + 1e-8
    train_norm = (train_windows - mean) / std
    test_norm = (test_windows - mean) / std
    return train_norm, test_norm


# ============================================================================
# NEURAL NETWORK MODELS
# ============================================================================

class RawEMGModel(nn.Module):
    """Model for raw EMG signals (4 inputs)"""
    def __init__(self):
        super(RawEMGModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 5)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class PairwiseRatioModel(nn.Module):
    """Model for pairwise ratios (6 inputs)"""
    def __init__(self):
        super(PairwiseRatioModel, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 5)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class WindowedStatisticsModel(nn.Module):
    """Model for windowed statistics (12 inputs: 4 means + 4 stds + 4 last values)"""
    def __init__(self):
        super(WindowedStatisticsModel, self).__init__()
        self.fc1 = nn.Linear(12, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 5)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class WindowedStatsWithRatiosModel(nn.Module):
    """Model for windowed statistics + pairwise ratios (14 inputs: 4 means + 4 stds + 6 ratios)"""
    def __init__(self):
        super(WindowedStatsWithRatiosModel, self).__init__()
        self.fc1 = nn.Linear(14, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 5)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ConvWithLastRawModel(nn.Module):
    """Conv model: Conv1d over channels + last raw values, then shared MLP head."""
    uses_conv_inputs = True

    def __init__(self):
        super(ConvWithLastRawModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=50)
        self.fc1 = nn.Linear(8 + 4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 5)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x_window, x_extra):
        x_window = x_window.permute(0, 2, 1)
        x_conv = torch.relu(self.conv(x_window)).squeeze(-1)
        x = torch.cat([x_conv, x_extra], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ConvWithPairwiseRatiosModel(nn.Module):
    """Conv model: Conv1d over channels + pairwise ratios, then shared MLP head."""
    uses_conv_inputs = True

    def __init__(self):
        super(ConvWithPairwiseRatiosModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=50)
        self.fc1 = nn.Linear(8 + 6, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 5)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x_window, x_extra):
        x_window = x_window.permute(0, 2, 1)
        x_conv = torch.relu(self.conv(x_window)).squeeze(-1)
        x = torch.cat([x_conv, x_extra], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def ccc_loss(x, y):
    """Concordance Correlation Coefficient Loss"""
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_var = torch.var(x)
    y_var = torch.var(y)
    x_std = torch.std(x)
    y_std = torch.std(y)

    rho = torch.mean((x - x_mean) * (y - y_mean)) / (x_std * y_std + 1e-8)
    ccc = (2 * rho * x_std * y_std) / (x_var + y_var + (x_mean - y_mean)**2 + 1e-8)

    return 1 - ccc


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(model, train_loader, num_epochs=NUM_EPOCHS, learning_rate=1e-3):
    """Train a model and return training losses"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
    
    losses = []
    model.train()
    
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            optimizer.zero_grad()

            if len(batch) == 3:
                window_batch, extra_batch, angle_batch = batch
                window_batch = torch.tensor(window_batch, dtype=torch.float32)
                extra_batch = torch.tensor(extra_batch, dtype=torch.float32)
                angle_batch = torch.tensor(angle_batch, dtype=torch.float32)
                outputs = model(window_batch, extra_batch)
            else:
                feature_batch, angle_batch = batch
                feature_batch = torch.tensor(feature_batch, dtype=torch.float32)
                angle_batch = torch.tensor(angle_batch, dtype=torch.float32)
                outputs = model(feature_batch)

            loss = ccc_loss(outputs, angle_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
    
    return losses


def evaluate_model(model, test_loader, test_angles):
    """Evaluate model on test set"""
    criterion = nn.MSELoss()
    model.eval()
    
    total_loss = 0.0
    angle_accuracies = np.zeros(5)
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            if len(batch) == 3:
                window_batch, extra_batch, angle_batch = batch
                window_batch = torch.tensor(window_batch, dtype=torch.float32)
                extra_batch = torch.tensor(extra_batch, dtype=torch.float32)
                angle_batch = torch.tensor(angle_batch, dtype=torch.float32)
                outputs = model(window_batch, extra_batch)
            else:
                feature_batch, angle_batch = batch
                feature_batch = torch.tensor(feature_batch, dtype=torch.float32)
                angle_batch = torch.tensor(angle_batch, dtype=torch.float32)
                outputs = model(feature_batch)

            loss = criterion(outputs, angle_batch)
            total_loss += loss.item()

            angle_batch_np = angle_batch.numpy()
            outputs_np = outputs.numpy()
            angle_accuracies += (1 - np.mean(np.abs(outputs_np - angle_batch_np), axis=0)) * 100
            num_batches += 1

    test_loss = total_loss / len(test_loader)
    angle_accuracies /= num_batches
    mean_accuracy = np.mean(angle_accuracies)
    
    return test_loss, angle_accuracies, mean_accuracy


# ============================================================================
# CONSOLE INTERFACE
# ============================================================================

def select_models():
    """Allow user to select which models to train"""
    print("\n" + "="*70)
    print("EMG ANGLE PREDICTION - MULTI-MODEL COMPARISON")
    print("="*70)
    print("\nAvailable Models:")
    print("  1. Raw EMG Signals (4 features)")
    print("  2. Pairwise Ratios (6 features)")
    print("  3. Windowed Statistics (trains w=10,50,100,250)")
    print("  4. Windowed Stats + Pairwise Ratios (trains w=10,50,100,250)")
    print("  5. Conv(2 filters, k=50) + Last Raw Values")
    print("  6. Conv(2 filters, k=50) + Pairwise Ratios")
    print("  7. Train All Models")
    print("\nEnter model numbers separated by commas (e.g., 1,5 or 7):")
    
    while True:
        user_input = input("Selection: ").strip()
        
        if user_input == "7":
            return [1, 2, 3, 4, 5, 6]
        
        try:
            selected = [int(x.strip()) for x in user_input.split(",")]
            if all(x in [1, 2, 3, 4, 5, 6] for x in selected):
                # Keep order while removing duplicates
                return list(dict.fromkeys(selected))
            else:
                print("Invalid selection. Please enter numbers 1-6 or 7.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Get model selection from user
    selected_models = select_models()

    base_configs = {
        1: {
            'run_id': '1',
            'name': 'Raw EMG Signals',
            'model_class': RawEMGModel,
            'feature_extractor': extract_raw_features
        },
        2: {
            'run_id': '2',
            'name': 'Pairwise Ratios',
            'model_class': PairwiseRatioModel,
            'feature_extractor': extract_pairwise_ratios
        },
        5: {
            'run_id': '5',
            'name': 'Conv + Last Raw Values',
            'model_class': ConvWithLastRawModel,
            'feature_extractor': extract_window_conv_with_last_raw
        },
        6: {
            'run_id': '6',
            'name': 'Conv + Pairwise Ratios',
            'model_class': ConvWithPairwiseRatiosModel,
            'feature_extractor': extract_window_conv_with_pairwise_ratios
        }
    }

    run_configs = []
    for model_id in selected_models:
        if model_id == 3:
            for ws in WINDOW_SIZES_STATS:
                run_configs.append({
                    'run_id': f'3_w{ws}',
                    'name': f'Windowed Statistics (w={ws})',
                    'model_class': WindowedStatisticsModel,
                    'feature_extractor': (lambda data, ws=ws: extract_windowed_statistics(data, window_size=ws))
                })
        elif model_id == 4:
            for ws in WINDOW_SIZES_STATS:
                run_configs.append({
                    'run_id': f'4_w{ws}',
                    'name': f'Windowed Stats + Pairwise Ratios (w={ws})',
                    'model_class': WindowedStatsWithRatiosModel,
                    'feature_extractor': (lambda data, ws=ws: extract_windowed_stats_with_pairwise_ratios(data, window_size=ws))
                })
        else:
            run_configs.append(base_configs[model_id])
    
    # ========================================================================
    # LOAD AND PREPROCESS DATA
    # ========================================================================
    
    print("\n[Loading data...]")
    data_train = pd.read_csv('./merged_emg_angles_sample_indexed3.csv')
    data_test = pd.read_csv('./merged_emg_angles_sample_indexed2.csv')
    
    data_train_emg = data_train[['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3']].values
    data_train_angles = data_train[['thumb_ext_angle', 'index_ext_angle', 'middle_ext_angle', 'ring_ext_angle', 'pinky_ext_angle']].values / 180.0
    
    data_test_emg = data_test[['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3']].values
    data_test_angles = data_test[['thumb_ext_angle', 'index_ext_angle', 'middle_ext_angle', 'ring_ext_angle', 'pinky_ext_angle']].values / 180.0
    
    print("[Preprocessing EMG data...]")
    train_emg_processed = preprocess_emg(preprocess_emg(data_train_emg)) #/ 200
    test_emg_processed = preprocess_emg(preprocess_emg(data_test_emg)) #/ 200
    
    # ========================================================================
    # TRAIN AND EVALUATE MODELS
    # ========================================================================
    
    results = {}
    angles = ['thumb_ext_angle', 'index_ext_angle', 'middle_ext_angle', 'ring_ext_angle', 'pinky_ext_angle']
    
    for config in tqdm(run_configs, desc="Processing Models"):
        print("\n" + "-"*70)
        print(f"Training: {config['name']}")
        print("-"*70)
        
        # Extract features
        print("[Extracting features...]")
        train_features_norm = None
        test_features_norm = None
        uses_conv_inputs = getattr(config['model_class'], 'uses_conv_inputs', False)

        if uses_conv_inputs:
            train_windows, train_extra = config['feature_extractor'](train_emg_processed)
            test_windows, test_extra = config['feature_extractor'](test_emg_processed)

            print(f"  Train windows shape: {train_windows.shape}")
            print(f"  Test windows shape: {test_windows.shape}")
            print(f"  Train extra shape: {train_extra.shape}")
            print(f"  Test extra shape: {test_extra.shape}")

            print("[Normalizing windows and extra features...]")
            train_windows_norm, test_windows_norm = normalize_windows(train_windows, test_windows)
            train_extra_norm, test_extra_norm = normalize_features(train_extra, test_extra)

            train_dataset = ConvEMGDataset(train_windows_norm, train_extra_norm, data_train_angles)
            test_dataset = ConvEMGDataset(test_windows_norm, test_extra_norm, data_test_angles)
        else:
            train_features = config['feature_extractor'](train_emg_processed)
            test_features = config['feature_extractor'](test_emg_processed)

            print(f"  Train features shape: {train_features.shape}")
            print(f"  Test features shape: {test_features.shape}")

            # Normalize features
            print("[Normalizing features...]")
            train_features_norm, test_features_norm = normalize_features(train_features, test_features)

            # Create datasets and dataloaders
            train_dataset = EMGDataset(train_features_norm, data_train_angles)
            test_dataset = EMGDataset(test_features_norm, data_test_angles)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Create and train model
        print("[Training model...]")
        model = config['model_class']()
        start_time = time.time()
        losses = train_model(model, train_loader, num_epochs=NUM_EPOCHS)
        train_time = time.time() - start_time
        
        # Evaluate model
        print("[Evaluating model...]")
        test_loss, angle_accuracies, mean_accuracy = evaluate_model(model, test_loader, data_test_angles)
        
        # Store results
        results[config['run_id']] = {
            'name': config['name'],
            'model': model,
            'losses': losses,
            'test_loss': test_loss,
            'angle_accuracies': angle_accuracies,
            'mean_accuracy': mean_accuracy,
            'train_time': train_time,
            'train_features': train_features_norm,
            'test_features': test_features_norm
        }
        
        # Print results
        print(f"\n✓ Test Loss (MSE): {test_loss:.4f}")
        print(f"✓ Mean Accuracy: {mean_accuracy:.2f}%")
        print(f"✓ Training Time: {train_time:.2f}s")
        print("\nAccuracy per finger:")
        for i, finger in enumerate(angles):
            print(f"  {finger}: {angle_accuracies[i]:.2f}%")
        
        # Save model
        model_path = f"ccc_model_{config['run_id']}_{config['name'].replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '').replace(',', '').replace('=', '').lower()}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"\n✓ Model saved as '{model_path}'")

        # ------------------------------------------------------------------
        # Generate and store predictions for the test set (for plotting)
        # ------------------------------------------------------------------
        print("[Generating predictions on test set...]")
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    window_batch, extra_batch, angle_batch = batch
                    window_batch_t = torch.tensor(window_batch, dtype=torch.float32)
                    extra_batch_t = torch.tensor(extra_batch, dtype=torch.float32)
                    outputs = model(window_batch_t, extra_batch_t).cpu().numpy()
                else:
                    feature_batch, angle_batch = batch
                    feature_batch_t = torch.tensor(feature_batch, dtype=torch.float32)
                    outputs = model(feature_batch_t).cpu().numpy()
                all_preds.append(outputs)

                # angle_batch may be numpy already; ensure numpy array
                if isinstance(angle_batch, np.ndarray):
                    all_trues.append(angle_batch)
                else:
                    try:
                        all_trues.append(np.array(angle_batch))
                    except Exception:
                        all_trues.append(angle_batch)

        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)

        results[config['run_id']]['preds'] = all_preds
        results[config['run_id']]['trues'] = all_trues

        # ------------------------------------------------------------------
        # Plot true vs predicted overlays (one subplot per finger)
        # ------------------------------------------------------------------
        print("[Plotting true vs predicted overlays...]")
        fig, axs = plt.subplots(len(angles), 1, figsize=(12, 2.5 * len(angles)), sharex=True)
        if len(angles) == 1:
            axs = [axs]

        for idx, finger in enumerate(angles):
            y_true = all_trues[:, idx]
            y_pred = all_preds[:, idx]

            # Smooth predictions for visual clarity
            window = min(250, max(3, len(y_pred)//10))
            y_pred_ma = pd.Series(y_pred).rolling(window, center=True, min_periods=1).mean().values

            axs[idx].plot(y_true, label='True', alpha=0.7)
            axs[idx].plot(y_pred_ma, label='Predicted (MA)', alpha=0.9)
            axs[idx].set_ylabel(finger.replace('_ext_angle', ''))
            axs[idx].legend(loc='upper right')

        axs[-1].set_xlabel('Sample Index')
        plt.suptitle(f"Predicted vs True - {config['name']}")
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # COMPARISON AND VISUALIZATION
    # ========================================================================
    
    if len(results) > 1:
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        # Comparison table
        print("\n{:<30} {:<12} {:<15} {:<12}".format("Model", "Test Loss", "Mean Accuracy", "Train Time"))
        print("-"*70)
        for config in run_configs:
            r = results[config['run_id']]
            print("{:<30} {:<12.4f} {:<15.2f}% {:<12.2f}s".format(
                r['name'], r['test_loss'], r['mean_accuracy'], r['train_time']
            ))
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Learning curves
        ax = axes[0]
        for config in run_configs:
            ax.plot(results[config['run_id']]['losses'], marker='o', label=results[config['run_id']]['name'], alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy comparison
        ax = axes[1]
        accuracies = [results[cfg['run_id']]['angle_accuracies'] for cfg in run_configs]
        
        x = np.arange(len(angles))
        width = min(0.8 / max(1, len(run_configs)), 0.25)
        
        for i, config in enumerate(run_configs):
            offset = (i - len(run_configs)/2 + 0.5) * width
            ax.bar(x + offset, accuracies[i], width, label=results[config['run_id']]['name'], alpha=0.8)
        
        ax.set_xlabel('Finger')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Per-Finger Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('_ext_angle', '') for f in angles], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
