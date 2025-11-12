import pandas as pd
import numpy as np

# --- 1. Configuration: Update these file paths ---

# Path to your hand tracking data CSV file
hand_data_path = r'C:\Users\VidaImre\OneDrive\University\hand\newelectrode_pos\sapiens-1105\test_set\test_set\test.csv'

# Path to your OpenBCI EMG data text file
emg_data_path = r'C:\Users\VidaImre\OneDrive\University\hand\newelectrode_pos\sapiens-1105\test_set\test_set\OpenBCI-RAW-2025-11-05_19-21-08.txt'

# Path for the output file
output_path = r'C:\Users\VidaImre\OneDrive\University\hand\newelectrode_pos\sapiens-1105\test_set\test_set\merged_emg_angles_sample_indexed.csv'


# --- 2. Load and Pre-process Hand Tracking Data (Angles) ---

print("Step 2: Loading and processing hand tracking data (angles only)...")

# Load the hand tracking data
hand_df = pd.read_csv(hand_data_path)

# Define the columns we want to keep
angle_columns = [
    'timestamp',
    'thumb_ext_angle',
    'index_ext_angle',
    'middle_ext_angle',
    'ring_ext_angle',
    'pinky_ext_angle',
    'hand_state'
]
hand_df = hand_df[angle_columns].copy()
hand_df.dropna(inplace=True)

# Convert timestamp from seconds to milliseconds (float) for accurate merging
hand_df['timestamp'] = hand_df['timestamp'] * 1000

print("Hand angle data processed successfully.")
print("-" * 50)


# --- 3. Load and Pre-process EMG Data ---

print("Step 3: Loading and processing EMG data...")

# Load the OpenBCI data file
emg_df = pd.read_csv(emg_data_path, comment='%', header=0)

# Clean up column names
emg_df.columns = emg_df.columns.str.strip()

# Keep only the timestamp and the EXG channels
emg_channels = [f'EXG Channel {i}' for i in range(8)]
columns_to_keep = ['Timestamp'] + emg_channels
emg_df = emg_df[columns_to_keep].copy()

# Rename and convert timestamp to float milliseconds
emg_df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
emg_df['timestamp'] = emg_df['timestamp'].astype(float) * 1000
emg_df.dropna(inplace=True)

print("EMG data processed successfully.")
print("-" * 50)


# --- 4. Synchronize and Merge DataFrames (Zero-Order Hold) ---

print("Step 4: Merging datasets using real timestamps...")

# Sort both DataFrames by the 'timestamp' column for merging
emg_df.sort_values('timestamp', inplace=True)
hand_df.sort_values('timestamp', inplace=True)

# Perform the merge using the real timestamps to ensure correct alignment
merged_df = pd.merge_asof(
    left=emg_df,
    right=hand_df,
    on='timestamp',
    direction='backward'
)

# Drop any initial rows that couldn't be merged
merged_df.dropna(inplace=True)

print("Merging complete.")
print("Shape of merged data (before timestamp replacement):", merged_df.shape)
print("-" * 50)


# --- 5. Replace Timestamp with a Sequential Sample Index ---

print("Step 5: Replacing real timestamps with a sequential sample index...")

# Generate a sequence of integers from 0 to (number of rows - 1)
# This replaces the original floating-point millisecond timestamps
merged_df['timestamp'] = np.arange(len(merged_df))

# Rename the column for clarity, as it's no longer a time value
merged_df.rename(columns={'timestamp': 'sample_index'}, inplace=True)

print("Timestamp column has been replaced with 'sample_index'.")
print("Final data head:\n", merged_df.head())
print("-" * 50)


# --- 6. Save the Final Merged File ---

print(f"Step 6: Saving the final indexed data to '{output_path}'...")

# Save the final dataframe to a CSV file
merged_df.to_csv(output_path, index=False)

print("Script finished successfully!")
print(f"The final data with a sequential sample index is saved at: {output_path}")