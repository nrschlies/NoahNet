# Import necessary libraries
import os
import numpy as np
import pandas as pd
import gc

# Machine learning libraries
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Step 1/10: Setting up paths and loading metadata...")

# Paths to data directories
train_dir = '/kaggle/input/ariel-data-challenge-2024/train/'
test_dir = '/kaggle/input/ariel-data-challenge-2024/test/'

# Load metadata files
train_adc_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2024/train_adc_info.csv')
train_labels = pd.read_csv('/kaggle/input/ariel-data-challenge-2024/train_labels.csv')

# Number of planets in the training set
train_planet_ids = train_adc_info['planet_id'].unique()

print("Completed Step 1/10.")

# Define helper functions
def restore_dynamic_range(signal_data, gain, offset):
    restored_signal = signal_data * gain + offset
    return restored_signal

def apply_calibration(signal_data, calibration_data):
    # Flatten calibration frames for dark, dead, and flat
    dark = calibration_data['dark'].flatten()
    dead = calibration_data['dead'].flatten()
    flat = calibration_data['flat'].flatten()

    # Check if shapes match for dark, dead, and flat
    if signal_data.shape[1] != dark.shape[0]:
        raise ValueError(f"Mismatch in signal and calibration data shapes: {signal_data.shape[1]} vs {dark.shape[0]}")
    
    # Subtract dark current
    signal_data -= dark
    # Correct for dead/hot pixels
    signal_data[:, dead == 1] = np.nan
    # Correct for flat field
    signal_data /= flat
    
    # Apply linearity correction to the first 192 pixels
    # linear_corr shape: (192, n_coeffs)
    linear_corr = calibration_data['linear_corr']
    
    # Apply the linear_corr only to the first 192 pixels
    n_pixels_corr = linear_corr.shape[0]  # 192 pixels
    n_pixels_signal = signal_data.shape[1]  # 1024 pixels
    
    if n_pixels_signal < n_pixels_corr:
        raise ValueError(f"Signal data has fewer pixels ({n_pixels_signal}) than required by linear_corr ({n_pixels_corr}).")
    
    # Apply the linear correction only to the first 192 pixels of the signal
    from numpy.polynomial.polynomial import polyval
    signal_data_T = signal_data[:, :192].T  # Apply linear correction only to the first 192 pixels
    corrected_signal_data_T = polyval(signal_data_T, linear_corr.T)
    signal_data[:, :192] = corrected_signal_data_T.T  # Update signal data
    
    return signal_data

def load_calibration_data(calib_dir):
    calibration_data = {}
    for calib_file in ['dark', 'dead', 'flat', 'linear_corr']:
        calib_path = os.path.join(calib_dir, f'{calib_file}.parquet')
        calib_df = pd.read_parquet(calib_path)
        calibration_data[calib_file] = calib_df.values
    return calibration_data

print("Step 2/10: Processing data for a single planet...")

# Select a planet ID
planet_id = train_planet_ids[0]
print(f"Processing Planet ID: {planet_id}")

# Load signal data
fgs1_signal_path = os.path.join(train_dir, str(planet_id), 'FGS1_signal.parquet')
fgs1_signal_df = pd.read_parquet(fgs1_signal_path)
print("Loaded FGS1 signal data.")

airs_ch0_signal_path = os.path.join(train_dir, str(planet_id), 'AIRS-CH0_signal.parquet')
airs_ch0_signal_df = pd.read_parquet(airs_ch0_signal_path)
print("Loaded AIRS-CH0 signal data.")

# Verify the shapes
print("FGS1 signal data shape:", fgs1_signal_df.values.shape)
print("AIRS-CH0 signal data shape:", airs_ch0_signal_df.values.shape)

# Adjust signal data if necessary
# For FGS1
expected_fgs1_pixels = 32 * 32  # 1024 pixels
if fgs1_signal_df.shape[1] > expected_fgs1_pixels:
    # Assuming extra columns, take the last expected_fgs1_pixels columns
    fgs1_signal_values = fgs1_signal_df.iloc[:, -expected_fgs1_pixels:].values
else:
    fgs1_signal_values = fgs1_signal_df.values

# For AIRS-CH0
expected_airs_pixels = 32 * 356  # 11392 pixels
if airs_ch0_signal_df.shape[1] > expected_airs_pixels:
    # Assuming extra columns, take the last expected_airs_pixels columns
    airs_signal_values = airs_ch0_signal_df.iloc[:, -expected_airs_pixels:].values
else:
    airs_signal_values = airs_ch0_signal_df.values

print("Adjusted signal data shapes:")
print("FGS1 signal data shape after adjustment:", fgs1_signal_values.shape)
print("AIRS-CH0 signal data shape after adjustment:", airs_signal_values.shape)

print("Completed Step 2/10.")

print("Step 3/10: Restoring dynamic range...")

# Get ADC info for the planet
adc_info_planet = train_adc_info[train_adc_info['planet_id'] == planet_id].iloc[0]

# Restore dynamic range for FGS1
fgs1_gain = adc_info_planet['FGS1_adc_gain']
fgs1_offset = adc_info_planet['FGS1_adc_offset']
fgs1_signal = restore_dynamic_range(fgs1_signal_values, fgs1_gain, fgs1_offset)
del fgs1_signal_df
gc.collect()
print("Restored dynamic range for FGS1.")

# Restore dynamic range for AIRS-CH0
airs_gain = adc_info_planet['AIRS-CH0_adc_gain']
airs_offset = adc_info_planet['AIRS-CH0_adc_offset']
airs_signal = restore_dynamic_range(airs_signal_values, airs_gain, airs_offset)
del airs_ch0_signal_df
gc.collect()
print("Restored dynamic range for AIRS-CH0.")

print("Completed Step 3/10.")

print("Step 4/10: Applying calibration corrections...")

# Load calibration data for FGS1
fgs1_calib_dir = os.path.join(train_dir, str(planet_id), 'FGS1_calibration')
fgs1_calib = load_calibration_data(fgs1_calib_dir)

# Ensure calibration data shapes match
print("FGS1 calibration data shapes:")
for key, value in fgs1_calib.items():
    print(f"{key}: {value.shape}")

# Check if the linear_corr shape is 192
if fgs1_calib['linear_corr'].shape[0] == 192:
    print("Linear correction applies only to the first 192 pixels.")
else:
    print(f"Warning: Unexpected linear_corr shape: {fgs1_calib['linear_corr'].shape}")

# Apply calibration to FGS1 signal
fgs1_signal_calibrated = apply_calibration(fgs1_signal, fgs1_calib)
print("Applied calibration to FGS1 signal.")
del fgs1_signal  # Free memory
gc.collect()

# Load calibration data for AIRS-CH0
airs_calib_dir = os.path.join(train_dir, str(planet_id), 'AIRS-CH0_calibration')
airs_calib = load_calibration_data(airs_calib_dir)

# Ensure calibration data shapes match
print("AIRS-CH0 calibration data shapes:")
for key, value in airs_calib.items():
    print(f"{key}: {value.shape}")

# Apply calibration to AIRS-CH0 signal
airs_signal_calibrated = apply_calibration(airs_signal, airs_calib)
print("Applied calibration to AIRS-CH0 signal.")
del airs_signal  # Free memory
gc.collect()

print("Completed Step 4/10.")

### Memory management enhancement:
# After each major step, clean up unnecessary objects and invoke garbage collection

# Handle missing values
print("Step 5/10: Handling missing values...")

# Handle missing values
fgs1_signal_calibrated = np.nan_to_num(fgs1_signal_calibrated, nan=np.nanmean(fgs1_signal_calibrated))
airs_signal_calibrated = np.nan_to_num(airs_signal_calibrated, nan=np.nanmean(airs_signal_calibrated))
print("Replaced NaNs with column means.")
gc.collect()

print("Completed Step 5/10.")

print("Step 6/10: Extracting features...")

# Extract features
fgs1_features = np.nanmean(fgs1_signal_calibrated, axis=1)
airs_features = np.nanmean(airs_signal_calibrated, axis=1)
print("Extracted features from FGS1 and AIRS-CH0.")

# Truncate to same length
num_samples = min(len(fgs1_features), len(airs_features))
fgs1_features = fgs1_features[:num_samples]
airs_features = airs_features[:num_samples]

# Combine features
combined_features = np.vstack((fgs1_features, airs_features)).T
print("Combined features.")
gc.collect()

print("Completed Step 6/10.")