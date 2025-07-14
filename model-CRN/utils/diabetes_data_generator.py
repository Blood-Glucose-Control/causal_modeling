# Copyright (c) 2024, Diabetes CRN Adaptation
"""
Diabetes data generation for CRN training pipeline.

This module generates diabetes training data using the data-api simulator
and formats it for CRN model training.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the diabetes-data-api to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'diabetes-data-api'))
from main import DiabetesAnalyzer

def generate_diabetes_training_data(total_days=90, window_days=7, seed=42):
    """
    Generate diabetes training data from one patient's timeline using sliding windows.
    
    Args:
        total_days: Total days of patient data to generate
        window_days: Length of each training window in days
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with training, validation, and test splits in CRN format
    """
    
    print(f"Generating {total_days} days of diabetes data for 1 patient...")
    print(f"Creating sliding windows of {window_days} days each...")
    
    # Initialize analyzer
    analyzer = DiabetesAnalyzer(seed=seed)
    
    # Generate long patient timeline
    patient_data = analyzer.generate_patient_data(
        n_days=total_days, 
        start_date='2024-01-01'
    )
    
    print(f"✓ Generated {len(patient_data)} data points ({total_days} days)")
    
    # Create sliding windows
    windows = create_sliding_windows(patient_data, window_days)
    
    print(f"✓ Created {len(windows)} training windows")
    
    # Convert to CRN format
    crn_data = convert_windows_to_crn_format(windows)
    
    # Split into train/validation/test
    train_split, val_split, test_split = split_windows(crn_data, train_ratio=0.7, val_ratio=0.15)
    
    # Compute scaling parameters from training data
    scaling_params = compute_scaling_params(train_split)
    
    return {
        'training_data': train_split,
        'validation_data': val_split,
        'test_data': test_split,
        'test_data_seq': test_split,  # For decoder evaluation
        'scaling_data': scaling_params
    }


def create_sliding_windows(patient_data, window_days):
    """
    Create sliding windows from one patient's timeline.
    
    Args:
        patient_data: Single patient DataFrame from diabetes simulator
        window_days: Length of each window in days
        
    Returns:
        List of DataFrame windows
    """
    
    # Calculate window size in timesteps (5-minute intervals)
    window_size = window_days * 24 * 12  # days * hours * (60min/5min)
    
    # Create sliding windows with 1-day step
    step_size = 24 * 12  # 1 day step
    
    windows = []
    
    for start_idx in range(0, len(patient_data) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window = patient_data.iloc[start_idx:end_idx].copy()
        windows.append(window)
    
    return windows


def convert_windows_to_crn_format(windows_list):
    """
    Convert list of window DataFrames to CRN format.
    
    Args:
        windows_list: List of window DataFrames
        
    Returns:
        Dictionary in CRN format with arrays for all windows
    """
    
    num_windows = len(windows_list)
    
    if num_windows == 0:
        raise ValueError("No window data provided")
    
    # Get the length of each window (should be same for all)
    window_length = len(windows_list[0])
    
    # Initialize arrays
    glucose_data = np.zeros((num_windows, window_length))
    insulin_data = np.zeros((num_windows, window_length))
    carbs_data = np.zeros((num_windows, window_length))
    exercise_data = np.zeros((num_windows, window_length))
    stress_data = np.zeros((num_windows, window_length))
    active_insulin_data = np.zeros((num_windows, window_length))
    sequence_lengths = np.full(num_windows, window_length)
    
    # Fill arrays with window data
    for i, window_data in enumerate(windows_list):
        glucose_data[i, :] = window_data['glucose'].values
        insulin_data[i, :] = window_data['insulin'].values
        carbs_data[i, :] = window_data['carbs'].values
        exercise_data[i, :] = window_data['exercise'].values
        stress_data[i, :] = window_data['stress'].values
        active_insulin_data[i, :] = window_data['active_insulin'].values
    
    return {
        'glucose': glucose_data,
        'insulin_doses': insulin_data,
        'carbs': carbs_data,
        'exercise': exercise_data,
        'stress': stress_data,
        'active_insulin': active_insulin_data,
        'sequence_lengths': sequence_lengths
    }


def split_windows(crn_data, train_ratio=0.7, val_ratio=0.15):
    """
    Split window data into train/validation/test sets.
    
    Args:
        crn_data: CRN format data dictionary
        train_ratio: Fraction of windows for training
        val_ratio: Fraction of windows for validation
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    
    num_windows = crn_data['glucose'].shape[0]
    
    # Calculate split indices
    train_end = int(num_windows * train_ratio)
    val_end = int(num_windows * (train_ratio + val_ratio))
    
    def split_data(data_dict, start_idx, end_idx):
        """Helper to split a data dictionary."""
        split_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1:
                split_dict[key] = value[start_idx:end_idx]
            else:
                split_dict[key] = value
        return split_dict
    
    # Create splits
    train_data = split_data(crn_data, 0, train_end)
    val_data = split_data(crn_data, train_end, val_end)
    test_data = split_data(crn_data, val_end, num_windows)
    
    print(f"Data split: {train_end} train, {val_end - train_end} validation, {num_windows - val_end} test windows")
    
    return train_data, val_data, test_data


def compute_scaling_params(train_data):
    """
    Compute mean and std for data normalization.
    
    Args:
        train_data: Training data dictionary
        
    Returns:
        Tuple of (mean_series, std_series)
    """
    
    # Flatten patient data for statistics
    means = {}
    stds = {}
    
    for key in ['glucose', 'carbs', 'exercise', 'stress', 'active_insulin']:
        if key in train_data:
            data_flat = train_data[key].flatten()
            means[key] = float(np.mean(data_flat))
            stds[key] = float(np.std(data_flat))
    
    # Special handling for insulin doses (only non-zero values)
    if 'insulin_doses' in train_data:
        insulin_flat = train_data['insulin_doses'].flatten()
        nonzero_insulin = insulin_flat[insulin_flat > 0]
        if len(nonzero_insulin) > 0:
            means['insulin_doses'] = float(np.mean(nonzero_insulin))
            stds['insulin_doses'] = float(np.std(nonzero_insulin))
        else:
            means['insulin_doses'] = 0.0
            stds['insulin_doses'] = 1.0
    
    return pd.Series(means), pd.Series(stds)


def get_diabetes_sim_data(total_days=30, window_days=7, seed=42, b_load=False, b_save=False, model_root='results'):
    """
    Main function to generate or load diabetes simulation data.
    
    This replaces get_cancer_sim_data() for diabetes training.
    
    Args:
        total_days: Total days of patient timeline to generate
        window_days: Length of each training window in days
        seed: Random seed
        b_load: Whether to load existing data (not implemented yet)
        b_save: Whether to save generated data (not implemented yet)
        model_root: Directory for saving/loading
        
    Returns:
        Dictionary with training/validation/test data in CRN format
    """
    
    # For now, always generate fresh data
    # TODO: Implement save/load functionality if needed
    
    return generate_diabetes_training_data(
        total_days=total_days,
        window_days=window_days,
        seed=seed
    )


if __name__ == "__main__":
    # Test the diabetes data generation
    print("Testing diabetes data generation...")
    
    # Generate small dataset for testing
    data = get_diabetes_sim_data(total_days=14, window_days=7, seed=42)
    
    print("Generated data structure:")
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape'):
                    print(f"    {subkey}: {subvalue.shape}")
                else:
                    print(f"    {subkey}: {type(subvalue)}")
        else:
            print(f"  {key}: {type(value)}")
    
    print("✓ Diabetes data generation successful!")