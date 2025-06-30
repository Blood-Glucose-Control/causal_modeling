"""
Glucose data adapter for CRN model.

Adapts the synthetic glucose dataset (ml_dataset.csv) to the CRN model's expected format.
The CRN expects data in the same format as the cancer simulation, with specific arrays
for covariates, treatments, and outputs.
"""

import numpy as np
import pandas as pd
import logging


def load_glucose_data(data_path):
    """
    Load glucose data from CSV file.
    
    Args:
        data_path (str): Path to the ml_dataset.csv file
        
    Returns:
        pandas.DataFrame: Loaded glucose data
    """
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def create_sequences(df, sequence_length=20, prediction_horizon=5):
    """
    Create sequences for CRN training from glucose data.
    
    Args:
        df (DataFrame): Glucose data
        sequence_length (int): Length of input sequences
        prediction_horizon (int): How far ahead to predict
        
    Returns:
        dict: Data formatted for CRN model
    """
    
    # Define covariates (features that influence glucose but aren't treatments)
    covariate_cols = ['hour', 'day_of_week', 'carbs', 'exercise', 'stress', 
                     'active_insulin', 'carb_impact', 'meal_insulin_delay', 
                     'is_weekend', 'time_since_last_meal', 'time_since_last_insulin']
    
    # Treatment is insulin dose
    treatment_cols = ['insulin']
    
    # Output is glucose level
    output_cols = ['glucose']
    
    sequences = []
    
    # Create overlapping sequences
    for i in range(len(df) - sequence_length - prediction_horizon + 1):
        seq_data = {}
        
        # Extract sequence data
        seq_df = df.iloc[i:i + sequence_length + prediction_horizon]
        
        # Current covariates (including glucose history for context)
        current_covariates = seq_df[covariate_cols + ['glucose']].iloc[:sequence_length].values
        
        # Previous treatments (shifted by 1 timestep)
        prev_treatments = np.zeros((sequence_length, len(treatment_cols)))
        if sequence_length > 1:
            prev_treatments[1:] = seq_df[treatment_cols].iloc[:sequence_length-1].values
        
        # Current treatments
        current_treatments = seq_df[treatment_cols].iloc[:sequence_length].values
        
        # Outputs (glucose levels) - predict next timestep
        outputs = seq_df[output_cols].iloc[1:sequence_length+1].values  # Shifted by 1 for prediction
        
        # Ensure output shape matches input length
        if outputs.shape[0] != sequence_length:
            outputs = seq_df[output_cols].iloc[:sequence_length].values
        
        # Future outputs for testing counterfactuals
        future_outputs = seq_df[output_cols].iloc[sequence_length:sequence_length + prediction_horizon].values
        
        # Active entries (all 1s since we have complete data)
        active_entries = np.ones((sequence_length, len(output_cols)))
        
        seq_data['current_covariates'] = current_covariates
        seq_data['previous_treatments'] = prev_treatments
        seq_data['current_treatments'] = current_treatments
        seq_data['outputs'] = outputs
        seq_data['future_outputs'] = future_outputs
        seq_data['active_entries'] = active_entries
        
        sequences.append(seq_data)
    
    return sequences


def split_data(sequences, train_ratio=0.7, val_ratio=0.15):
    """
    Split sequences into train/validation/test sets.
    
    Args:
        sequences (list): List of sequence dictionaries
        train_ratio (float): Proportion for training
        val_ratio (float): Proportion for validation
        
    Returns:
        tuple: (train_sequences, val_sequences, test_sequences)
    """
    n_total = len(sequences)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train:n_train + n_val]
    test_sequences = sequences[n_train + n_val:]
    
    return train_sequences, val_sequences, test_sequences


def sequences_to_arrays(sequences):
    """
    Convert list of sequence dictionaries to numpy arrays for CRN model.
    
    Args:
        sequences (list): List of sequence dictionaries
        
    Returns:
        dict: Arrays formatted for CRN model
    """
    if not sequences:
        return {}
    
    # Stack all sequences
    current_covariates = np.stack([seq['current_covariates'] for seq in sequences])
    previous_treatments = np.stack([seq['previous_treatments'] for seq in sequences])
    current_treatments = np.stack([seq['current_treatments'] for seq in sequences])
    outputs = np.stack([seq['outputs'] for seq in sequences])
    active_entries = np.stack([seq['active_entries'] for seq in sequences])
    
    return {
        'current_covariates': current_covariates,
        'previous_treatments': previous_treatments,
        'current_treatments': current_treatments,
        'outputs': outputs,
        'active_entries': active_entries
    }


def get_scaling_params(training_data):
    """
    Calculate scaling parameters for normalization.
    
    Args:
        training_data (dict): Training data arrays
        
    Returns:
        tuple: (means_series, stds_series)
    """
    means = {}
    stds = {}
    
    # Scale glucose (outputs)
    glucose_values = training_data['outputs'].flatten()
    means['glucose'] = np.mean(glucose_values)
    stds['glucose'] = np.std(glucose_values)
    
    # Scale insulin (treatments)
    insulin_values = training_data['current_treatments'].flatten()
    means['insulin'] = np.mean(insulin_values)
    stds['insulin'] = np.std(insulin_values)
    
    # Scale covariates
    for i, covariate_name in enumerate(['hour', 'day_of_week', 'carbs', 'exercise', 'stress', 
                                       'active_insulin', 'carb_impact', 'meal_insulin_delay', 
                                       'is_weekend', 'time_since_last_meal', 'time_since_last_insulin', 'glucose_history']):
        covariate_values = training_data['current_covariates'][:, :, i].flatten()
        means[covariate_name] = np.mean(covariate_values)
        stds[covariate_name] = np.std(covariate_values)
    
    return pd.Series(means), pd.Series(stds)


def get_glucose_sim_data(data_path, sequence_length=20, prediction_horizon=5):
    """
    Main function to load and process glucose data for CRN model.
    
    Args:
        data_path (str): Path to ml_dataset.csv
        sequence_length (int): Length of input sequences
        prediction_horizon (int): Prediction horizon for decoder
        
    Returns:
        dict: Processed data in CRN format
    """
    logging.info(f"Loading glucose data from {data_path}")
    
    # Load data
    df = load_glucose_data(data_path)
    logging.info(f"Loaded {len(df)} time points")
    
    # Create sequences
    sequences = create_sequences(df, sequence_length, prediction_horizon)
    logging.info(f"Created {len(sequences)} sequences")
    
    # Split data
    train_seq, val_seq, test_seq = split_data(sequences)
    logging.info(f"Split: {len(train_seq)} train, {len(val_seq)} val, {len(test_seq)} test")
    
    # Convert to arrays
    training_data = sequences_to_arrays(train_seq)
    validation_data = sequences_to_arrays(val_seq)
    test_data = sequences_to_arrays(test_seq)
    
    # Add future outputs to test data for counterfactual evaluation
    if test_seq:
        test_data['future_outputs'] = np.stack([seq['future_outputs'] for seq in test_seq])
    
    # Calculate scaling parameters
    scaling_data = get_scaling_params(training_data)
    
    # Calculate data dimensions
    num_covariates = training_data['current_covariates'].shape[2]
    num_treatments = training_data['current_treatments'].shape[2]
    num_outputs = training_data['outputs'].shape[2]
    
    logging.info(f"Data dimensions: {num_covariates} covariates, {num_treatments} treatments, {num_outputs} outputs")
    
    return {
        'training_data': training_data,
        'validation_data': validation_data,
        'test_data': test_data,
        'scaling_data': scaling_data,
        'num_covariates': num_covariates,
        'num_treatments': num_treatments,
        'num_outputs': num_outputs,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon
    }