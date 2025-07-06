"""
Glucose-specific evaluation utilities for CRN model.
Adapted from evaluation_utils.py for glucose data format.
"""

import numpy as np
import pandas as pd
from CRN_model import CRN_Model
import pickle


def write_results_to_file(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=2)


def append_results_to_file(filename, data):
    with open(filename, 'a+b') as handle:
        pickle.dump(data, handle, protocol=2)


def load_trained_model(dataset_test, hyperparams_file, model_name, model_folder, b_decoder_model=False, ordinal_treatments=False):
    _, length, num_covariates = dataset_test['current_covariates'].shape
    num_treatments = dataset_test['current_treatments'].shape[-1]
    num_outputs = dataset_test['outputs'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    print("Loading best hyperparameters for model")
    with open(hyperparams_file, 'rb') as handle:
        best_hyperparams = pickle.load(handle)

    model = CRN_Model(params, best_hyperparams)
    if (b_decoder_model):
        model = CRN_Model(params, best_hyperparams, b_train_decoder=True)

    model.load_model(model_name=model_name, model_folder=model_folder, ordinal_treatments=ordinal_treatments)
    return model


def get_processed_data(raw_sim_data, scaling_params, ordinal_treatments=False):
    """
    Create formatted data to train both encoder and seq2seq architecture for glucose data.
    Follows the exact same pattern as cancer evaluation_utils.py
    
    Args:
        raw_sim_data: Raw simulation data
        scaling_params: Scaling parameters (mean, std)
        ordinal_treatments: If True, use ordinal (continuous) treatment encoding instead of one-hot
    """
    mean, std = scaling_params

    horizon = 1
    offset = 1

    # Extract the original sequence length from the data
    num_patients, original_length, _ = raw_sim_data['current_covariates'].shape
    sequence_lengths = raw_sim_data.get('sequence_lengths', np.full(num_patients, original_length))

    # Normalize glucose (like cancer_volume normalization)
    glucose_normalized = (raw_sim_data['outputs'][:, :, 0] - mean['glucose']) / std['glucose']
    
    # Create patient types (glucose history mean as "patient type")
    glucose_history_mean = np.mean(raw_sim_data['current_covariates'][:, :, -1], axis=1)  # glucose_history column
    patient_types = (glucose_history_mean - mean['glucose_history']) / std['glucose_history']
    
    # Expand patient_types to time dimension (like cancer code does)
    patient_types_expanded = np.stack([patient_types for t in range(original_length - offset)], axis=1)
    
    # Current covariates: glucose + patient_type (following cancer pattern)
    current_covariates = np.concatenate([
        glucose_normalized[:, :-offset, np.newaxis],  # glucose values (like cancer_volume)
        patient_types_expanded[:, :, np.newaxis]      # patient types
    ], axis=-1)
    
    # Outputs: next timestep glucose (following cancer pattern)
    outputs = glucose_normalized[:, horizon:, np.newaxis]
    
    # Treatments: Handle ordinal vs one-hot encoding
    insulin_dosage = raw_sim_data['current_treatments'][:, :, 0]
    insulin_application = (insulin_dosage > 0).astype(float)
    
    # Normalize insulin dosage
    insulin_dosage_norm = (insulin_dosage - mean['insulin']) / std['insulin']
    
    if ordinal_treatments:
        # Ordinal encoding: Use normalized dosage values directly
        current_treatments = insulin_dosage_norm[:, :original_length - offset, np.newaxis]
        previous_treatments = insulin_dosage_norm[:, :original_length - offset - 1, np.newaxis]
        
        # Set up input/output scaling for ordinal treatments
        input_means = np.array([mean['glucose'], mean['glucose_history'], mean['insulin']])  # 2 covariates + 1 treatment
        input_stds = np.array([std['glucose'], std['glucose_history'], std['insulin']])      # normalized dosage
        
        num_treatments = 1  # Single continuous dosage value
    else:
        # One-hot encoding: Create treatment combinations (no insulin, insulin) - 2 categories instead of 4
        treatments = np.zeros((num_patients, original_length - offset, 2))
        for patient_id in range(num_patients):
            for timestep in range(original_length - offset):
                if insulin_application[patient_id, timestep] == 0:
                    treatments[patient_id, timestep] = [1, 0]  # no insulin
                else:
                    treatments[patient_id, timestep] = [0, 1]  # insulin given
        
        # Previous treatments (shifted by 1)
        previous_treatments = treatments[:, :-1, :]
        current_treatments = treatments
        
        # Set up input/output scaling for one-hot treatments
        input_means = np.array([mean['glucose'], mean['glucose_history'], 0, 0])  # 2 covariates + 2 treatments
        input_stds = np.array([std['glucose'], std['glucose_history'], 1, 1])    # binary treatments have std=1
        
        num_treatments = 2  # Two categories: no insulin, insulin
    
    output_means = mean['glucose']
    output_stds = std['glucose']
    
    # Active entries (follow cancer pattern)
    active_entries = np.zeros(outputs.shape)
    for i in range(sequence_lengths.shape[0]):
        sequence_length = int(min(sequence_lengths[i], outputs.shape[1]))
        active_entries[i, :sequence_length, :] = 1

    # Add unscaled outputs (required by evaluation)
    unscaled_outputs = outputs * std['glucose'] + mean['glucose']
    
    processed_data = {
        'current_covariates': current_covariates,
        'previous_treatments': previous_treatments,
        'current_treatments': current_treatments,
        'outputs': outputs,
        'active_entries': active_entries,
        'unscaled_outputs': unscaled_outputs,
        'sequence_lengths': sequence_lengths,
        'input_means': input_means,
        'input_stds': input_stds,
        'output_means': output_means,
        'output_stds': output_stds,
        'num_treatments': num_treatments
    }

    return processed_data


def get_mse_over_trajectory(prediction, true_y, scaling_params):
    """
    Compute MSE over the trajectory for glucose predictions.
    """
    mean, std = scaling_params
    
    # Denormalize predictions and true values
    if std['glucose'] > 0:
        prediction_denorm = prediction * std['glucose'] + mean['glucose']
        true_y_denorm = true_y * std['glucose'] + mean['glucose']
    else:
        prediction_denorm = prediction
        true_y_denorm = true_y
    
    mse = np.mean((prediction_denorm - true_y_denorm) ** 2)
    return mse


def compute_scaling_params_for_test(test_data, scaling_params):
    """
    Apply scaling parameters to test data.
    """
    mean, std = scaling_params
    
    # Process test data in the same way as training data
    processed_test = get_processed_data(test_data, scaling_params)
    
    return processed_test