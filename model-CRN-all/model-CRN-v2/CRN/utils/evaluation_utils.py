# Copyright (c) 2024, Diabetes CRN Adaptation
# Evaluation utilities for diabetes CRN modeling

import numpy as np
import pandas as pd
import pickle
import os

from CRN_model import CRN_Model


def write_results_to_file(filename, data):
    """Write results to pickle file."""
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=2)


def append_results_to_file(filename, data):
    """Append results to pickle file."""
    with open(filename, 'a+b') as handle:
        pickle.dump(data, handle, protocol=2)


def load_trained_model(dataset_test, hyperparams_file, model_name, model_folder, b_decoder_model=False):
    """Load a trained CRN model for diabetes."""
    _, length, num_covariates = dataset_test['current_covariates'].shape
    num_treatments = dataset_test['current_treatments'].shape[-1]  # Should be 1 for diabetes
    num_outputs = dataset_test['outputs'].shape[-1]

    params = {
        'num_treatments': num_treatments,
        'num_covariates': num_covariates,
        'num_outputs': num_outputs,
        'max_sequence_length': length,
        'num_epochs': 100
    }

    print("Loading best hyperparameters for model")
    with open(hyperparams_file, 'rb') as handle:
        best_hyperparams = pickle.load(handle)

    model = CRN_Model(params, best_hyperparams, b_train_decoder=b_decoder_model)
    model.load_model(model_name=model_name, model_folder=model_folder)
    return model


def process_diabetes_data_for_crn(patient_data, max_sequence_length=60):
    """
    Process diabetes data from data-api for CRN training.
    
    Args:
        patient_data: DataFrame from DiabetesAnalyzer.generate_patient_data()
        max_sequence_length: Maximum sequence length for CRN
        
    Returns:
        Tuple of (training_data, validation_data, test_data) dictionaries
    """
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Counterfactual-Recurrent-Network'))
    from utils.insulin_encoding import InsulinEncoder, DiabetesDataProcessor
    
    # Initialize encoder and processor
    insulin_encoder = InsulinEncoder(num_dose_levels=5)
    processor = DiabetesDataProcessor(insulin_encoder, max_sequence_length=max_sequence_length)
    
    # Process the data
    processed_data = processor.process_patient_data(patient_data)
    
    # Split data into train/validation/test (70/15/15 split)
    num_sequences = processed_data['current_covariates'].shape[0]
    train_end = int(0.7 * num_sequences)
    val_end = int(0.85 * num_sequences)
    
    def split_dataset(data_dict, start_idx, end_idx):
        return {key: value[start_idx:end_idx] for key, value in data_dict.items()}
    
    training_data = split_dataset(processed_data, 0, train_end)
    validation_data = split_dataset(processed_data, train_end, val_end)
    test_data = split_dataset(processed_data, val_end, num_sequences)
    
    return training_data, validation_data, test_data


def get_diabetes_crn_params(max_sequence_length=60):
    """
    Get CRN parameters for diabetes modeling.
    
    Args:
        max_sequence_length: Maximum sequence length
        
    Returns:
        Dictionary with CRN model parameters
    """
    return {
        'num_covariates': 4,  # glucose, carbs, exercise, stress
        'num_outputs': 1,     # glucose prediction
        'num_treatments': 1,  # insulin dose (integer encoding)
        'max_sequence_length': max_sequence_length,
        'num_epochs': 100
    }


def get_default_hyperparams():
    """Get default hyperparameters for diabetes CRN training."""
    return {
        'rnn_hidden_units': 24,
        'br_size': 12,
        'fc_hidden_units': 36,
        'learning_rate': 0.01,
        'batch_size': 64,
        'rnn_keep_prob': 0.9
    }


def train_diabetes_crn(patient_data, model_name, model_folder, 
                      max_sequence_length=60, hyperparams=None):
    """
    Train CRN model on diabetes data.
    
    Args:
        patient_data: DataFrame from DiabetesAnalyzer.generate_patient_data()
        model_name: Name for saving the model
        model_folder: Folder for saving the model
        max_sequence_length: Maximum sequence length
        hyperparams: Hyperparameters dict (uses defaults if None)
        
    Returns:
        Tuple of (trained_model, test_data)
    """
    # Process data
    training_data, validation_data, test_data = process_diabetes_data_for_crn(
        patient_data, max_sequence_length=max_sequence_length
    )
    
    # Get model parameters
    params = get_diabetes_crn_params(max_sequence_length=max_sequence_length)
    
    # Use default hyperparameters if none provided
    if hyperparams is None:
        hyperparams = get_default_hyperparams()
    
    # Create results directory
    os.makedirs(model_folder, exist_ok=True)
    
    # Create and train model
    model = CRN_Model(params, hyperparams)
    model.train(training_data, validation_data, model_name=model_name, model_folder=model_folder)
    
    return model, test_data


def evaluate_diabetes_model(model, test_data):
    """
    Evaluate trained diabetes CRN model.
    
    Args:
        model: Trained CRN_Model instance
        test_data: Test dataset dictionary
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    test_mse, mse_by_timestep = model.evaluate_predictions(test_data)
    
    # Get balancing representations for analysis
    balancing_reps = model.get_balancing_reps(test_data)
    
    return {
        'test_mse': test_mse,
        'mse_by_timestep': mse_by_timestep,
        'balancing_representations': balancing_reps
    }


def hyperparameter_search(patient_data, model_name, model_folder, 
                         search_space=None, max_sequence_length=60):
    """
    Perform hyperparameter search for diabetes CRN.
    
    Args:
        patient_data: DataFrame from DiabetesAnalyzer.generate_patient_data()
        model_name: Base name for saving models
        model_folder: Folder for saving models
        search_space: Dictionary defining hyperparameter ranges
        max_sequence_length: Maximum sequence length
        
    Returns:
        Dictionary with best hyperparameters and performance
    """
    if search_space is None:
        search_space = {
            'rnn_hidden_units': [16, 24, 32],
            'br_size': [8, 12, 16],
            'fc_hidden_units': [24, 36, 48],
            'learning_rate': [0.001, 0.01, 0.02],
            'batch_size': [32, 64, 128],
            'rnn_keep_prob': [0.8, 0.9, 0.95]
        }
    
    # Process data once
    training_data, validation_data, test_data = process_diabetes_data_for_crn(
        patient_data, max_sequence_length=max_sequence_length
    )
    
    params = get_diabetes_crn_params(max_sequence_length=max_sequence_length)
    
    best_mse = float('inf')
    best_hyperparams = None
    
    # Simple grid search (can be replaced with more sophisticated methods)
    import itertools
    
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    for combination in itertools.product(*values):
        hyperparams = dict(zip(keys, combination))
        
        print(f"Testing hyperparams: {hyperparams}")
        
        # Train model with current hyperparameters
        model = CRN_Model(params, hyperparams)
        model.train(training_data, validation_data, 
                   model_name=f"{model_name}_search", model_folder=model_folder)
        
        # Evaluate on validation set
        val_mse, _ = model.evaluate_predictions(validation_data)
        
        print(f"Validation MSE: {val_mse:.4f}")
        
        if val_mse < best_mse:
            best_mse = val_mse
            best_hyperparams = hyperparams.copy()
            print(f"New best MSE: {best_mse:.4f}")
    
    # Save best hyperparameters
    hyperparams_file = os.path.join(model_folder, f"{model_name}_best_hyperparams.pkl")
    with open(hyperparams_file, 'wb') as handle:
        pickle.dump(best_hyperparams, handle)
    
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Best validation MSE: {best_mse:.4f}")
    
    return {
        'best_hyperparams': best_hyperparams,
        'best_mse': best_mse,
        'hyperparams_file': hyperparams_file
    }