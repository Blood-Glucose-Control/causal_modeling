# Copyright (c) 2020, Ioana Bica

import numpy as np
import pandas as pd

from CRN_model import CRN_Model
from utils.insulin_encoding import InsulinEncoder, process_diabetes_treatments

import pickle


def write_results_to_file(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=2)

def append_results_to_file(filename, data):
    with open(filename, 'a+b') as handle:
        pickle.dump(data, handle, protocol=2)


def load_trained_model(dataset_test, hyperparams_file, model_name, model_folder, b_decoder_model=False):
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

    model.load_model(model_name=model_name, model_folder=model_folder)
    return model


def get_processed_data(raw_sim_data,
                       scaling_params,
                       treatment_encoding='onehot'):
    """
    Create formatted data to train both encoder and seq2seq architecture.
    
    Args:
        raw_sim_data: Raw simulation data
        scaling_params: Mean and std for normalization
        treatment_encoding: 'onehot' for cancer (backward compatibility) or 'integer' for diabetes
    """
    mean, std = scaling_params

    horizon = 1
    offset = 1

    # Handle cancer data (backward compatibility)
    if treatment_encoding == 'onehot':
        return get_processed_data_cancer(raw_sim_data, scaling_params)
    
    # Handle diabetes data with integer encoding
    elif treatment_encoding == 'integer':
        return get_processed_data_diabetes(raw_sim_data, scaling_params)
    
    else:
        raise ValueError(f"Unknown treatment encoding: {treatment_encoding}")


def get_processed_data_cancer(raw_sim_data, scaling_params):
    """
    Original cancer data processing with one-hot treatment encoding.
    """
    mean, std = scaling_params

    horizon = 1
    offset = 1

    mean['chemo_application'] = 0
    mean['radio_application'] = 0
    std['chemo_application'] = 1
    std['radio_application'] = 1

    input_means = mean[
        ['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()
    input_stds = std[['cancer_volume', 'patient_types', 'chemo_application', 'radio_application']].values.flatten()

    # Continuous values
    cancer_volume = (raw_sim_data['cancer_volume'] - mean['cancer_volume']) / std['cancer_volume']
    patient_types = (raw_sim_data['patient_types'] - mean['patient_types']) / std['patient_types']

    patient_types = np.stack([patient_types for t in range(cancer_volume.shape[1])], axis=1)

    # Binary application
    chemo_application = raw_sim_data['chemo_application']
    radio_application = raw_sim_data['radio_application']
    sequence_lengths = raw_sim_data['sequence_lengths']

    # Convert treatments to one-hot encoding

    treatments = np.concatenate(
        [chemo_application[:, :-offset, np.newaxis], radio_application[:, :-offset, np.newaxis]], axis=-1)

    one_hot_treatments = np.zeros(shape=(treatments.shape[0], treatments.shape[1], 4))
    for patient_id in range(treatments.shape[0]):
        for timestep in range(treatments.shape[1]):
            if (treatments[patient_id][timestep][0] == 0 and treatments[patient_id][timestep][1] == 0):
                one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]
            elif (treatments[patient_id][timestep][0] == 1 and treatments[patient_id][timestep][1] == 0):
                one_hot_treatments[patient_id][timestep] = [0, 1, 0, 0]
            elif (treatments[patient_id][timestep][0] == 0 and treatments[patient_id][timestep][1] == 1):
                one_hot_treatments[patient_id][timestep] = [0, 0, 1, 0]
            elif (treatments[patient_id][timestep][0] == 1 and treatments[patient_id][timestep][1] == 1):
                one_hot_treatments[patient_id][timestep] = [0, 0, 0, 1]

    one_hot_previous_treatments = one_hot_treatments[:, :-1, :]

    current_covariates = np.concatenate(
        [cancer_volume[:, :-offset, np.newaxis], patient_types[:, :-offset, np.newaxis]], axis=-1)
    outputs = cancer_volume[:, horizon:, np.newaxis]

    output_means = mean[['cancer_volume']].values.flatten()[0]  # because we only need scalars here
    output_stds = std[['cancer_volume']].values.flatten()[0]

    print(outputs.shape)

    # Add active entires
    active_entries = np.zeros(outputs.shape)

    for i in range(sequence_lengths.shape[0]):
        sequence_length = int(sequence_lengths[i])
        active_entries[i, :sequence_length, :] = 1

    raw_sim_data['current_covariates'] = current_covariates
    raw_sim_data['previous_treatments'] = one_hot_previous_treatments
    raw_sim_data['current_treatments'] = one_hot_treatments
    raw_sim_data['outputs'] = outputs
    raw_sim_data['active_entries'] = active_entries

    raw_sim_data['unscaled_outputs'] = (outputs * std['cancer_volume'] + mean['cancer_volume'])
    raw_sim_data['input_means'] = input_means
    raw_sim_data['inputs_stds'] = input_stds
    raw_sim_data['output_means'] = output_means
    raw_sim_data['output_stds'] = output_stds

    return raw_sim_data


def get_processed_data_diabetes(raw_sim_data, scaling_params):
    """
    Diabetes data processing with integer treatment encoding.
    
    Expects raw_sim_data to contain:
    - glucose: glucose levels over time
    - insulin_doses: continuous insulin doses
    - other covariates (carbs, exercise, stress, etc.)
    """
    mean, std = scaling_params
    
    horizon = 1
    offset = 1
    
    # Extract diabetes-specific variables
    glucose = raw_sim_data['glucose']
    insulin_doses = raw_sim_data['insulin_doses']
    sequence_lengths = raw_sim_data['sequence_lengths']
    
    # Normalize glucose (main outcome)
    if 'glucose' in mean:
        normalized_glucose = (glucose - mean['glucose']) / std['glucose']
    else:
        # Fallback normalization
        normalized_glucose = (glucose - glucose.mean()) / glucose.std()
    
    # Process covariates (add other diabetes covariates as needed)
    covariates_list = []
    
    # Add glucose as primary covariate
    covariates_list.append(normalized_glucose[:, :-offset, np.newaxis])
    
    # Add other diabetes covariates if available
    for covariate in ['carbs', 'exercise', 'stress', 'active_insulin']:
        if covariate in raw_sim_data:
            covariate_data = raw_sim_data[covariate]
            # Normalize if scaling params available
            if covariate in mean and covariate in std:
                normalized_covariate = (covariate_data - mean[covariate]) / std[covariate]
            else:
                # Simple standardization
                normalized_covariate = (covariate_data - covariate_data.mean()) / covariate_data.std()
            covariates_list.append(normalized_covariate[:, :-offset, np.newaxis])
    
    current_covariates = np.concatenate(covariates_list, axis=-1)
    
    # Process insulin doses with integer encoding
    encoder = InsulinEncoder(num_dose_levels=5, encoding_type='integer')
    
    # Discretize and encode insulin doses
    from utils.insulin_encoding import discretize_insulin_doses
    discrete_doses = discretize_insulin_doses(insulin_doses, num_levels=5)
    encoded_doses = encoder.encode_doses(discrete_doses)
    
    # Reshape for CRN format: [batch, time, features]
    # Note: previous_treatments should be 1 timestep shorter than current_treatments
    # because build_feed_dictionary will add a zero column at the beginning
    if encoded_doses.ndim == 2:
        encoded_treatments = encoded_doses[:, :-offset, np.newaxis]  # Remove last timestep  
        encoded_previous_treatments = encoded_doses[:, :-offset-1, np.newaxis]  # Remove last 2 timesteps
    else:
        encoded_treatments = encoded_doses[:, :-offset]
        encoded_previous_treatments = encoded_doses[:, :-offset-1]
    
    # Set outputs (glucose prediction)
    outputs = normalized_glucose[:, horizon:, np.newaxis]
    
    # Add active entries - adjust for the shortened sequences
    active_entries = np.zeros(outputs.shape)
    for i in range(sequence_lengths.shape[0]):
        # Sequence length should match the output length after processing
        actual_length = min(int(sequence_lengths[i]) - horizon, outputs.shape[1])
        active_entries[i, :actual_length, :] = 1
    
    # Update data dictionary
    raw_sim_data['current_covariates'] = current_covariates
    raw_sim_data['previous_treatments'] = encoded_previous_treatments
    raw_sim_data['current_treatments'] = encoded_treatments
    raw_sim_data['outputs'] = outputs
    raw_sim_data['active_entries'] = active_entries
    
    # Store normalization parameters
    if 'glucose' in mean and 'glucose' in std:
        raw_sim_data['output_means'] = mean['glucose']
        raw_sim_data['output_stds'] = std['glucose']
    else:
        raw_sim_data['output_means'] = glucose.mean()
        raw_sim_data['output_stds'] = glucose.std()
        
    raw_sim_data['unscaled_outputs'] = (outputs * raw_sim_data['output_stds'] + raw_sim_data['output_means'])
    
    # Store encoder for later use
    raw_sim_data['insulin_encoder'] = encoder
    
    print(f"Processed diabetes data - outputs shape: {outputs.shape}")
    print(f"Treatment encoding: integer, range: [{encoded_treatments.min():.3f}, {encoded_treatments.max():.3f}]")
    
    return raw_sim_data


def get_mse_at_follow_up_time(mean, output, active_entires):
        mses = np.sum(np.sum((mean - output) ** 2 * active_entires, axis=-1), axis=0) \
               / active_entires.sum(axis=0).sum(axis=-1)

        return pd.Series(mses, index=[idx for idx in range(len(mses))])


def train_BR_optimal_model(dataset_train, dataset_val, hyperparams_file, model_name, model_folder,
                           b_decoder_model=False):
    _, length, num_covariates = dataset_train['current_covariates'].shape
    num_treatments = dataset_train['current_treatments'].shape[-1]
    num_outputs = dataset_train['outputs'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    print("Loading best hyperparameters for model")
    with open(hyperparams_file, 'rb') as handle:
        best_hyperparams = pickle.load(handle)

    print("Best Hyperparameters")
    print(best_hyperparams)

    if (b_decoder_model):
        print(best_hyperparams)
        model = CRN_Model(params, best_hyperparams, b_train_decoder=True)
    else:
        model = CRN_Model(params, best_hyperparams)
    model.train(dataset_train, dataset_val, model_name=model_name, model_folder=model_folder)



