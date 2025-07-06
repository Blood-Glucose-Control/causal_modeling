"""
Glucose-specific CRN evaluation functions.
Adapted from CRN_encoder_evaluate.py and CRN_decoder_evaluate.py for glucose data.
"""

import logging
import numpy as np
from CRN_model import CRN_Model
from utils.glucose_evaluation_utils import write_results_to_file, load_trained_model, get_processed_data


def fit_CRN_encoder_glucose(dataset_train, dataset_val, model_name, model_dir, hyperparams_file,
                           b_hyperparam_opt, ordinal_treatments=False):
    _, length, num_covariates = dataset_train['current_covariates'].shape
    num_treatments = dataset_train['current_treatments'].shape[-1]
    num_outputs = dataset_train['outputs'].shape[-1]
    num_inputs = dataset_train['current_covariates'].shape[-1] + dataset_train['current_treatments'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    hyperparams = dict()
    num_simulations = 5 if b_hyperparam_opt else 1  # Reduced for testing
    best_validation_mse = 1000000

    if b_hyperparam_opt:
        logging.info("Performing hyperparameter optimization")
        for simulation in range(num_simulations):
            logging.info("Simulation {} out of {}".format(simulation + 1, num_simulations))

            hyperparams['rnn_hidden_units'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * num_inputs)
            hyperparams['br_size'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * num_inputs)
            hyperparams['fc_hidden_units'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * (hyperparams['br_size']))
            hyperparams['learning_rate'] = np.random.choice([0.01, 0.001])
            hyperparams['batch_size'] = np.random.choice([64, 128, 256])
            hyperparams['rnn_keep_prob'] = np.random.choice([0.7, 0.8, 0.9])

            logging.info("Current hyperparams used for training \n {}".format(hyperparams))
            model = CRN_Model(params, hyperparams)
            model.train(dataset_train, dataset_val, model_name, model_dir, ordinal_treatments=ordinal_treatments)
            validation_mse, _ = model.evaluate_predictions(dataset_val)

            if (validation_mse < best_validation_mse):
                logging.info(
                    "Updating best validation loss | Previous best validation loss: {} | Current best validation loss: {}".format(
                        best_validation_mse, validation_mse))
                best_validation_mse = validation_mse
                best_hyperparams = hyperparams.copy()

        write_results_to_file(hyperparams_file, best_hyperparams)

    else:
        # Use default hyperparams
        hyperparams['rnn_hidden_units'] = 60
        hyperparams['br_size'] = 60
        hyperparams['fc_hidden_units'] = 60
        hyperparams['learning_rate'] = 0.001
        hyperparams['batch_size'] = 128
        hyperparams['rnn_keep_prob'] = 0.8
        
        best_hyperparams = hyperparams.copy()
        write_results_to_file(hyperparams_file, best_hyperparams)
        
        logging.info("Using default hyperparams: {}".format(hyperparams))
        model = CRN_Model(params, hyperparams)
        model.train(dataset_train, dataset_val, model_name, model_dir, ordinal_treatments=ordinal_treatments)

    return best_hyperparams


def test_CRN_encoder_glucose(pickle_map, models_dir, encoder_model_name, encoder_hyperparams_file,
                            b_encoder_hyperparm_tuning, ordinal_treatments=False):

    logging.info("Fitting encoder")
    
    training_data = pickle_map['training_data']
    validation_data = pickle_map['validation_data']
    test_data = pickle_map['test_data']
    scaling_data = pickle_map['scaling_data']

    training_processed = get_processed_data(training_data, scaling_data, ordinal_treatments=ordinal_treatments)
    validation_processed = get_processed_data(validation_data, scaling_data, ordinal_treatments=ordinal_treatments)
    test_processed = get_processed_data(test_data, scaling_data, ordinal_treatments=ordinal_treatments)

    # Train the encoder
    fit_CRN_encoder_glucose(training_processed, validation_processed, encoder_model_name, models_dir,
                           encoder_hyperparams_file, b_encoder_hyperparm_tuning, ordinal_treatments=ordinal_treatments)

    # Load and evaluate the trained model
    model = load_trained_model(test_processed, encoder_hyperparams_file, encoder_model_name, models_dir, ordinal_treatments=ordinal_treatments)
    test_rmse, _ = model.evaluate_predictions(test_processed)

    logging.info("Test RMSE for encoder: {}".format(test_rmse))
    return test_rmse


def test_CRN_decoder_glucose(pickle_map, max_projection_horizon, projection_horizon,
                            models_dir, encoder_model_name, encoder_hyperparams_file,
                            decoder_model_name, decoder_hyperparams_file, b_decoder_hyperparm_tuning, ordinal_treatments=False):

    logging.info("Fitting decoder")
    
    training_data = pickle_map['training_data']
    validation_data = pickle_map['validation_data']
    test_data = pickle_map['test_data']
    scaling_data = pickle_map['scaling_data']

    training_processed = get_processed_data(training_data, scaling_data, ordinal_treatments=ordinal_treatments)
    validation_processed = get_processed_data(validation_data, scaling_data, ordinal_treatments=ordinal_treatments)
    test_processed = get_processed_data(test_data, scaling_data, ordinal_treatments=ordinal_treatments)

    # For simplicity, use same hyperparams as encoder for decoder
    # In a full implementation, you'd want separate decoder hyperparameter optimization
    if not b_decoder_hyperparm_tuning:
        # Copy encoder hyperparams to decoder
        with open(encoder_hyperparams_file, 'rb') as f:
            import pickle
            encoder_hyperparams = pickle.load(f)
        write_results_to_file(decoder_hyperparams_file, encoder_hyperparams)

    # Load encoder and decoder models
    encoder_model = load_trained_model(test_processed, encoder_hyperparams_file, encoder_model_name, models_dir, ordinal_treatments=ordinal_treatments)
    
    # For decoder evaluation, we would need to implement the sequence-to-sequence prediction
    # For now, return the encoder RMSE as a placeholder
    test_rmse, _ = encoder_model.evaluate_predictions(test_processed)
    
    logging.info("Test RMSE for decoder (placeholder): {}".format(test_rmse))
    return test_rmse