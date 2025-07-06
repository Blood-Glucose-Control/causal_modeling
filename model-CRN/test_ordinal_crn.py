#!/usr/bin/env python3

"""
Test script for ordinal treatment encoding in CRN model.
This script demonstrates how to use the modified CRN with continuous insulin dosage.
"""

import numpy as np
import argparse
import logging
from CRN_model import CRN_Model
from utils.glucose_evaluation_utils import get_processed_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_glucose_data(num_patients=100, sequence_length=20, seed=42):
    """Generate synthetic glucose data with continuous insulin dosing."""
    np.random.seed(seed)
    
    # Generate realistic glucose and insulin data
    glucose_baseline = np.random.normal(120, 20, num_patients)  # mg/dL
    insulin_doses = np.random.exponential(2, (num_patients, sequence_length))  # units, continuous
    
    # Create time series with realistic patterns
    current_covariates = np.zeros((num_patients, sequence_length, 2))
    current_treatments = np.zeros((num_patients, sequence_length, 1))
    outputs = np.zeros((num_patients, sequence_length, 1))
    
    for i in range(num_patients):
        glucose_trajectory = [glucose_baseline[i]]
        for t in range(sequence_length):
            # Simulate glucose dynamics with insulin effect
            prev_glucose = glucose_trajectory[-1] if t > 0 else glucose_baseline[i]
            insulin_effect = -insulin_doses[i, t] * 10  # insulin reduces glucose
            noise = np.random.normal(0, 5)
            
            new_glucose = prev_glucose + insulin_effect + noise
            new_glucose = max(50, min(300, new_glucose))  # physiological bounds
            
            glucose_trajectory.append(new_glucose)
            
            # Store data
            current_covariates[i, t, 0] = prev_glucose  # current glucose
            current_covariates[i, t, 1] = glucose_baseline[i]  # glucose history (patient type)
            current_treatments[i, t, 0] = insulin_doses[i, t]  # continuous insulin dose
            outputs[i, t, 0] = new_glucose  # next glucose
    
    return {
        'current_covariates': current_covariates,
        'current_treatments': current_treatments,
        'outputs': outputs
    }

def test_ordinal_encoding():
    """Test ordinal treatment encoding vs one-hot encoding."""
    logger.info("Testing ordinal treatment encoding...")
    
    # Generate synthetic data
    raw_data = generate_synthetic_glucose_data(num_patients=50, sequence_length=15)
    
    # Create scaling parameters
    scaling_params = (
        {  # means
            'glucose': 120.0,
            'glucose_history': 120.0,
            'insulin': 2.0
        },
        {  # stds
            'glucose': 30.0,
            'glucose_history': 20.0,
            'insulin': 1.5
        }
    )
    
    # Test one-hot encoding
    logger.info("Processing data with one-hot encoding...")
    data_onehot = get_processed_data(raw_data, scaling_params, ordinal_treatments=False)
    logger.info(f"One-hot treatments shape: {data_onehot['current_treatments'].shape}")
    logger.info(f"One-hot num_treatments: {data_onehot['num_treatments']}")
    
    # Test ordinal encoding
    logger.info("Processing data with ordinal encoding...")
    data_ordinal = get_processed_data(raw_data, scaling_params, ordinal_treatments=True)
    logger.info(f"Ordinal treatments shape: {data_ordinal['current_treatments'].shape}")
    logger.info(f"Ordinal num_treatments: {data_ordinal['num_treatments']}")
    
    # Test model creation and basic forward pass
    logger.info("Testing model creation...")
    
    # Parameters for ordinal model
    params_ordinal = {
        'num_treatments': data_ordinal['num_treatments'],
        'num_covariates': data_ordinal['current_covariates'].shape[-1],
        'num_outputs': data_ordinal['outputs'].shape[-1],
        'max_sequence_length': data_ordinal['current_treatments'].shape[1],
        'num_epochs': 5
    }
    
    # Basic hyperparameters
    hyperparams = {
        'br_size': 32,
        'rnn_hidden_units': 64,
        'fc_hidden_units': 32,
        'batch_size': 10,
        'rnn_keep_prob': 0.8,
        'learning_rate': 0.001
    }
    
    # Create model with ordinal treatments
    logger.info("Creating CRN model with ordinal treatments...")
    model = CRN_Model(params_ordinal, hyperparams)
    
    # Test training setup
    logger.info("Testing training setup...")
    try:
        model.train(data_ordinal, data_ordinal, "test_ordinal", "test_results", ordinal_treatments=True)
        logger.info("✓ Ordinal CRN training completed successfully!")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test ordinal treatment encoding in CRN')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    logger.info("Starting CRN ordinal treatment encoding test...")
    
    try:
        test_ordinal_encoding()
        logger.info("✓ All tests passed! Ordinal treatment encoding is working correctly.")
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())