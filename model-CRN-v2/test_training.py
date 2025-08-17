#!/usr/bin/env python3
"""
Simple test of CRN training with diabetes data.
"""

import sys
import os

# Add paths for modules
crn_path = os.path.join(os.path.dirname(__file__), 'CRN')
api_path = os.path.join(os.path.dirname(__file__), 'diabetes-data-api')

sys.path.insert(0, crn_path)
sys.path.insert(0, api_path)

import numpy as np
import logging

from main import DiabetesAnalyzer
from utils.insulin_encoding import InsulinEncoder, DiabetesDataProcessor
from CRN_model import CRN_Model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_full_pipeline():
    """Test the complete diabetes CRN pipeline."""
    
    # Step 1: Generate diabetes data
    logging.info("=== Testing Complete Diabetes CRN Pipeline ===")
    logging.info("Generating diabetes data...")
    
    analyzer = DiabetesAnalyzer(seed=42)
    patient_data = analyzer.generate_patient_data(n_days=5, start_date='2024-01-01')
    
    insulin_interventions = (patient_data['insulin'] > 0).sum()
    logging.info(f"✓ Generated {len(patient_data)} data points with {insulin_interventions} interventions")
    
    # Step 2: Process data for CRN
    logging.info("Processing data for CRN...")
    
    encoder = InsulinEncoder(num_dose_levels=5)
    processor = DiabetesDataProcessor(encoder, max_sequence_length=36)  # 3 hours
    
    processed_data = processor.process_patient_data(patient_data)
    
    # Debug: Check actual sequence length
    actual_seq_len = processed_data['current_covariates'].shape[1]
    logging.info(f"Actual sequence length: {actual_seq_len}")
    
    # Split into train/val/test
    num_sequences = processed_data['current_covariates'].shape[0]
    train_end = int(0.7 * num_sequences)
    val_end = int(0.85 * num_sequences)
    
    def split_dataset(data_dict, start_idx, end_idx):
        result = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                result[key] = value[start_idx:end_idx]
            else:
                result[key] = value  # Keep scalar values as-is
        return result
    
    training_data = split_dataset(processed_data, 0, train_end)
    validation_data = split_dataset(processed_data, train_end, val_end)
    test_data = split_dataset(processed_data, val_end, num_sequences)
    
    logging.info(f"✓ Split data: train={len(training_data['outputs'])}, val={len(validation_data['outputs'])}, test={len(test_data['outputs'])}")
    
    # Step 3: Set up model parameters
    params = {
        'num_covariates': 4,  # glucose, carbs, exercise, stress
        'num_outputs': 1,     # glucose prediction
        'num_treatments': 1,  # insulin dose (integer encoding)
        'max_sequence_length': actual_seq_len,  # Use actual length from data
        'num_epochs': 3  # Short test
    }
    
    hyperparams = {
        'rnn_hidden_units': 16,  # Small for quick test
        'br_size': 8,
        'fc_hidden_units': 24,
        'learning_rate': 0.01,
        'batch_size': 16,
        'rnn_keep_prob': 0.9
    }
    
    logging.info("Creating and training CRN model...")
    
    # Step 4: Create and train model
    try:
        # Create results directory
        os.makedirs('results/test_models', exist_ok=True)
        
        model = CRN_Model(params, hyperparams)
        model.train(training_data, validation_data, 
                   model_name='diabetes_test', model_folder='results/test_models')
        
        logging.info("✓ Model training completed!")
        
        # Step 5: Evaluate model
        test_mse, _ = model.evaluate_predictions(test_data)
        logging.info(f"✓ Test MSE: {test_mse:.6f}")
        
        # Step 6: Test counterfactual analysis
        logging.info("Testing counterfactual analysis...")
        
        interventions = analyzer.counterfactual_model.list_interventions(patient_data)
        if len(interventions) > 0:
            intervention = interventions[0]
            result = analyzer.analyze_intervention(
                patient_data,
                intervention_id=intervention['id'],
                analysis_type='dose',
                dose_factor=1.2,
                before_minutes=60,
                after_minutes=120
            )
            logging.info(f"✓ Counterfactual analysis: {result.shape}")
        
        logging.info("=== ALL TESTS PASSED ===")
        logging.info("The complete diabetes CRN pipeline is working!")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    if not success:
        sys.exit(1)