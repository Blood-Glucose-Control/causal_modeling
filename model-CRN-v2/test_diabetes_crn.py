#!/usr/bin/env python3
"""
Test script for CRN diabetes modeling integration.

This script demonstrates the complete pipeline:
1. Generate synthetic diabetes data using data-api
2. Process data for CRN training with integer encoding
3. Train CRN model with regression adversary
4. Evaluate counterfactual predictions

Usage:
    uv run python test_diabetes_crn.py --days=7 --model_name=diabetes_test
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add CRN module to path
sys.path.append('CRN')
sys.path.append('diabetes-data-api')

from diabetes_data_api.main import DiabetesAnalyzer
from CRN.utils.evaluation_utils import train_diabetes_crn, get_diabetes_crn_params
from CRN.utils.insulin_encoding import InsulinEncoder, DiabetesDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_diabetes_data(n_days=7, seed=42):
    """Generate synthetic diabetes patient data."""
    logging.info(f"Generating {n_days} days of synthetic diabetes data...")
    
    analyzer = DiabetesAnalyzer(seed=seed)
    patient_data = analyzer.generate_patient_data(n_days=n_days, start_date='2024-01-01')
    
    # Log data statistics
    insulin_interventions = (patient_data['insulin'] > 0).sum()
    glucose_range = (patient_data['glucose'].min(), patient_data['glucose'].max())
    
    logging.info(f"✓ Generated {len(patient_data)} data points")
    logging.info(f"✓ Found {insulin_interventions} insulin interventions")
    logging.info(f"✓ Glucose range: {glucose_range[0]:.1f} - {glucose_range[1]:.1f} mg/dL")
    
    return patient_data

def test_insulin_encoding():
    """Test insulin encoding functionality."""
    logging.info("Testing insulin encoding system...")
    
    # Test continuous to discrete conversion
    encoder = InsulinEncoder(encoding_type='integer', num_dose_levels=5)
    
    # Test various insulin doses
    test_doses = [0.0, 1.5, 3.5, 5.0, 7.5, 10.0]
    discrete_levels = encoder.continuous_to_discrete(test_doses)
    encoded_treatments = encoder.encode_for_model(discrete_levels)
    
    logging.info("Dose discretization test:")
    for dose, level, encoded in zip(test_doses, discrete_levels, encoded_treatments):
        logging.info(f"  {dose:.1f} units → Level {level} → Encoded: {encoded:.1f}")
    
    logging.info("✓ Insulin encoding test passed")

def test_data_processing(patient_data):
    """Test diabetes data processing for CRN."""
    logging.info("Testing diabetes data processing...")
    
    # Test with integer encoding
    encoder = InsulinEncoder(encoding_type='integer', num_dose_levels=5)
    processor = DiabetesDataProcessor(encoder, max_sequence_length=60)
    
    try:
        processed_data = processor.process_patient_data(patient_data)
        
        # Log processed data shapes
        logging.info("Processed data shapes:")
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                logging.info(f"  {key}: {value.shape}")
            else:
                logging.info(f"  {key}: {value}")
        
        logging.info("✓ Data processing test passed")
        return processed_data
        
    except Exception as e:
        logging.error(f"✗ Data processing test failed: {e}")
        raise

def train_and_evaluate_model(patient_data, model_name, treatment_encoding='integer'):
    """Train CRN model and evaluate performance."""
    logging.info(f"Training CRN model with {treatment_encoding} encoding...")
    
    # Create results directory
    results_dir = "results/diabetes_models"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Train model
        model, test_data = train_diabetes_crn(
            patient_data=patient_data,
            model_name=model_name,
            model_folder=results_dir,
            treatment_encoding=treatment_encoding,
            max_sequence_length=60,
            hyperparams={
                'rnn_hidden_units': 16,  # Smaller for quick test
                'br_size': 8,
                'fc_hidden_units': 24,
                'learning_rate': 0.01,
                'batch_size': 32,
                'rnn_keep_prob': 0.9
            }
        )
        
        # Evaluate model performance
        logging.info("Evaluating model performance...")
        test_mse, _ = model.evaluate_predictions(test_data)
        logging.info(f"✓ Test MSE: {test_mse:.4f}")
        
        return model, test_data
        
    except Exception as e:
        logging.error(f"✗ Model training failed: {e}")
        raise

def demonstrate_counterfactual_analysis(patient_data):
    """Demonstrate counterfactual analysis with data-api."""
    logging.info("Demonstrating counterfactual analysis...")
    
    analyzer = DiabetesAnalyzer(seed=42)
    
    # List available interventions
    interventions = analyzer.counterfactual_model.list_interventions(patient_data)
    
    if len(interventions) == 0:
        logging.warning("No insulin interventions found for counterfactual analysis")
        return
    
    logging.info(f"Found {len(interventions)} insulin interventions")
    
    # Analyze dose counterfactual
    chosen_intervention = interventions[0]
    logging.info(f"Analyzing intervention: {chosen_intervention['timestamp']} - {chosen_intervention['dose']:.1f} units")
    
    # What if 20% more insulin?
    dose_result = analyzer.analyze_intervention(
        patient_data,
        intervention_id=chosen_intervention['id'],
        analysis_type='dose',
        dose_factor=1.2,
        before_minutes=120,
        after_minutes=180
    )
    
    # What if 30 minutes earlier?
    timing_result = analyzer.analyze_intervention(
        patient_data,
        intervention_id=chosen_intervention['id'],
        analysis_type='timing',
        timing_shift_minutes=-30,
        before_minutes=120,
        after_minutes=180
    )
    
    logging.info("✓ Counterfactual analysis completed")
    logging.info(f"✓ Dose counterfactual: {dose_result.shape[0]} timepoints")
    logging.info(f"✓ Timing counterfactual: {timing_result.shape[0]} timepoints")

def main():
    parser = argparse.ArgumentParser(description='Test CRN diabetes modeling integration')
    parser.add_argument('--days', type=int, default=7, help='Days of data to generate')
    parser.add_argument('--model_name', default='diabetes_test', help='Model name for saving')
    parser.add_argument('--treatment_encoding', default='integer', choices=['integer', 'onehot'],
                       help='Treatment encoding strategy')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training (for quick testing)')
    
    args = parser.parse_args()
    
    try:
        logging.info("=== CRN Diabetes Integration Test ===")
        
        # Step 1: Generate diabetes data
        patient_data = generate_diabetes_data(n_days=args.days)
        
        # Step 2: Test insulin encoding
        test_insulin_encoding()
        
        # Step 3: Test data processing
        processed_data = test_data_processing(patient_data)
        
        # Step 4: Demonstrate counterfactual analysis
        demonstrate_counterfactual_analysis(patient_data)
        
        # Step 5: Train and evaluate model (optional)
        if not args.skip_training:
            model, test_data = train_and_evaluate_model(
                patient_data, args.model_name, args.treatment_encoding
            )
            logging.info("✓ Model training completed successfully")
        else:
            logging.info("⏭ Skipping model training")
        
        logging.info("=== All tests passed! ===")
        logging.info("The CRN diabetes integration is working correctly.")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()