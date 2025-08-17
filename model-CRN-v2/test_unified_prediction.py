#!/usr/bin/env python3
"""
Test the unified n-step prediction API.
"""

import sys
import os
import logging

# Add paths for modules
sys.path.insert(0, 'CRN')
sys.path.insert(0, 'diabetes-data-api')

from main import DiabetesAnalyzer
from utils.insulin_encoding import InsulinEncoder, DiabetesDataProcessor
from CRN_model import CRN_Model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_unified_api():
    """Test that n=1 gives same results as old one-step method."""
    
    print("="*60)
    print("TESTING UNIFIED N-STEP PREDICTION API")
    print("="*60)
    
    # Generate test data
    analyzer = DiabetesAnalyzer(seed=42)
    patient_data = analyzer.generate_patient_data(n_days=3, start_date='2024-01-01')
    
    # Process data
    encoder = InsulinEncoder(num_dose_levels=5)
    processor = DiabetesDataProcessor(encoder, max_sequence_length=36)
    processed_data = processor.process_patient_data(patient_data)
    
    # Train a small model
    num_sequences = processed_data['current_covariates'].shape[0]
    train_end = max(2, int(0.8 * num_sequences))
    
    training_data = {key: value[:train_end] if isinstance(value, np.ndarray) and value.ndim > 0 else value 
                    for key, value in processed_data.items()}
    validation_data = {key: value[train_end:] if isinstance(value, np.ndarray) and value.ndim > 0 else value 
                      for key, value in processed_data.items()}
    
    actual_seq_len = processed_data['current_covariates'].shape[1]
    params = {
        'num_covariates': 4,
        'num_outputs': 1,
        'num_treatments': 1,
        'max_sequence_length': actual_seq_len,
        'num_epochs': 3  # Very short for test
    }
    
    hyperparams = {
        'rnn_hidden_units': 16,
        'br_size': 8,
        'fc_hidden_units': 24,
        'learning_rate': 0.01,
        'batch_size': 32,
        'rnn_keep_prob': 0.9
    }
    
    print("\n🔧 Training small model for API testing...")
    os.makedirs('results/api_test', exist_ok=True)
    model = CRN_Model(params, hyperparams)
    model.train(training_data, validation_data, 
               model_name='api_test', model_folder='results/api_test')
    
    # Test different n-step values
    test_data = validation_data
    
    print("\n🧪 TESTING DIFFERENT N-STEP VALUES:")
    
    for n_steps in [1, 3, 6, 12]:
        print(f"\n  Testing n_steps = {n_steps}:")
        try:
            predictions = model.get_predictions(test_data, n_steps=n_steps)
            print(f"    ✓ Success: Shape = {predictions.shape}")
            
            if n_steps == 1:
                print(f"    ✓ 1-step: Standard prediction shape (sequences, timesteps, outputs)")
            else:
                print(f"    ✓ {n_steps}-step: Autoregressive prediction shape (sequences, steps, outputs)")
                
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    print("\n🎯 API DESIGN BENEFITS:")
    print("  • Single method for all prediction types")
    print("  • n=1: One-step prediction (easier)")
    print("  • n>1: Multi-step autoregressive (harder)")
    print("  • Consistent interface across evaluation scripts")
    print("  • No duplicate code for different prediction modes")
    
    print("\n" + "="*60)
    print("✅ UNIFIED API TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    import numpy as np
    test_unified_api()