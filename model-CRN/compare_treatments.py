#!/usr/bin/env python3

"""
Compare ordinal vs one-hot treatment encoding performance.
Runs both models with reduced epochs for quick comparison.
"""

import os
import logging
import numpy as np
import time
from CRN_glucose_evaluate import test_CRN_encoder_glucose
from utils.glucose_simulation import get_glucose_sim_data

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def run_comparison():
    """Run both ordinal and one-hot models for comparison."""
    
    # Load glucose data
    data_path = '../Data/ml_dataset.csv'
    pickle_map = get_glucose_sim_data(
        data_path=data_path,
        sequence_length=20,
        prediction_horizon=5
    )
    
    results = {}
    
    # Test Ordinal Treatment Encoding
    logging.info("=" * 60)
    logging.info("TESTING ORDINAL TREATMENT ENCODING")
    logging.info("=" * 60)
    
    start_time = time.time()
    
    ordinal_model_name = 'encoder_comparison_ordinal'
    ordinal_results_dir = 'results_comparison_ordinal'
    if not os.path.exists(ordinal_results_dir):
        os.mkdir(ordinal_results_dir)
    
    ordinal_models_dir = f'{ordinal_results_dir}/crn_models'
    if not os.path.exists(ordinal_models_dir):
        os.mkdir(ordinal_models_dir)
    
    ordinal_hyperparams_file = f'{ordinal_results_dir}/{ordinal_model_name}_best_hyperparams.txt'
    
    try:
        rmse_ordinal = test_CRN_encoder_glucose(
            pickle_map=pickle_map,
            models_dir=ordinal_models_dir,
            encoder_model_name=ordinal_model_name,
            encoder_hyperparams_file=ordinal_hyperparams_file,
            b_encoder_hyperparm_tuning=False,
            ordinal_treatments=True
        )
        results['ordinal_rmse'] = rmse_ordinal
        results['ordinal_time'] = time.time() - start_time
        logging.info(f"✓ Ordinal RMSE: {rmse_ordinal:.4f}")
    except Exception as e:
        logging.error(f"✗ Ordinal training failed: {e}")
        results['ordinal_rmse'] = None
        results['ordinal_time'] = None
    
    # Test One-Hot Treatment Encoding
    logging.info("=" * 60)
    logging.info("TESTING ONE-HOT TREATMENT ENCODING")
    logging.info("=" * 60)
    
    start_time = time.time()
    
    onehot_model_name = 'encoder_comparison_onehot'
    onehot_results_dir = 'results_comparison_onehot'
    if not os.path.exists(onehot_results_dir):
        os.mkdir(onehot_results_dir)
    
    onehot_models_dir = f'{onehot_results_dir}/crn_models'
    if not os.path.exists(onehot_models_dir):
        os.mkdir(onehot_models_dir)
    
    onehot_hyperparams_file = f'{onehot_results_dir}/{onehot_model_name}_best_hyperparams.txt'
    
    try:
        rmse_onehot = test_CRN_encoder_glucose(
            pickle_map=pickle_map,
            models_dir=onehot_models_dir,
            encoder_model_name=onehot_model_name,
            encoder_hyperparams_file=onehot_hyperparams_file,
            b_encoder_hyperparm_tuning=False,
            ordinal_treatments=False
        )
        results['onehot_rmse'] = rmse_onehot
        results['onehot_time'] = time.time() - start_time
        logging.info(f"✓ One-Hot RMSE: {rmse_onehot:.4f}")
    except Exception as e:
        logging.error(f"✗ One-Hot training failed: {e}")
        results['onehot_rmse'] = None
        results['onehot_time'] = None
    
    # Print Comparison Results
    print("\\n" + "=" * 80)
    print("FINAL COMPARISON RESULTS")
    print("=" * 80)
    
    if results['ordinal_rmse'] is not None and results['onehot_rmse'] is not None:
        print(f"Ordinal Treatment Encoding:")
        print(f"  • RMSE: {results['ordinal_rmse']:.6f}")
        print(f"  • Training Time: {results['ordinal_time']:.1f}s")
        print(f"  • Architecture: Continuous dosage, MSE loss")
        print(f"  • Capability: Timing + Intensity")
        
        print(f"\\nOne-Hot Treatment Encoding:")
        print(f"  • RMSE: {results['onehot_rmse']:.6f}")
        print(f"  • Training Time: {results['onehot_time']:.1f}s")
        print(f"  • Architecture: Binary categories, Cross-entropy loss")
        print(f"  • Capability: Timing only")
        
        improvement = ((results['onehot_rmse'] - results['ordinal_rmse']) / results['onehot_rmse']) * 100
        print(f"\\nPerformance Comparison:")
        if improvement > 0:
            print(f"  ✓ Ordinal encoding is {improvement:.2f}% better (lower RMSE)")
        elif improvement < 0:
            print(f"  • One-hot encoding is {-improvement:.2f}% better (lower RMSE)")
        else:
            print(f"  • Both encodings performed equally")
        
        print(f"\\nKey Insight:")
        print(f"  Even if RMSE is similar, ordinal encoding provides:")
        print(f"  ✓ Continuous treatment dosage modeling")
        print(f"  ✓ Better interpretability for dosing decisions")
        print(f"  ✓ More suitable for treatment optimization")
    else:
        print("Could not complete comparison due to training errors.")
    
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    results = run_comparison()