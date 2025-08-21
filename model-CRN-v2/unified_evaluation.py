#!/usr/bin/env python3
"""
Unified CRN evaluation using the n-step prediction API.

This script can test both one-step (n=1) and multi-step (n>1) prediction
using the same codebase.
"""

import sys
import os
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add paths for modules
sys.path.insert(0, 'CRN')
sys.path.insert(0, 'diabetes-data-api')

from main import DiabetesAnalyzer
from utils.insulin_encoding import InsulinEncoder, DiabetesDataProcessor
from CRN_model import CRN_Model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class UnifiedCounterfactualEvaluation:
    """
    Unified evaluation that can test any n-step prediction horizon.
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        self.analyzer = DiabetesAnalyzer(seed=seed)
        self.encoder = InsulinEncoder(num_dose_levels=5)
        self.processor = DiabetesDataProcessor(self.encoder, max_sequence_length=36)
    
    def run_evaluation(self, n_days=7, n_steps=1, n_interventions=5):
        """
        Run counterfactual evaluation with specified prediction horizon.
        
        Args:
            n_days: Days of patient data to generate
            n_steps: Prediction horizon (1=one-step, >1=multi-step)
            n_interventions: Number of interventions to test
        """
        prediction_type = "ONE-STEP" if n_steps == 1 else f"{n_steps}-STEP AUTOREGRESSIVE"
        logging.info(f"=== UNIFIED CRN EVALUATION: {prediction_type} ===")
        
        if n_steps == 1:
            logging.info("Testing immediate glucose response prediction (easier)")
        else:
            logging.info(f"Testing {n_steps*5}-minute glucose trajectory prediction (harder)")
        
        # Generate and train
        baseline_data = self.analyzer.generate_patient_data(n_days=n_days, start_date='2024-01-01')
        crn_model = self._train_model(baseline_data)
        
        # Test interventions
        interventions = self.analyzer.counterfactual_model.list_interventions(baseline_data)
        test_interventions = interventions[:n_interventions]
        
        results = []
        
        for i, intervention in enumerate(test_interventions):
            logging.info(f"Evaluating intervention {i+1}/{len(test_interventions)} [{prediction_type}]")
            
            # Test dose changes
            for dose_factor in [0.8, 1.2]:
                logging.info(f"  Testing {dose_factor:.1f}x dose with {n_steps}-step prediction")
                result = self._evaluate_counterfactual(
                    baseline_data, intervention, crn_model, n_steps,
                    'dose', dose_factor=dose_factor
                )
                if result:
                    results.append(result)
            
            # Test timing changes  
            for timing_shift in [-20, 20]:
                logging.info(f"  Testing {timing_shift:+d}min timing with {n_steps}-step prediction")
                result = self._evaluate_counterfactual(
                    baseline_data, intervention, crn_model, n_steps,
                    'timing', timing_shift_minutes=timing_shift
                )
                if result:
                    results.append(result)
        
        return self._analyze_results(results, n_steps)
    
    def _train_model(self, baseline_data):
        """Train CRN model on baseline data."""
        processed_data = self.processor.process_patient_data(baseline_data)
        
        num_sequences = processed_data['current_covariates'].shape[0]
        train_end = max(4, int(0.8 * num_sequences))
        
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
            'num_epochs': 15
        }
        
        hyperparams = {
            'rnn_hidden_units': 32,
            'br_size': 16,
            'fc_hidden_units': 48,
            'learning_rate': 0.005,
            'batch_size': 32,
            'rnn_keep_prob': 0.9
        }
        
        os.makedirs('results/unified_evaluation', exist_ok=True)
        model = CRN_Model(params, hyperparams)
        model.train(training_data, validation_data, 
                   model_name='unified_eval', model_folder='results/unified_evaluation')
        
        return model
    
    def _evaluate_counterfactual(self, baseline_data, intervention, crn_model, n_steps,
                               cf_type, dose_factor=None, timing_shift_minutes=None):
        """Evaluate a single counterfactual with n-step prediction."""
        try:
            # Generate ground truth
            if cf_type == 'dose':
                gt_data = self.analyzer.analyze_intervention(
                    baseline_data,
                    intervention_id=intervention['id'],
                    analysis_type='dose',
                    dose_factor=dose_factor,
                    before_minutes=60,
                    after_minutes=n_steps * 5
                )
                cf_params = {'dose_factor': dose_factor}
            else:  # timing
                gt_data = self.analyzer.analyze_intervention(
                    baseline_data,
                    intervention_id=intervention['id'],
                    analysis_type='timing',
                    timing_shift_minutes=timing_shift_minutes,
                    before_minutes=60,
                    after_minutes=n_steps * 5
                )
                cf_params = {'timing_shift_minutes': timing_shift_minutes}
            
            # Extract ground truth trajectory
            cf_columns = [col for col in gt_data.columns if col.startswith('cf') and col.endswith('_glucose')]
            if not cf_columns:
                return None
            
            intervention_time = intervention['timestamp']
            prediction_start = intervention_time
            prediction_end = prediction_start + pd.Timedelta(minutes=n_steps * 5)
            
            prediction_window = gt_data.loc[prediction_start:prediction_end]
            gt_glucose = prediction_window[cf_columns[0]].dropna().values
            
            if len(gt_glucose) < n_steps:
                return None
            
            gt_glucose = gt_glucose[:n_steps]
            
            # Get CRN prediction
            crn_glucose = self._get_crn_prediction(
                baseline_data, intervention, crn_model, n_steps, cf_type, cf_params
            )
            
            if crn_glucose is None or len(crn_glucose) < n_steps:
                return None
            
            crn_glucose = crn_glucose[:n_steps]
            
            # Calculate metrics
            mse = mean_squared_error(gt_glucose, crn_glucose)
            mae = mean_absolute_error(gt_glucose, crn_glucose)
            correlation = np.corrcoef(gt_glucose, crn_glucose)[0, 1] if len(set(gt_glucose)) > 1 else 0
            
            return {
                'cf_type': cf_type,
                'cf_params': cf_params,
                'n_steps': n_steps,
                'gt_glucose': gt_glucose,
                'crn_glucose': crn_glucose,
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'correlation': correlation
            }
            
        except Exception as e:
            logging.warning(f"Failed to evaluate counterfactual: {e}")
            return None
    
    def _get_crn_prediction(self, baseline_data, intervention, crn_model, n_steps, cf_type, cf_params):
        """Get CRN prediction using unified n-step API."""
        try:
            # Create modified data
            modified_data = baseline_data.copy()
            
            if cf_type == 'dose':
                intervention_time = intervention['timestamp']
                new_dose = intervention['dose'] * cf_params['dose_factor']
                modified_data.loc[intervention_time, 'insulin'] = new_dose
            else:  # timing
                original_time = intervention['timestamp']
                new_time = original_time + pd.Timedelta(minutes=cf_params['timing_shift_minutes'])
                modified_data.loc[original_time, 'insulin'] = 0.0
                if new_time in modified_data.index:
                    modified_data.loc[new_time, 'insulin'] = intervention['dose']
            
            # Process and predict
            processed_data = self.processor.process_patient_data(modified_data)
            predictions = crn_model.get_predictions(processed_data, n_steps=n_steps)
            denorm_predictions = predictions * processed_data['output_stds'] + processed_data['output_means']
            
            if n_steps == 1:
                # One-step: extract from sequence dimension
                return np.mean(denorm_predictions, axis=0).flatten()
            else:
                # Multi-step: extract from step dimension
                return denorm_predictions[0, :, 0]  # [sequence, steps, outputs]
            
        except Exception as e:
            logging.warning(f"Failed to get CRN prediction: {e}")
            return None
    
    def _analyze_results(self, results, n_steps):
        """Analyze evaluation results."""
        if not results:
            logging.error("No successful evaluations!")
            return {}
        
        rmse_scores = [r['rmse'] for r in results]
        correlation_scores = [r['correlation'] for r in results if not np.isnan(r['correlation'])]
        
        summary = {
            'n_steps': n_steps,
            'prediction_horizon_minutes': n_steps * 5,
            'total_evaluations': len(results),
            'mean_rmse_mg_dl': np.mean(rmse_scores),
            'mean_correlation': np.mean(correlation_scores) if correlation_scores else 0,
            'rmse_std': np.std(rmse_scores)
        }
        
        self._print_report(summary)
        return summary
    
    def _print_report(self, summary):
        """Print evaluation report."""
        n_steps = summary['n_steps']
        horizon = summary['prediction_horizon_minutes']
        
        print("\n" + "="*60)
        print(f"UNIFIED EVALUATION REPORT: {n_steps}-STEP PREDICTION")
        print("="*60)
        
        print(f"\n🎯 PREDICTION PERFORMANCE")
        print(f"   Prediction Mode: {'One-step' if n_steps == 1 else f'{n_steps}-step autoregressive'}")
        print(f"   Time Horizon: {horizon} minutes")
        print(f"   Evaluations: {summary['total_evaluations']}")
        print(f"   RMSE: {summary['mean_rmse_mg_dl']:.2f} ± {summary['rmse_std']:.2f} mg/dL")
        print(f"   Correlation: {summary['mean_correlation']:.3f}")
        
        # Assessment
        rmse = summary['mean_rmse_mg_dl']
        if n_steps == 1:
            if rmse < 20:
                assessment = "🟢 EXCELLENT one-step prediction"
            elif rmse < 30:
                assessment = "🟡 GOOD one-step prediction"
            else:
                assessment = "🔴 POOR one-step prediction"
        else:
            if rmse < 30:
                assessment = f"🟢 EXCELLENT {n_steps}-step trajectory prediction"
            elif rmse < 50:
                assessment = f"🟡 GOOD {n_steps}-step trajectory prediction"
            else:
                assessment = f"🔴 POOR {n_steps}-step trajectory prediction"
        
        print(f"\n⭐ ASSESSMENT: {assessment}")
        print("="*60)


def main():
    """Run unified evaluations for comparison."""
    evaluator = UnifiedCounterfactualEvaluation(seed=42)
    
    print("🔬 COMPARING DIFFERENT PREDICTION HORIZONS")
    print("="*60)
    
    # Test different prediction horizons
    for n_steps in [1, 6, 12]:
        results = evaluator.run_evaluation(n_days=7, n_steps=n_steps, n_interventions=3)
        print()  # Space between evaluations
    
    print("\n🎯 SUMMARY: Unified API enables easy comparison of prediction horizons!")
    print("  • n=1: Tests immediate response (easier)")
    print("  • n=6: Tests 30-minute trajectory (moderate)")  
    print("  • n=12: Tests 1-hour trajectory (challenging)")


if __name__ == "__main__":
    import pandas as pd
    main()