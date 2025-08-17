#!/usr/bin/env python3
"""
Train CRN model for diabetes management.

This script provides a complete pipeline for training the Counterfactual Recurrent Network
on diabetes data using ordinal insulin dosage encoding and regression adversarial training.

Usage:
    uv run python train_diabetes_crn.py --days=14 --model_name=diabetes_crn_v1
    uv run python train_diabetes_crn.py --days=30 --model_name=diabetes_crn_v1 --hyperparameter_search
"""

import argparse
import sys
import os
import logging

# Add paths for modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'CRN'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'diabetes-data-api'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Counterfactual-Recurrent-Network'))

from main import DiabetesAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def generate_diabetes_training_data(n_days=14, seed=42):
    """Generate comprehensive diabetes training data."""
    logging.info(f"Generating {n_days} days of diabetes training data...")
    
    analyzer = DiabetesAnalyzer(seed=seed)
    patient_data = analyzer.generate_patient_data(n_days=n_days, start_date='2024-01-01')
    
    # Log data statistics
    insulin_interventions = (patient_data['insulin'] > 0).sum()
    glucose_stats = {
        'mean': patient_data['glucose'].mean(),
        'std': patient_data['glucose'].std(),
        'min': patient_data['glucose'].min(),
        'max': patient_data['glucose'].max()
    }
    
    logging.info(f"✓ Generated {len(patient_data)} data points (5-min intervals)")
    logging.info(f"✓ Found {insulin_interventions} insulin interventions")
    logging.info(f"✓ Glucose stats: μ={glucose_stats['mean']:.1f}, σ={glucose_stats['std']:.1f}, "
                f"range=[{glucose_stats['min']:.1f}, {glucose_stats['max']:.1f}] mg/dL")
    
    return patient_data


def train_model(patient_data, model_name, model_folder, hyperparams=None, max_sequence_length=60):
    """Train the CRN model on diabetes data."""
    logging.info("Training CRN model for diabetes...")
    
    try:
        model, test_data = train_diabetes_crn(
            patient_data=patient_data,
            model_name=model_name,
            model_folder=model_folder,
            max_sequence_length=max_sequence_length,
            hyperparams=hyperparams
        )
        
        logging.info("✓ Model training completed successfully")
        return model, test_data
        
    except Exception as e:
        logging.error(f"✗ Model training failed: {e}")
        raise


def evaluate_model(model, test_data, model_name):
    """Evaluate the trained model."""
    logging.info("Evaluating model performance...")
    
    try:
        results = evaluate_diabetes_model(model, test_data)
        
        logging.info(f"✓ Test MSE: {results['test_mse']:.6f}")
        logging.info(f"✓ MSE by timestep shape: {results['mse_by_timestep'].shape}")
        logging.info(f"✓ Balancing representations shape: {results['balancing_representations'].shape}")
        
        return results
        
    except Exception as e:
        logging.error(f"✗ Model evaluation failed: {e}")
        raise


def run_hyperparameter_search(patient_data, model_name, model_folder, max_sequence_length=60):
    """Run hyperparameter search for optimal model configuration."""
    logging.info("Running hyperparameter search...")
    
    # Define search space for diabetes-specific optimization
    search_space = {
        'rnn_hidden_units': [16, 24, 32],
        'br_size': [8, 12, 16],
        'fc_hidden_units': [24, 36, 48],
        'learning_rate': [0.001, 0.01],
        'batch_size': [32, 64],
        'rnn_keep_prob': [0.8, 0.9]
    }
    
    try:
        search_results = hyperparameter_search(
            patient_data=patient_data,
            model_name=model_name,
            model_folder=model_folder,
            search_space=search_space,
            max_sequence_length=max_sequence_length
        )
        
        logging.info("✓ Hyperparameter search completed")
        return search_results
        
    except Exception as e:
        logging.error(f"✗ Hyperparameter search failed: {e}")
        raise


def demonstrate_counterfactual_analysis(patient_data):
    """Demonstrate counterfactual analysis capabilities."""
    logging.info("Demonstrating counterfactual analysis...")
    
    try:
        analyzer = DiabetesAnalyzer(seed=42)
        interventions = analyzer.counterfactual_model.list_interventions(patient_data)
        
        if len(interventions) == 0:
            logging.warning("No interventions found for counterfactual analysis")
            return
        
        # Select first intervention for demonstration
        intervention = interventions[0]
        logging.info(f"Analyzing intervention: {intervention['timestamp']} - {intervention['dose']:.1f} units")
        
        # Dose counterfactual: What if 20% more insulin?
        dose_result = analyzer.analyze_intervention(
            patient_data,
            intervention_id=intervention['id'],
            analysis_type='dose',
            dose_factor=1.2,
            before_minutes=120,
            after_minutes=180
        )
        
        # Timing counterfactual: What if 30 minutes earlier?
        timing_result = analyzer.analyze_intervention(
            patient_data,
            intervention_id=intervention['id'],
            analysis_type='timing',
            timing_shift_minutes=-30,
            before_minutes=120,
            after_minutes=180
        )
        
        logging.info("✓ Counterfactual analysis demonstration completed")
        logging.info(f"  - Dose analysis: {dose_result.shape}")
        logging.info(f"  - Timing analysis: {timing_result.shape}")
        
    except Exception as e:
        logging.error(f"✗ Counterfactual analysis failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Train CRN model for diabetes management')
    parser.add_argument('--days', type=int, default=14, 
                       help='Days of patient data to generate (default: 14)')
    parser.add_argument('--model_name', default='diabetes_crn', 
                       help='Model name for saving (default: diabetes_crn)')
    parser.add_argument('--model_folder', default='results/diabetes_models', 
                       help='Folder for saving models (default: results/diabetes_models)')
    parser.add_argument('--max_sequence_length', type=int, default=60,
                       help='Maximum sequence length for CRN (default: 60)')
    parser.add_argument('--hyperparameter_search', action='store_true',
                       help='Run hyperparameter search before training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    try:
        logging.info("=== CRN Diabetes Training Pipeline ===")
        
        # Step 1: Generate diabetes training data
        patient_data = generate_diabetes_training_data(n_days=args.days, seed=args.seed)
        
        # Step 2: Demonstrate counterfactual analysis
        demonstrate_counterfactual_analysis(patient_data)
        
        # Step 3: Hyperparameter search (optional)
        hyperparams = None
        if args.hyperparameter_search:
            search_results = run_hyperparameter_search(
                patient_data, args.model_name, args.model_folder, args.max_sequence_length
            )
            hyperparams = search_results['best_hyperparams']
        else:
            hyperparams = get_default_hyperparams()
            hyperparams['num_epochs'] = args.epochs
        
        logging.info(f"Using hyperparameters: {hyperparams}")
        
        # Step 4: Train model
        model, test_data = train_model(
            patient_data, args.model_name, args.model_folder, 
            hyperparams, args.max_sequence_length
        )
        
        # Step 5: Evaluate model
        results = evaluate_model(model, test_data, args.model_name)
        
        logging.info("=== Training Pipeline Completed Successfully ===")
        logging.info(f"Model saved to: {args.model_folder}/{args.model_name}_final.ckpt")
        logging.info(f"Final test MSE: {results['test_mse']:.6f}")
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()