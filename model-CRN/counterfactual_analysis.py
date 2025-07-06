#!/usr/bin/env python
"""
Counterfactual analysis tool for glucose predictions using trained CRN model.

This script allows you to:
1. Select a time period from the training data
2. Modify insulin timing/dosage scenarios
3. Generate counterfactual glucose predictions
4. Visualize the results
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils.glucose_simulation import get_glucose_sim_data
from utils.glucose_evaluation_utils import get_processed_data, load_trained_model
import os

def load_glucose_data_raw(data_path):
    """Load raw glucose data for counterfactual analysis"""
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def create_counterfactual_scenarios(baseline_insulin, scenario_type="delayed", **kwargs):
    """
    Create different insulin scenarios for counterfactual analysis.
    
    Args:
        baseline_insulin: Original insulin sequence
        scenario_type: Type of counterfactual scenario
        **kwargs: Additional parameters for specific scenarios
    
    Returns:
        dict: Different insulin scenarios
    """
    scenarios = {"baseline": baseline_insulin.copy()}
    
    if scenario_type == "delayed":
        # Delay insulin by specified timesteps
        delay_steps = kwargs.get("delay_steps", 2)
        delayed_insulin = np.zeros_like(baseline_insulin)
        delayed_insulin[delay_steps:] = baseline_insulin[:-delay_steps]
        scenarios["delayed"] = delayed_insulin
        
    elif scenario_type == "increased_dose":
        # Increase insulin dose by specified factor
        dose_factor = kwargs.get("dose_factor", 1.5)
        increased_insulin = baseline_insulin * dose_factor
        scenarios["increased_dose"] = increased_insulin
        
    elif scenario_type == "reduced_dose":
        # Reduce insulin dose by specified factor
        dose_factor = kwargs.get("dose_factor", 0.7)
        reduced_insulin = baseline_insulin * dose_factor
        scenarios["reduced_dose"] = reduced_insulin
        
    elif scenario_type == "no_insulin":
        # Remove all insulin
        no_insulin = np.zeros_like(baseline_insulin)
        scenarios["no_insulin"] = no_insulin
        
    elif scenario_type == "early":
        # Give insulin earlier
        early_steps = kwargs.get("early_steps", 2)
        early_insulin = np.zeros_like(baseline_insulin)
        early_insulin[:-early_steps] = baseline_insulin[early_steps:]
        scenarios["early"] = early_insulin
        
    elif scenario_type == "all":
        # Generate all scenario types
        scenarios.update(create_counterfactual_scenarios(baseline_insulin, "delayed", delay_steps=2))
        scenarios.update(create_counterfactual_scenarios(baseline_insulin, "increased_dose", dose_factor=1.5))
        scenarios.update(create_counterfactual_scenarios(baseline_insulin, "reduced_dose", dose_factor=0.7))
        scenarios.update(create_counterfactual_scenarios(baseline_insulin, "no_insulin"))
        scenarios.update(create_counterfactual_scenarios(baseline_insulin, "early", early_steps=2))
    
    return scenarios

def predict_counterfactual(model, processed_data, patient_idx, new_insulin_sequence, scaling_params):
    """
    Generate counterfactual prediction using the trained CRN model.
    
    Args:
        model: Trained CRN model
        processed_data: Processed data dictionary
        patient_idx: Index of the patient/sequence to analyze
        new_insulin_sequence: New insulin treatment sequence
        scaling_params: Scaling parameters for normalization
        
    Returns:
        np.array: Predicted glucose sequence
    """
    mean, std = scaling_params
    
    # Extract baseline data for the selected patient
    baseline_covariates = processed_data['current_covariates'][patient_idx:patient_idx+1].copy()
    baseline_treatments = processed_data['current_treatments'][patient_idx:patient_idx+1].copy()
    baseline_active_entries = processed_data['active_entries'][patient_idx:patient_idx+1].copy()
    
    # Modify the treatment sequence
    # Convert insulin values to binary application (following our processing logic)
    insulin_application = (new_insulin_sequence > 0).astype(float)
    
    # Update treatments - remember we use 2-class system: [no_insulin, insulin]
    modified_treatments = baseline_treatments.copy()
    sequence_length = modified_treatments.shape[1]
    
    for timestep in range(min(len(insulin_application), sequence_length)):
        if insulin_application[timestep] == 0:
            modified_treatments[0, timestep] = [1, 0]  # no insulin
        else:
            modified_treatments[0, timestep] = [0, 1]  # insulin given
    
    # Create modified data dictionary
    modified_data = {
        'current_covariates': baseline_covariates,
        'current_treatments': modified_treatments,
        'previous_treatments': baseline_treatments[:, :-1, :],  # Shifted treatments
        'active_entries': baseline_active_entries,
        'outputs': processed_data['outputs'][patient_idx:patient_idx+1],  # Not used for prediction
        'unscaled_outputs': processed_data['unscaled_outputs'][patient_idx:patient_idx+1],
        'sequence_lengths': processed_data['sequence_lengths'][patient_idx:patient_idx+1],
        'output_means': processed_data['output_means'],
        'output_stds': processed_data['output_stds']
    }
    
    # Generate prediction
    predicted_glucose_norm, _ = model.evaluate_predictions(modified_data)
    
    # Denormalize the prediction
    predicted_glucose = predicted_glucose_norm * std['glucose'] + mean['glucose']
    
    return predicted_glucose

def visualize_counterfactuals(time_points, scenarios_results, baseline_glucose, title="Counterfactual Glucose Predictions"):
    """
    Visualize counterfactual analysis results.
    
    Args:
        time_points: Time points for x-axis
        scenarios_results: Dict of scenario name -> predicted glucose values
        baseline_glucose: Original glucose values for comparison
        title: Plot title
    """
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Glucose predictions
    plt.subplot(2, 1, 1)
    
    # Plot baseline glucose (actual)
    plt.plot(time_points, baseline_glucose, 'k-', linewidth=2, label='Actual Glucose', alpha=0.8)
    
    # Plot counterfactual scenarios
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for i, (scenario_name, glucose_pred) in enumerate(scenarios_results.items()):
        if scenario_name != 'baseline':
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            # Handle case where prediction might be a single value
            pred_values = glucose_pred.flatten()
            if len(pred_values) == 1:
                # If single prediction, repeat it across all time points
                pred_values = np.repeat(pred_values[0], len(time_points))
            
            plt.plot(time_points, pred_values, color=color, linestyle=linestyle, 
                    linewidth=2, label=f'Predicted: {scenario_name}')
    
    plt.xlabel('Time Points')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.title(f'{title} - Glucose Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Glucose differences from baseline
    plt.subplot(2, 1, 2)
    
    baseline_pred = scenarios_results.get('baseline', baseline_glucose).flatten()
    if len(baseline_pred) == 1:
        baseline_pred = np.repeat(baseline_pred[0], len(time_points))
    
    for i, (scenario_name, glucose_pred) in enumerate(scenarios_results.items()):
        if scenario_name != 'baseline':
            pred_values = glucose_pred.flatten()
            if len(pred_values) == 1:
                pred_values = np.repeat(pred_values[0], len(time_points))
            
            difference = pred_values - baseline_pred
            color = colors[i % len(colors)]
            plt.plot(time_points, difference, color=color, linewidth=2, label=f'Δ {scenario_name}')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Time Points')
    plt.ylabel('Glucose Difference (mg/dL)')
    plt.title('Difference from Baseline Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_counterfactual_analysis(data_path, model_name, patient_idx=0, sequence_start=100, 
                               scenario_type="all", **scenario_kwargs):
    """
    Main function to run counterfactual analysis.
    
    Args:
        data_path: Path to glucose dataset
        model_name: Name of trained model
        patient_idx: Index of patient/sequence to analyze
        sequence_start: Starting point in the time series
        scenario_type: Type of counterfactual scenario
        **scenario_kwargs: Additional scenario parameters
    """
    
    # Load data and model
    print("Loading glucose data and trained model...")
    
    # Load processed data (same as training)
    pickle_map = get_glucose_sim_data(data_path, sequence_length=10, prediction_horizon=5)
    test_data = pickle_map['test_data']
    scaling_data = pickle_map['scaling_data']
    
    # Process test data
    test_processed = get_processed_data(test_data, scaling_data)
    
    # Load trained model
    models_dir = 'results/crn_models'
    encoder_hyperparams_file = f'results/encoder_{model_name}_best_hyperparams.txt'
    encoder_model_name = f'encoder_{model_name}'  # Don't add _final, it's added automatically
    
    if not os.path.exists(encoder_hyperparams_file):
        print(f"Error: Model hyperparameters not found at {encoder_hyperparams_file}")
        print("Please train the model first using test_crn_glucose.py")
        return
    
    model = load_trained_model(test_processed, encoder_hyperparams_file, encoder_model_name, models_dir)
    
    # Load raw data for insulin sequences
    raw_df = load_glucose_data_raw(data_path)
    
    # Select a specific time period
    if patient_idx >= len(test_processed['current_treatments']):
        patient_idx = 0
        print(f"Patient index too high, using patient_idx=0")
    
    # Get baseline insulin sequence for the selected patient
    baseline_insulin = test_data['current_treatments'][patient_idx, :, 0]
    baseline_glucose = test_data['outputs'][patient_idx, :, 0]
    
    print(f"\nAnalyzing patient/sequence {patient_idx}")
    print(f"Baseline insulin doses: {baseline_insulin}")
    print(f"Baseline glucose levels: {baseline_glucose}")
    
    # Create counterfactual scenarios
    print(f"\nGenerating {scenario_type} counterfactual scenarios...")
    scenarios = create_counterfactual_scenarios(baseline_insulin, scenario_type, **scenario_kwargs)
    
    # Generate predictions for each scenario
    scenarios_results = {}
    mean, std = scaling_data
    
    for scenario_name, insulin_sequence in scenarios.items():
        print(f"Predicting scenario: {scenario_name}")
        
        if scenario_name == "baseline":
            # For baseline, use the actual prediction from the model
            baseline_data = {
                'current_covariates': test_processed['current_covariates'][patient_idx:patient_idx+1],
                'current_treatments': test_processed['current_treatments'][patient_idx:patient_idx+1],
                'previous_treatments': test_processed['previous_treatments'][patient_idx:patient_idx+1],
                'active_entries': test_processed['active_entries'][patient_idx:patient_idx+1],
                'outputs': test_processed['outputs'][patient_idx:patient_idx+1],
                'unscaled_outputs': test_processed['unscaled_outputs'][patient_idx:patient_idx+1],
                'sequence_lengths': test_processed['sequence_lengths'][patient_idx:patient_idx+1],
                'output_means': test_processed['output_means'],
                'output_stds': test_processed['output_stds']
            }
            pred_glucose_norm, _ = model.evaluate_predictions(baseline_data)
            pred_glucose = pred_glucose_norm * std['glucose'] + mean['glucose']
        else:
            # For counterfactuals, modify the insulin sequence
            pred_glucose = predict_counterfactual(model, test_processed, patient_idx, 
                                                insulin_sequence, scaling_data)
        
        scenarios_results[scenario_name] = pred_glucose
        
        print(f"  Insulin: {insulin_sequence}")
        print(f"  Predicted glucose: {pred_glucose.flatten()}")
        print()
    
    # Create time points for visualization
    time_points = np.arange(len(baseline_glucose))
    
    # Visualize results
    visualize_counterfactuals(time_points, scenarios_results, baseline_glucose, 
                            f"Counterfactual Analysis - Patient {patient_idx}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("COUNTERFACTUAL ANALYSIS SUMMARY")
    print("="*60)
    
    baseline_pred = scenarios_results['baseline'].flatten()
    if len(baseline_pred) == 1:
        baseline_pred = np.repeat(baseline_pred[0], len(time_points))
    
    for scenario_name, glucose_pred in scenarios_results.items():
        if scenario_name != 'baseline':
            glucose_flat = glucose_pred.flatten()
            if len(glucose_flat) == 1:
                glucose_flat = np.repeat(glucose_flat[0], len(time_points))
                
            mean_diff = np.mean(glucose_flat - baseline_pred)
            max_diff = np.max(np.abs(glucose_flat - baseline_pred))
            
            print(f"\n{scenario_name.upper()}:")
            print(f"  Mean glucose difference: {mean_diff:.2f} mg/dL")
            print(f"  Max absolute difference: {max_diff:.2f} mg/dL")
            print(f"  Predicted glucose level: {glucose_flat[0]:.2f} mg/dL")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Counterfactual Analysis for Glucose Predictions')
    parser.add_argument('--data_path', default='../Data/ml_dataset.csv', 
                       help='Path to glucose dataset')
    parser.add_argument('--model_name', default='crn_glucose_basic', 
                       help='Name of trained model (without encoder_ prefix)')
    parser.add_argument('--patient_idx', type=int, default=0, 
                       help='Patient/sequence index to analyze')
    parser.add_argument('--scenario_type', default='all', 
                       choices=['delayed', 'increased_dose', 'reduced_dose', 'no_insulin', 'early', 'all'],
                       help='Type of counterfactual scenario')
    parser.add_argument('--delay_steps', type=int, default=2, 
                       help='Steps to delay insulin (for delayed scenario)')
    parser.add_argument('--dose_factor', type=float, default=1.5, 
                       help='Factor to multiply dose (for dose scenarios)')
    parser.add_argument('--early_steps', type=int, default=2, 
                       help='Steps to advance insulin (for early scenario)')
    
    args = parser.parse_args()
    
    scenario_kwargs = {
        'delay_steps': args.delay_steps,
        'dose_factor': args.dose_factor,
        'early_steps': args.early_steps
    }
    
    run_counterfactual_analysis(
        data_path=args.data_path,
        model_name=args.model_name,
        patient_idx=args.patient_idx,
        scenario_type=args.scenario_type,
        **scenario_kwargs
    )