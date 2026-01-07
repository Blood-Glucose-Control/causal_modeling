#!/usr/bin/env python
# Counterfactual evaluation script for trained glucose model

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

# Import from our modules
from model.seq2seq import Encoder, AttentionDecoder, Seq2Seq
from counterfactuals.counterfactual_testing import (
    CounterfactualScenarioGenerator,
    CounterfactualEvaluator,
    generate_glucose_counterfactuals,
    plot_glucose_counterfactuals,
    save_test_scenarios
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluate_counterfactuals.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("evaluate_counterfactuals")

def load_model(checkpoint_path, device='cuda'):
    """Load a trained T4 model"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['args']
    
    # Initialize model components
    encoder = Encoder(
        input_dim=model_args.vital_num,
        output_dim=1,
        x_static_size=model_args.demo_dim,
        emb_dim=model_args.emb_dim,
        hid_dim=model_args.hidden_dim,
        n_layers=model_args.layer_num,
        dropout=model_args.dropout,
        device=device
    )
    
    decoder = AttentionDecoder(
        output_dim=1,
        x_static_size=model_args.demo_dim,
        emb_dim=model_args.emb_dim,
        hid_dim=model_args.hidden_dim,
        n_layers=model_args.layer_num,
        dropout=model_args.dropout
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded successfully (trained for {checkpoint['epoch']+1} epochs)")
    return model

def evaluate_dose_counterfactuals(model, device='cuda', seed=42, n_scenarios=10):
    """Evaluate model on dose-based counterfactual scenarios"""
    logger.info("Generating dose-based counterfactual scenarios")
    
    # Generate test scenarios
    generator = CounterfactualScenarioGenerator(seed=seed)
    base_data = generator.generate_base_data(days=7)
    
    # Create output directory
    output_dir = "counterfactual_results/dose"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample intervention windows
    windows = generator.sample_intervention_windows(
        base_data, 
        n_samples=n_scenarios, 
        hours_before=2,
        hours_after=5
    )
    
    results = []
    
    # Different dose multipliers to test
    dose_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    # Process each intervention window
    for i, (start_time, intervention_time, end_time) in enumerate(windows):
        logger.info(f"Processing scenario {i+1}/{len(windows)} at {intervention_time}")
        
        # Extract window data
        window_data = base_data.loc[start_time:end_time].copy()
        
        # Original insulin dose at intervention
        original_dose = window_data.loc[intervention_time, 'insulin']
        
        if original_dose <= 0:
            logger.info(f"Skipping scenario {i+1} - no insulin at intervention time")
            continue
        
        # Test different dose multipliers
        scenario_results = []
        
        for multiplier in dose_multipliers:
            logger.info(f"Testing dose multiplier: {multiplier}")
            
            # Generate counterfactual predictions
            cf_result = generate_glucose_counterfactuals(
                model=model,
                data_window=window_data,
                intervention_time=intervention_time,
                dose_multiplier=multiplier,
                timing_shift=None,
                device=device
            )
            
            # Store results
            scenario_results.append(cf_result)
            
            # Create visualization for this counterfactual
            fig = plot_glucose_counterfactuals(window_data, cf_result)
            plt.savefig(f"{output_dir}/scenario_{i+1}_dose_{multiplier:.2f}.png")
            plt.close()
        
        # Create combined visualization showing all dose multipliers
        create_combined_dose_plot(window_data, intervention_time, scenario_results, dose_multipliers, 
                                 f"{output_dir}/scenario_{i+1}_combined.png")
        
        results.append({
            'scenario_id': i+1,
            'intervention_time': intervention_time,
            'original_dose': original_dose,
            'results': scenario_results
        })
    
    # Create summary of results
    create_dose_summary(results, f"{output_dir}/summary.png")
    
    logger.info(f"Dose counterfactual evaluation completed. Results saved to {output_dir}")
    return results

def evaluate_timing_counterfactuals(model, device='cuda', seed=42, n_scenarios=10):
    """Evaluate model on timing-based counterfactual scenarios"""
    logger.info("Generating timing-based counterfactual scenarios")
    
    # Generate test scenarios
    generator = CounterfactualScenarioGenerator(seed=seed)
    base_data = generator.generate_base_data(days=7)
    
    # Create output directory
    output_dir = "counterfactual_results/timing"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample intervention windows
    windows = generator.sample_intervention_windows(
        base_data, 
        n_samples=n_scenarios, 
        hours_before=2,
        hours_after=5
    )
    
    results = []
    
    # Different timing shifts to test (in minutes)
    timing_shifts = [-30, -15, 0, 15, 30]
    
    # Process each intervention window
    for i, (start_time, intervention_time, end_time) in enumerate(windows):
        logger.info(f"Processing scenario {i+1}/{len(windows)} at {intervention_time}")
        
        # Extract window data
        window_data = base_data.loc[start_time:end_time].copy()
        
        # Original insulin dose at intervention
        original_dose = window_data.loc[intervention_time, 'insulin']
        
        if original_dose <= 0:
            logger.info(f"Skipping scenario {i+1} - no insulin at intervention time")
            continue
        
        # Test different timing shifts
        scenario_results = []
        
        for shift in timing_shifts:
            logger.info(f"Testing timing shift: {shift} minutes")
            
            # Generate counterfactual predictions
            cf_result = generate_glucose_counterfactuals(
                model=model,
                data_window=window_data,
                intervention_time=intervention_time,
                dose_multiplier=None,
                timing_shift=shift,
                device=device
            )
            
            # Store results
            scenario_results.append(cf_result)
            
            # Create visualization for this counterfactual
            fig = plot_glucose_counterfactuals(window_data, cf_result)
            plt.savefig(f"{output_dir}/scenario_{i+1}_timing_{shift:+d}.png")
            plt.close()
        
        # Create combined visualization showing all timing shifts
        create_combined_timing_plot(window_data, intervention_time, scenario_results, timing_shifts, 
                                  f"{output_dir}/scenario_{i+1}_combined.png")
        
        results.append({
            'scenario_id': i+1,
            'intervention_time': intervention_time,
            'original_dose': original_dose,
            'results': scenario_results
        })
    
    # Create summary of results
    create_timing_summary(results, f"{output_dir}/summary.png")
    
    logger.info(f"Timing counterfactual evaluation completed. Results saved to {output_dir}")
    return results

def create_combined_dose_plot(data_window, intervention_time, scenario_results, dose_multipliers, output_file):
    """Create a combined plot showing glucose curves for different dose multipliers"""
    plt.figure(figsize=(14, 8))
    
    # Plot historical data
    historical_data = data_window.loc[:intervention_time]
    plt.plot(historical_data.index, historical_data['glucose'], 'b-', label='Historical', linewidth=2)
    
    # Plot counterfactuals with different colors
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    
    for i, (result, multiplier) in enumerate(zip(scenario_results, dose_multipliers)):
        pred_df = result['counterfactual_predictions']
        label = 'Factual (No Change)' if multiplier == 1.0 else f'Dose × {multiplier}'
        linestyle = '--' if multiplier != 1.0 else '-.'
        plt.plot(pred_df.index, pred_df['glucose'], color=colors[i], linestyle=linestyle, 
                 label=label, linewidth=2)
    
    # Mark intervention point
    plt.axvline(x=intervention_time, color='black', linestyle='--')
    plt.annotate('Intervention', 
               xy=(intervention_time, data_window.loc[intervention_time, 'glucose']),
               xytext=(15, 15),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->'),
               fontsize=10)
    
    # Format plot
    plt.title('Effect of Insulin Dose on Blood Glucose', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Blood Glucose (mg/dL)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Add target range
    plt.axhspan(70, 180, color='green', alpha=0.1, label='Target Range')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_combined_timing_plot(data_window, intervention_time, scenario_results, timing_shifts, output_file):
    """Create a combined plot showing glucose curves for different timing shifts"""
    plt.figure(figsize=(14, 8))
    
    # Plot historical data
    historical_data = data_window.loc[:intervention_time]
    plt.plot(historical_data.index, historical_data['glucose'], 'b-', label='Historical', linewidth=2)
    
    # Plot counterfactuals with different colors
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    
    for i, (result, shift) in enumerate(zip(scenario_results, timing_shifts)):
        pred_df = result['counterfactual_predictions']
        if shift == 0:
            label = 'Factual (No Change)'
            linestyle = '-.'
        else:
            label = f'Earlier {abs(shift)}m' if shift < 0 else f'Later {shift}m'
            linestyle = '--'
        plt.plot(pred_df.index, pred_df['glucose'], color=colors[i], linestyle=linestyle, 
                 label=label, linewidth=2)
    
    # Mark intervention point
    plt.axvline(x=intervention_time, color='black', linestyle='--')
    plt.annotate('Intervention', 
               xy=(intervention_time, data_window.loc[intervention_time, 'glucose']),
               xytext=(15, 15),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->'),
               fontsize=10)
    
    # Format plot
    plt.title('Effect of Insulin Timing on Blood Glucose', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Blood Glucose (mg/dL)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Add target range
    plt.axhspan(70, 180, color='green', alpha=0.1, label='Target Range')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_dose_summary(results, output_file):
    """Create a summary plot showing the effect of different dose multipliers across scenarios"""
    plt.figure(figsize=(10, 6))
    
    dose_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    # Collect effect sizes for each multiplier
    effect_data = {multiplier: [] for multiplier in dose_multipliers}
    
    for scenario in results:
        for i, multiplier in enumerate(dose_multipliers):
            # Get the mean effect compared to factual
            cf_result = scenario['results'][i]
            factual_pred = scenario['results'][2]['factual_predictions']['glucose'].values  # 1.0 is index 2
            counterfactual_pred = cf_result['counterfactual_predictions']['glucose'].values
            
            # Calculate mean effect
            effect = np.mean(counterfactual_pred - factual_pred)
            effect_data[multiplier].append(effect)
    
    # Calculate statistics
    means = [np.mean(effect_data[m]) for m in dose_multipliers]
    stds = [np.std(effect_data[m]) for m in dose_multipliers]
    
    # Create bar chart
    plt.bar(range(len(dose_multipliers)), means, yerr=stds, capsize=10)
    plt.xticks(range(len(dose_multipliers)), [str(m) for m in dose_multipliers])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Format plot
    plt.title('Mean Effect of Insulin Dose on Blood Glucose', fontsize=14)
    plt.xlabel('Dose Multiplier', fontsize=12)
    plt.ylabel('Mean Glucose Change (mg/dL)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_timing_summary(results, output_file):
    """Create a summary plot showing the effect of different timing shifts across scenarios"""
    plt.figure(figsize=(10, 6))
    
    timing_shifts = [-30, -15, 0, 15, 30]
    
    # Collect effect sizes for each shift
    effect_data = {shift: [] for shift in timing_shifts}
    
    for scenario in results:
        for i, shift in enumerate(timing_shifts):
            # Get the mean effect compared to factual
            cf_result = scenario['results'][i]
            factual_pred = scenario['results'][2]['factual_predictions']['glucose'].values  # 0 is index 2
            counterfactual_pred = cf_result['counterfactual_predictions']['glucose'].values
            
            # Calculate mean effect
            effect = np.mean(counterfactual_pred - factual_pred)
            effect_data[shift].append(effect)
    
    # Calculate statistics
    means = [np.mean(effect_data[s]) for s in timing_shifts]
    stds = [np.std(effect_data[s]) for s in timing_shifts]
    
    # Create bar chart
    plt.bar(range(len(timing_shifts)), means, yerr=stds, capsize=10)
    plt.xticks(range(len(timing_shifts)), [f"{s:+d} min" for s in timing_shifts])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Format plot
    plt.title('Mean Effect of Insulin Timing on Blood Glucose', fontsize=14)
    plt.xlabel('Timing Shift', fontsize=12)
    plt.ylabel('Mean Glucose Change (mg/dL)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate T4 model with counterfactuals')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='both', choices=['dose', 'timing', 'both'], 
                        help='Type of counterfactuals to evaluate')
    parser.add_argument('--scenarios', type=int, default=10, help='Number of scenarios to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run model on')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("counterfactual_results", exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device=args.device)
    
    # Run evaluations
    if args.mode in ['dose', 'both']:
        evaluate_dose_counterfactuals(
            model=model,
            device=args.device,
            seed=args.seed,
            n_scenarios=args.scenarios
        )
        
    if args.mode in ['timing', 'both']:
        evaluate_timing_counterfactuals(
            model=model,
            device=args.device,
            seed=args.seed,
            n_scenarios=args.scenarios
        )
    
    logger.info("Counterfactual evaluation completed")

if __name__ == "__main__":
    main() 