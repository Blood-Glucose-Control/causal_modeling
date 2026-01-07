"""
GLUCOSE COMPARISON: Test causal transformer on glucose data

Runs model on glucose counterfactual data:
- Uses synthetic glucose simulator from src/data/generator.py
- Trains on multi-step glucose prediction with insulin/carb interventions
- Evaluates counterfactual reasoning capabilities
"""

import os
import argparse
import json
import numpy as np
import torch
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.data.generator import DiabetesAnalyzer, CounterfactualAnalyzer
from src.data.dataset import GlucoseDataset
from src.models.causal_transformer import GlucoseTransformerCRN
from src.training.trainer import CT_Trainer
from src.evaluation.glucose_evaluator import GlucoseCounterfactualEvaluator


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device():
    """Get best available device (MPS for Mac, CUDA for NVIDIA, CPU fallback)"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def generate_glucose_data(num_days, train_ratio, val_ratio, seed=42):
    """
    Generate glucose data and split into train/val/test

    Returns:
        train_df, val_df, test_df, scaler
    """
    print(f"Generating {num_days} days of synthetic glucose data (seed={seed})...")
    analyzer = DiabetesAnalyzer(seed=seed)
    full_df = analyzer.generate_patient_data(n_days=num_days)

    # Calculate split indices based on time
    n = len(full_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = full_df.iloc[:train_end]
    val_df = full_df.iloc[train_end:val_end]
    test_df = full_df.iloc[val_end:]

    print(f"Splits: Train={len(train_df)} ({len(train_df)/(12*24):.1f} days), "
          f"Val={len(val_df)} ({len(val_df)/(12*24):.1f} days), "
          f"Test={len(test_df)} ({len(test_df)/(12*24):.1f} days)")

    # Fit Scaler on Train
    print("Fitting StandardScaler on training data...")
    train_features = GlucoseDataset.extract_features(train_df)
    scaler = StandardScaler()
    scaler.fit(train_features)

    return train_df, val_df, test_df, scaler


def train_and_evaluate_model(model, config, seed, run_dir, test_df, scaler, full_patient_data):
    """Train and evaluate glucose model"""
    print(f"\n{'='*80}")
    print(f"GLUCOSE MODEL | SEED: {seed}")
    print(f"{'='*80}\n")

    # Generate data with this seed
    train_df, val_df, _, train_scaler = generate_glucose_data(
        num_days=config['num_days'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        seed=seed
    )

    # Create datasets
    train_ds = GlucoseDataset(
        train_df, train_scaler,
        history_window=config['history_window'],
        prediction_horizon=config['prediction_horizon'],
        stride=config.get('stride', 6)
    )
    val_ds = GlucoseDataset(
        val_df, train_scaler,
        history_window=config['history_window'],
        prediction_horizon=config['prediction_horizon'],
        stride=config.get('stride', 6)
    )
    test_ds = GlucoseDataset(
        test_df, scaler,
        history_window=config['history_window'],
        prediction_horizon=config['prediction_horizon'],
        stride=config.get('stride', 6)
    )

    print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}, Test windows: {len(test_ds)}")

    # Train
    trainer_config = config.copy()
    trainer_config['save_dir'] = run_dir
    trainer = CT_Trainer(model, train_ds, val_ds, trainer_config)
    trainer.train()

    # Load best model
    best_model_path = os.path.join(run_dir, 'best_model.pt')
    device = get_device()
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    total_loss = 0.0
    total_samples = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in test_loader:
            encoder_inputs = batch['encoder_inputs'].to(device)
            future_treatments = batch['future_treatments'].to(device)
            future_outcomes = batch['future_outcomes'].to(device)

            # Forward pass
            outputs = model(encoder_inputs, future_treatments)
            pred_outcomes = outputs['pred_outcomes']

            # MSE loss
            loss = torch.nn.functional.mse_loss(pred_outcomes, future_outcomes)
            total_loss += loss.item() * encoder_inputs.size(0)
            total_samples += encoder_inputs.size(0)

            predictions.append(pred_outcomes.cpu().numpy())
            actuals.append(future_outcomes.cpu().numpy())

    test_mse = total_loss / total_samples
    test_rmse = np.sqrt(test_mse)

    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Compute per-timestep RMSE
    timestep_rmse = {}
    for t in range(predictions.shape[1]):
        t_rmse = np.sqrt(np.mean((predictions[:, t, 0] - actuals[:, t, 0]) ** 2))
        timestep_rmse[t+1] = float(t_rmse)

    print(f"\nFactual Test RMSE: {test_rmse:.4f}")
    print(f"Per-timestep RMSE (first 10):")
    for t in range(min(10, len(timestep_rmse))):
        print(f"  τ={t+1}: {timestep_rmse[t+1]:.4f}")

    # Counterfactual Evaluation
    print("\n" + "="*80)
    print("COUNTERFACTUAL EVALUATION")
    print("="*80)

    evaluator = GlucoseCounterfactualEvaluator(model, train_scaler, device=device)

    # Evaluate dose counterfactuals on full patient data
    dose_results = evaluator.evaluate_dose_counterfactuals(
        patient_data=full_patient_data,
        dose_factors=[0.8, 1.0, 1.2, 1.5],
        num_interventions=config.get('num_cf_interventions', 20),
        history_window=config['history_window'],
        prediction_horizon=config['prediction_horizon']
    )

    # Evaluate timing counterfactuals
    timing_results = evaluator.evaluate_timing_counterfactuals(
        patient_data=full_patient_data,
        timing_shifts=[-30, -15, 0, 15, 30],
        num_interventions=config.get('num_cf_interventions', 20),
        history_window=config['history_window'],
        prediction_horizon=config['prediction_horizon']
    )

    # Print counterfactual results
    evaluator.print_results(dose_results=dose_results, timing_results=timing_results)

    results = {
        'model': 'Glucose Transformer CRN',
        'seed': seed,
        'test_rmse': float(test_rmse),
        'timestep_rmse': timestep_rmse,
        'dose_counterfactuals': dose_results,
        'timing_counterfactuals': timing_results
    }

    return results


def aggregate_results(all_results, output_dir):
    """Aggregate results across seeds"""
    output_dir = Path(output_dir)

    # Compute mean and std across seeds
    rmse_values = [r['test_rmse'] for r in all_results]
    mean_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values, ddof=1) if len(rmse_values) > 1 else 0.0

    # Per-timestep aggregation
    all_timesteps = all_results[0]['timestep_rmse'].keys()
    timestep_stats = {}
    for t in all_timesteps:
        t_values = [r['timestep_rmse'][t] for r in all_results]
        timestep_stats[t] = {
            'mean': float(np.mean(t_values)),
            'std': float(np.std(t_values, ddof=1)) if len(t_values) > 1 else 0.0
        }

    # Aggregate counterfactual results
    # Dose counterfactuals
    dose_factors = list(all_results[0]['dose_counterfactuals'].keys())
    dose_cf_summary = {}
    for factor in dose_factors:
        rmse_means = [r['dose_counterfactuals'][factor]['rmse_mean']
                      for r in all_results if not np.isnan(r['dose_counterfactuals'][factor]['rmse_mean'])]
        mae_means = [r['dose_counterfactuals'][factor]['mae_mean']
                     for r in all_results if not np.isnan(r['dose_counterfactuals'][factor]['mae_mean'])]

        if len(rmse_means) > 0:
            dose_cf_summary[factor] = {
                'rmse_mean': float(np.mean(rmse_means)),
                'rmse_std': float(np.std(rmse_means, ddof=1)) if len(rmse_means) > 1 else 0.0,
                'mae_mean': float(np.mean(mae_means)),
                'mae_std': float(np.std(mae_means, ddof=1)) if len(mae_means) > 1 else 0.0,
                'n_seeds': len(rmse_means)
            }
        else:
            dose_cf_summary[factor] = {
                'rmse_mean': np.nan,
                'rmse_std': np.nan,
                'mae_mean': np.nan,
                'mae_std': np.nan,
                'n_seeds': 0
            }

    # Timing counterfactuals
    timing_shifts = list(all_results[0]['timing_counterfactuals'].keys())
    timing_cf_summary = {}
    for shift in timing_shifts:
        rmse_means = [r['timing_counterfactuals'][shift]['rmse_mean']
                      for r in all_results if not np.isnan(r['timing_counterfactuals'][shift]['rmse_mean'])]
        mae_means = [r['timing_counterfactuals'][shift]['mae_mean']
                     for r in all_results if not np.isnan(r['timing_counterfactuals'][shift]['mae_mean'])]

        if len(rmse_means) > 0:
            timing_cf_summary[shift] = {
                'rmse_mean': float(np.mean(rmse_means)),
                'rmse_std': float(np.std(rmse_means, ddof=1)) if len(rmse_means) > 1 else 0.0,
                'mae_mean': float(np.mean(mae_means)),
                'mae_std': float(np.std(mae_means, ddof=1)) if len(mae_means) > 1 else 0.0,
                'n_seeds': len(rmse_means)
            }
        else:
            timing_cf_summary[shift] = {
                'rmse_mean': np.nan,
                'rmse_std': np.nan,
                'mae_mean': np.nan,
                'mae_std': np.nan,
                'n_seeds': 0
            }

    summary = {
        'n_seeds': len(all_results),
        'overall_rmse': {
            'mean': float(mean_rmse),
            'std': float(std_rmse)
        },
        'timestep_rmse': timestep_stats,
        'dose_counterfactuals': dose_cf_summary,
        'timing_counterfactuals': timing_cf_summary
    }

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("GLUCOSE MODEL EVALUATION SUMMARY (Aggregated Across Seeds)")
    print("="*80)
    print(f"Seeds: {len(all_results)}")
    print(f"\nFACTUAL PREDICTION (Future Forecasting)")
    print(f"Overall RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} mg/dL")
    print(f"\nPer-timestep RMSE (first 10):")
    for t in range(min(10, len(timestep_stats))):
        t_key = t + 1
        print(f"  τ={t_key}: {timestep_stats[t_key]['mean']:.4f} ± {timestep_stats[t_key]['std']:.4f} mg/dL")

    # Print counterfactual summary
    print("\n" + "="*80)
    print("COUNTERFACTUAL PREDICTION ACCURACY (vs Ground Truth)")
    print("="*80)

    print("\nDOSE COUNTERFACTUALS:")
    print(f"{'Dose Factor':<15} {'RMSE (mean±std)':<25} {'MAE (mean±std)':<25} {'Seeds':<10}")
    print("-"*80)
    for factor in sorted([float(f) for f in dose_factors]):
        factor_key = str(factor) if str(factor) in dose_cf_summary else factor
        result = dose_cf_summary[factor_key]
        if not np.isnan(result['rmse_mean']):
            print(f"{factor:<15.2f} "
                  f"{result['rmse_mean']:>8.3f}±{result['rmse_std']:<10.3f} "
                  f"{result['mae_mean']:>8.3f}±{result['mae_std']:<10.3f} "
                  f"{result['n_seeds']:<10}")
        else:
            print(f"{factor:<15.2f} {'N/A':<25} {'N/A':<25} {result['n_seeds']:<10}")

    print("\nTIMING COUNTERFACTUALS:")
    print(f"{'Shift (min)':<15} {'RMSE (mean±std)':<25} {'MAE (mean±std)':<25} {'Seeds':<10}")
    print("-"*80)
    for shift in sorted([int(s) for s in timing_shifts]):
        shift_key = str(shift) if str(shift) in timing_cf_summary else shift
        result = timing_cf_summary[shift_key]
        if not np.isnan(result['rmse_mean']):
            print(f"{shift:<15} "
                  f"{result['rmse_mean']:>8.3f}±{result['rmse_std']:<10.3f} "
                  f"{result['mae_mean']:>8.3f}±{result['mae_std']:<10.3f} "
                  f"{result['n_seeds']:<10}")
        else:
            print(f"{shift:<15} {'N/A':<25} {'N/A':<25} {result['n_seeds']:<10}")

    print("="*80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate glucose causal transformer")
    parser.add_argument('--output_dir', type=str, default='experiments/glucose',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default='config/glucose_config.yaml',
                       help='Config file for glucose model')
    args = parser.parse_args()

    # Load config
    print("Loading configuration...")
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("="*80)
    print("GLUCOSE CAUSAL TRANSFORMER TRAINING")
    print("="*80)
    print(f"Data: {config['num_days']} days")
    print(f"History window: {config['history_window']} timesteps ({config['history_window']*5/60:.1f} hours)")
    print(f"Prediction horizon: {config['prediction_horizon']} timesteps ({config['prediction_horizon']*5/60:.1f} hours)")
    print(f"Seeds: {config['seeds']}")
    print(f"Epochs: {config['epochs']}")
    print(f"LR: {config['lr']}")
    print(f"d_model: {config['d_model']}")
    print("="*80)

    # Generate shared test set
    print("\nGenerating shared test set (seed=999)...")
    _, _, test_df, test_scaler = generate_glucose_data(
        num_days=config['num_days'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        seed=999
    )

    # Generate full patient data for counterfactual evaluation
    print("Generating full patient data for counterfactual evaluation (seed=999)...")
    analyzer = DiabetesAnalyzer(seed=999)
    full_patient_data = analyzer.generate_patient_data(n_days=config['num_days'])

    all_results = []

    for seed in config['seeds']:
        run_dir = output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create model
        model = GlucoseTransformerCRN(
            input_dim=9,  # glucose, insulin, carbs, exercise, stress, active_insulin, carb_impact, sin_hour, cos_hour
            treatment_dim=6,  # insulin, carbs, exercise, stress, sin_hour, cos_hour
            output_dim=1,  # glucose prediction
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            br_size=config['br_size']
        )

        # Train and evaluate
        results = train_and_evaluate_model(
            model, config, seed, str(run_dir), test_df, test_scaler, full_patient_data
        )

        # Save results
        with open(run_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        all_results.append(results)

    # Aggregate results
    summary = aggregate_results(all_results, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
