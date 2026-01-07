"""
Glucose Counterfactual Evaluator

Evaluates model's ability to predict counterfactual glucose outcomes
when insulin doses or timing are modified.
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from src.data.generator import CounterfactualAnalyzer
from src.data.dataset import GlucoseDataset


class GlucoseCounterfactualEvaluator:
    """Evaluate model on counterfactual 'what-if' scenarios"""

    def __init__(self, model, scaler, device='cpu'):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.cf_analyzer = CounterfactualAnalyzer(seed=999)

    def evaluate_dose_counterfactuals(self,
                                      patient_data,
                                      dose_factors=[0.8, 1.2, 1.5],
                                      num_interventions=20,
                                      history_window=144,
                                      prediction_horizon=36):
        """
        Evaluate dose counterfactuals: "What if insulin dose was X% different?"

        Args:
            patient_data: DataFrame with patient glucose data
            dose_factors: List of dose multipliers (e.g., 1.2 = 20% more insulin)
            num_interventions: Number of interventions to test
            history_window: History window in timesteps (144 = 12 hours)
            prediction_horizon: Prediction horizon in timesteps (36 = 3 hours)

        Returns:
            Dict with results per dose_factor
        """
        # Get all interventions
        interventions = self.cf_analyzer.list_interventions(patient_data)

        if len(interventions) == 0:
            raise ValueError("No interventions found in patient data")

        # Sample interventions to test (avoid testing all for speed)
        sample_size = min(num_interventions, len(interventions))
        np.random.seed(42)  # For reproducibility
        sampled_interventions = np.random.choice(len(interventions), sample_size, replace=False)

        print(f"\nEvaluating {sample_size} interventions with {len(dose_factors)} dose modifications...")

        results = {factor: {'rmse': [], 'mae': [], 'predictions': [], 'targets': []}
                   for factor in dose_factors}

        for intv_idx in tqdm(sampled_interventions, desc="Testing interventions"):
            intervention = interventions[intv_idx]
            intervention_id = intervention['id']
            intervention_time = intervention['timestamp']

            # Need enough history before intervention
            time_from_start = (intervention_time - patient_data.index[0]).total_seconds() / 60 / 5
            if time_from_start < history_window:
                continue

            # Need enough future after intervention
            time_to_end = (patient_data.index[-1] - intervention_time).total_seconds() / 60 / 5
            if time_to_end < prediction_horizon:
                continue

            # For each dose modification
            for dose_factor in dose_factors:
                try:
                    # Generate counterfactual with modified dose
                    cf_data = self.cf_analyzer.generate_dose_counterfactual(
                        base_data=patient_data,
                        intervention_id=intervention_id,
                        new_dose_factor=dose_factor,
                        before_minutes=history_window * 5,  # Convert to minutes
                        after_minutes=prediction_horizon * 5,
                        num_scenarios=1
                    )

                    # Extract counterfactual ground truth
                    cf_number = 1  # First counterfactual
                    cf_glucose_col = f'cf{cf_number}_glucose'
                    cf_insulin_col = f'cf{cf_number}_insulin'

                    if cf_glucose_col not in cf_data.columns:
                        continue

                    # Get history (before intervention)
                    history_start_idx = patient_data.index.get_loc(intervention_time) - history_window
                    history_end_idx = patient_data.index.get_loc(intervention_time)

                    if history_start_idx < 0:
                        continue

                    # Extract and normalize history features
                    history_df = patient_data.iloc[history_start_idx:history_end_idx]
                    history_features = GlucoseDataset.extract_features(history_df)
                    history_normalized = self.scaler.transform(history_features)
                    history_tensor = torch.from_numpy(history_normalized).float().unsqueeze(0).to(self.device)

                    # Extract counterfactual future treatments
                    future_start_idx = history_end_idx
                    future_end_idx = future_start_idx + prediction_horizon

                    if future_end_idx > len(cf_data):
                        continue

                    # Build future treatment features with MODIFIED insulin
                    future_treatments = []
                    for idx in range(prediction_horizon):
                        row_idx = future_start_idx + idx
                        row_time = cf_data.index[row_idx]
                        hour = row_time.hour + row_time.minute / 60

                        future_treatments.append([
                            cf_data[cf_insulin_col].iloc[row_idx],  # MODIFIED insulin
                            cf_data['carbs'].iloc[row_idx],
                            cf_data['exercise'].iloc[row_idx],
                            cf_data['stress'].iloc[row_idx],
                            np.sin(2 * np.pi * hour / 24),
                            np.cos(2 * np.pi * hour / 24)
                        ])

                    future_treatments = np.array(future_treatments)

                    # Normalize treatments (indices 1,2,3,4,7,8 in full feature set of 9)
                    dummy_features = np.zeros((len(future_treatments), 9))
                    dummy_features[:, [1, 2, 3, 4, 7, 8]] = future_treatments
                    normalized = self.scaler.transform(dummy_features)
                    future_treatments_normalized = normalized[:, [1, 2, 3, 4, 7, 8]]

                    future_treatments_tensor = torch.from_numpy(future_treatments_normalized).float().unsqueeze(0).to(self.device)

                    # Model prediction
                    with torch.no_grad():
                        pred = self.model.predict_counterfactual(history_tensor, future_treatments_tensor)
                        pred_np = pred.cpu().numpy()[0, :, 0]  # [pred_horizon]

                    # Denormalize prediction (glucose is index 0)
                    dummy_pred = np.zeros((len(pred_np), 9))
                    dummy_pred[:, 0] = pred_np
                    denormalized = self.scaler.inverse_transform(dummy_pred)
                    pred_glucose = denormalized[:, 0]

                    # Get ground truth counterfactual glucose
                    target_glucose = cf_data[cf_glucose_col].iloc[future_start_idx:future_end_idx].values

                    # Compute errors
                    rmse = np.sqrt(np.mean((pred_glucose - target_glucose) ** 2))
                    mae = np.mean(np.abs(pred_glucose - target_glucose))

                    results[dose_factor]['rmse'].append(rmse)
                    results[dose_factor]['mae'].append(mae)
                    results[dose_factor]['predictions'].extend(pred_glucose.tolist())
                    results[dose_factor]['targets'].extend(target_glucose.tolist())

                except Exception as e:
                    # Skip this intervention if there's an error
                    continue

        # Aggregate results
        summary = {}
        for factor in dose_factors:
            if len(results[factor]['rmse']) > 0:
                summary[factor] = {
                    'rmse_mean': np.mean(results[factor]['rmse']),
                    'rmse_std': np.std(results[factor]['rmse']),
                    'mae_mean': np.mean(results[factor]['mae']),
                    'mae_std': np.std(results[factor]['mae']),
                    'n_samples': len(results[factor]['rmse'])
                }
            else:
                summary[factor] = {
                    'rmse_mean': np.nan,
                    'rmse_std': np.nan,
                    'mae_mean': np.nan,
                    'mae_std': np.nan,
                    'n_samples': 0
                }

        return summary

    def evaluate_timing_counterfactuals(self,
                                        patient_data,
                                        timing_shifts=[-30, -15, 15, 30],
                                        num_interventions=20,
                                        history_window=144,
                                        prediction_horizon=36):
        """
        Evaluate timing counterfactuals: "What if insulin was taken X minutes earlier/later?"

        Args:
            patient_data: DataFrame with patient glucose data
            timing_shifts: List of time shifts in minutes (negative = earlier, positive = later)
            num_interventions: Number of interventions to test
            history_window: History window in timesteps
            prediction_horizon: Prediction horizon in timesteps

        Returns:
            Dict with results per timing_shift
        """
        interventions = self.cf_analyzer.list_interventions(patient_data)

        if len(interventions) == 0:
            raise ValueError("No interventions found in patient data")

        sample_size = min(num_interventions, len(interventions))
        np.random.seed(42)
        sampled_interventions = np.random.choice(len(interventions), sample_size, replace=False)

        print(f"\nEvaluating {sample_size} interventions with {len(timing_shifts)} timing modifications...")

        results = {shift: {'rmse': [], 'mae': [], 'predictions': [], 'targets': []}
                   for shift in timing_shifts}

        for intv_idx in tqdm(sampled_interventions, desc="Testing timing shifts"):
            intervention = interventions[intv_idx]
            intervention_id = intervention['id']
            intervention_time = intervention['timestamp']

            # Need enough history and future (with buffer for timing shifts)
            time_from_start = (intervention_time - patient_data.index[0]).total_seconds() / 60 / 5
            time_to_end = (patient_data.index[-1] - intervention_time).total_seconds() / 60 / 5

            # Need buffer of 60 timesteps (5 hours) for timing shifts
            if time_from_start < history_window + 60 or time_to_end < prediction_horizon + 60:
                continue

            for timing_shift in timing_shifts:
                try:
                    # Generate timing counterfactual
                    cf_data = self.cf_analyzer.generate_timing_counterfactual(
                        base_data=patient_data,
                        intervention_id=intervention_id,
                        timing_shift_minutes=timing_shift,
                        before_minutes=history_window * 5,
                        after_minutes=prediction_horizon * 5,
                        num_scenarios=1
                    )

                    # Similar processing as dose counterfactuals
                    cf_number = 1
                    cf_glucose_col = f'cf{cf_number}_glucose'
                    cf_insulin_col = f'cf{cf_number}_insulin'

                    if cf_glucose_col not in cf_data.columns:
                        continue

                    # Get history (before ORIGINAL intervention time)
                    history_start_idx = patient_data.index.get_loc(intervention_time) - history_window
                    history_end_idx = patient_data.index.get_loc(intervention_time)

                    if history_start_idx < 0:
                        continue

                    # Extract and normalize history
                    history_df = patient_data.iloc[history_start_idx:history_end_idx]
                    history_features = GlucoseDataset.extract_features(history_df)
                    history_normalized = self.scaler.transform(history_features)
                    history_tensor = torch.from_numpy(history_normalized).float().unsqueeze(0).to(self.device)

                    # Extract future with SHIFTED insulin timing
                    future_start_idx = history_end_idx
                    future_end_idx = future_start_idx + prediction_horizon

                    if future_end_idx > len(cf_data):
                        continue

                    # Build future treatment features
                    future_treatments = []
                    for idx in range(prediction_horizon):
                        row_idx = future_start_idx + idx
                        row_time = cf_data.index[row_idx]
                        hour = row_time.hour + row_time.minute / 60

                        future_treatments.append([
                            cf_data[cf_insulin_col].iloc[row_idx],  # SHIFTED insulin
                            cf_data['carbs'].iloc[row_idx],
                            cf_data['exercise'].iloc[row_idx],
                            cf_data['stress'].iloc[row_idx],
                            np.sin(2 * np.pi * hour / 24),
                            np.cos(2 * np.pi * hour / 24)
                        ])

                    future_treatments = np.array(future_treatments)

                    # Normalize
                    dummy_features = np.zeros((len(future_treatments), 9))
                    dummy_features[:, [1, 2, 3, 4, 7, 8]] = future_treatments
                    normalized = self.scaler.transform(dummy_features)
                    future_treatments_normalized = normalized[:, [1, 2, 3, 4, 7, 8]]

                    future_treatments_tensor = torch.from_numpy(future_treatments_normalized).float().unsqueeze(0).to(self.device)

                    # Model prediction
                    with torch.no_grad():
                        pred = self.model.predict_counterfactual(history_tensor, future_treatments_tensor)
                        pred_np = pred.cpu().numpy()[0, :, 0]

                    # Denormalize
                    dummy_pred = np.zeros((len(pred_np), 9))
                    dummy_pred[:, 0] = pred_np
                    denormalized = self.scaler.inverse_transform(dummy_pred)
                    pred_glucose = denormalized[:, 0]

                    # Ground truth
                    target_glucose = cf_data[cf_glucose_col].iloc[future_start_idx:future_end_idx].values

                    # Compute errors
                    rmse = np.sqrt(np.mean((pred_glucose - target_glucose) ** 2))
                    mae = np.mean(np.abs(pred_glucose - target_glucose))

                    results[timing_shift]['rmse'].append(rmse)
                    results[timing_shift]['mae'].append(mae)
                    results[timing_shift]['predictions'].extend(pred_glucose.tolist())
                    results[timing_shift]['targets'].extend(target_glucose.tolist())

                except Exception as e:
                    continue

        # Aggregate
        summary = {}
        for shift in timing_shifts:
            if len(results[shift]['rmse']) > 0:
                summary[shift] = {
                    'rmse_mean': np.mean(results[shift]['rmse']),
                    'rmse_std': np.std(results[shift]['rmse']),
                    'mae_mean': np.mean(results[shift]['mae']),
                    'mae_std': np.std(results[shift]['mae']),
                    'n_samples': len(results[shift]['rmse'])
                }
            else:
                summary[shift] = {
                    'rmse_mean': np.nan,
                    'rmse_std': np.nan,
                    'mae_mean': np.nan,
                    'mae_std': np.nan,
                    'n_samples': 0
                }

        return summary

    def print_results(self, dose_results=None, timing_results=None):
        """Print evaluation results"""
        if dose_results:
            print("\n" + "="*80)
            print("DOSE COUNTERFACTUAL EVALUATION")
            print("="*80)
            print(f"{'Dose Factor':<15} {'RMSE (mean±std)':<30} {'MAE (mean±std)':<30} {'N':<10}")
            print("-"*80)

            for factor in sorted(dose_results.keys()):
                result = dose_results[factor]
                if not np.isnan(result['rmse_mean']):
                    print(f"{factor:<15.2f} "
                          f"{result['rmse_mean']:>8.3f}±{result['rmse_std']:<12.3f} "
                          f"{result['mae_mean']:>8.3f}±{result['mae_std']:<12.3f} "
                          f"{result['n_samples']:<10}")
                else:
                    print(f"{factor:<15.2f} {'N/A':<30} {'N/A':<30} {result['n_samples']:<10}")
            print("="*80)

        if timing_results:
            print("\n" + "="*80)
            print("TIMING COUNTERFACTUAL EVALUATION")
            print("="*80)
            print(f"{'Time Shift (min)':<15} {'RMSE (mean±std)':<30} {'MAE (mean±std)':<30} {'N':<10}")
            print("-"*80)

            for shift in sorted(timing_results.keys()):
                result = timing_results[shift]
                if not np.isnan(result['rmse_mean']):
                    print(f"{shift:<15} "
                          f"{result['rmse_mean']:>8.3f}±{result['rmse_std']:<12.3f} "
                          f"{result['mae_mean']:>8.3f}±{result['mae_std']:<12.3f} "
                          f"{result['n_samples']:<10}")
                else:
                    print(f"{shift:<15} {'N/A':<30} {'N/A':<30} {result['n_samples']:<10}")
            print("="*80)
