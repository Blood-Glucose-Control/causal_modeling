"""
Evaluation script matching Melnychuk et al.'s Causal Transformer paper methodology.
Evaluates τ-step-ahead counterfactual predictions on cancer simulation data.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class CancerEvaluator:
    """Evaluate model on τ-step-ahead counterfactual prediction"""

    def __init__(self, model, scaler, device='cpu'):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate_tau_step_ahead(self, test_data_cf, projection_horizon=5, history_window=15):
        """
        Evaluate τ-step-ahead prediction on counterfactual sequences

        Args:
            test_data_cf: Dict from simulate_counterfactuals_treatment_seq
            projection_horizon: Maximum τ (e.g., 5 means τ=1,2,3,4,5)
            history_window: History length for encoder

        Returns:
            Dict mapping τ -> RMSE
        """
        cancer_volume = test_data_cf['cancer_volume']
        chemo_application = test_data_cf['chemo_application']
        radio_application = test_data_cf['radio_application']
        patient_types = test_data_cf['patient_types']
        patient_current_t = test_data_cf['patient_current_t'].astype(int)
        sequence_lengths = test_data_cf['sequence_lengths'].astype(int)

        # Initialize results for each τ
        tau_results = {tau: {'predictions': [], 'targets': [], 'squared_errors': []}
                      for tau in range(1, projection_horizon + 1)}

        with torch.no_grad():
            for i in tqdm(range(len(sequence_lengths)), desc="Evaluating counterfactuals"):
                seq_len = sequence_lengths[i]
                current_t = patient_current_t[i]

                # Need enough history
                if current_t < history_window:
                    continue

                # Extract history window
                history_start = current_t - history_window
                history_end = current_t

                history_features = []
                for t in range(history_start, history_end):
                    history_features.append([
                        cancer_volume[i, t],
                        chemo_application[i, t],
                        radio_application[i, t],
                        patient_types[i]
                    ])
                history_features = np.array(history_features)

                # Normalize history
                history_normalized = self.scaler.transform(history_features)
                history_tensor = torch.from_numpy(history_normalized).float().unsqueeze(0).to(self.device)

                # For each τ, predict τ steps ahead
                for tau in range(1, min(projection_horizon + 1, seq_len - current_t)):
                    # Extract future treatments for τ steps
                    future_treatments = []
                    for t in range(current_t, current_t + tau):
                        future_treatments.append([
                            chemo_application[i, t],
                            radio_application[i, t]
                        ])
                    future_treatments = np.array(future_treatments)

                    # Normalize treatments
                    # We need to normalize using the full feature scaler but only take treatment columns
                    dummy_features = np.zeros((tau, 4))
                    dummy_features[:, 1:3] = future_treatments
                    normalized = self.scaler.transform(dummy_features)
                    future_treatments_normalized = normalized[:, 1:3]

                    future_treatments_tensor = torch.from_numpy(future_treatments_normalized).float().unsqueeze(0).to(self.device)

                    # Predict
                    pred = self.model.predict_counterfactual(history_tensor, future_treatments_tensor)
                    pred_np = pred.cpu().numpy()  # [1, tau, 1]

                    # Get ground truth for τ-th step (index τ-1 in 0-indexed array)
                    target_t = current_t + tau
                    target = cancer_volume[i, target_t]

                    # Denormalize prediction (cancer_volume is feature 0)
                    # Create dummy array for inverse transform
                    dummy_pred = np.zeros((1, 4))
                    dummy_pred[0, 0] = pred_np[0, tau-1, 0]
                    denormalized = self.scaler.inverse_transform(dummy_pred)
                    pred_denorm = denormalized[0, 0]

                    # Store results
                    tau_results[tau]['predictions'].append(pred_denorm)
                    tau_results[tau]['targets'].append(target)
                    tau_results[tau]['squared_errors'].append((pred_denorm - target) ** 2)

        # Calculate RMSE for each τ
        rmse_results = {}
        for tau in range(1, projection_horizon + 1):
            if len(tau_results[tau]['squared_errors']) > 0:
                rmse = np.sqrt(np.mean(tau_results[tau]['squared_errors']))
                rmse_results[tau] = {
                    'rmse': rmse,
                    'n_samples': len(tau_results[tau]['squared_errors']),
                    'predictions': np.array(tau_results[tau]['predictions']),
                    'targets': np.array(tau_results[tau]['targets'])
                }
            else:
                rmse_results[tau] = {'rmse': np.nan, 'n_samples': 0}

        return rmse_results

    def evaluate_one_step_ahead(self, test_data_factual, history_window=15):
        """
        Evaluate one-step-ahead prediction on factual test data (sanity check)

        Args:
            test_data_factual: Dict from simulate_factual
            history_window: History length

        Returns:
            RMSE on one-step-ahead predictions
        """
        cancer_volume = test_data_factual['cancer_volume']
        chemo_application = test_data_factual['chemo_application']
        radio_application = test_data_factual['radio_application']
        patient_types = test_data_factual['patient_types']
        sequence_lengths = test_data_factual['sequence_lengths'].astype(int)

        squared_errors = []

        with torch.no_grad():
            for i in tqdm(range(len(sequence_lengths)), desc="Evaluating factual"):
                seq_len = sequence_lengths[i]

                # Slide through sequence
                for current_t in range(history_window, seq_len - 1):
                    # History
                    history_features = []
                    for t in range(current_t - history_window, current_t):
                        history_features.append([
                            cancer_volume[i, t],
                            chemo_application[i, t],
                            radio_application[i, t],
                            patient_types[i]
                        ])
                    history_features = np.array(history_features)
                    history_normalized = self.scaler.transform(history_features)
                    history_tensor = torch.from_numpy(history_normalized).float().unsqueeze(0).to(self.device)

                    # Next treatment
                    next_treatment = np.array([[
                        chemo_application[i, current_t],
                        radio_application[i, current_t]
                    ]])

                    # Normalize
                    dummy = np.zeros((1, 4))
                    dummy[:, 1:3] = next_treatment
                    normalized = self.scaler.transform(dummy)
                    next_treatment_normalized = normalized[:, 1:3]
                    next_treatment_tensor = torch.from_numpy(next_treatment_normalized).float().unsqueeze(0).to(self.device)

                    # Predict
                    pred = self.model.predict_counterfactual(history_tensor, next_treatment_tensor)
                    pred_np = pred.cpu().numpy()[0, 0, 0]

                    # Denormalize
                    dummy_pred = np.zeros((1, 4))
                    dummy_pred[0, 0] = pred_np
                    denormalized = self.scaler.inverse_transform(dummy_pred)
                    pred_denorm = denormalized[0, 0]

                    # Target
                    target = cancer_volume[i, current_t + 1]

                    squared_errors.append((pred_denorm - target) ** 2)

        rmse = np.sqrt(np.mean(squared_errors))
        return rmse

    def print_results(self, rmse_results):
        """Print results in paper format"""
        print("\n" + "=" * 80)
        print("τ-STEP-AHEAD COUNTERFACTUAL PREDICTION RESULTS")
        print("=" * 80)
        print(f"{'τ':<5} {'RMSE':<15} {'N Samples':<15}")
        print("-" * 80)

        for tau in sorted(rmse_results.keys()):
            result = rmse_results[tau]
            if not np.isnan(result['rmse']):
                print(f"{tau:<5} {result['rmse']:<15.4f} {result['n_samples']:<15}")
            else:
                print(f"{tau:<5} {'N/A':<15} {result['n_samples']:<15}")

        print("=" * 80)
