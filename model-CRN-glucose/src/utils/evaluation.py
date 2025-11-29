import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class CounterfactualEvaluator:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.device = next(model.parameters()).device
        self.model.eval()
        
    def visualize_scenario(self, idx, modification_func=None, save_path=None):
        """
        Visualize a single scenario and its counterfactual.
        
        idx: Index in the dataset
        modification_func: Function that takes (future_treatments) and returns modified version.
                           If None, just shows factual prediction.
        """
        sample = self.dataset[idx]
        
        # Prepare inputs
        history = sample['encoder_inputs'].unsqueeze(0).to(self.device)
        future_treat = sample['future_treatments'].unsqueeze(0).to(self.device)
        actual_outcome = sample['future_outcomes'].numpy()
        
        # 1. Predict Factual
        with torch.no_grad():
            # predict_counterfactual is inference mode (no GRL)
            factual_pred = self.model.predict_counterfactual(history, future_treat)
            
        factual_pred_np = factual_pred.cpu().numpy()[0]
        
        # Plotting setup
        plt.figure(figsize=(12, 6))
        
        # Plot History (Last 2 hours for context)
        history_len = history.shape[1]
        display_hist = 24 # Show last 2 hours of history
        hist_glucose = history[0, -display_hist:, 0].cpu().numpy() # Glucose is idx 0
        
        t_hist = np.arange(-display_hist, 0) * 5 # minutes
        t_pred = np.arange(0, len(factual_pred_np)) * 5 # minutes
        
        # Denormalize if possible (assuming we have access to scaler parameters)
        # For visualization, we'll plot normalized values for now, or use scaler inverse if attached
        
        # Plot Context
        plt.plot(t_hist, hist_glucose, 'k:', label='History (Glucose)')
        
        # Plot Actual Future
        plt.plot(t_pred, actual_outcome, 'k-', alpha=0.3, label='Actual Outcome')
        
        # Plot Factual Prediction
        plt.plot(t_pred, factual_pred_np, 'b-', linewidth=2, label='Model Prediction (Factual)')
        
        # 2. Predict Counterfactual if requested
        if modification_func:
            # Clone and modify
            cf_treat = future_treat.clone()
            cf_treat = modification_func(cf_treat)
            
            with torch.no_grad():
                cf_pred = self.model.predict_counterfactual(history, cf_treat)
            
            cf_pred_np = cf_pred.cpu().numpy()[0]
            plt.plot(t_pred, cf_pred_np, 'r--', linewidth=2, label='Counterfactual Prediction')
            
            # Highlight the treatment difference
            # Assuming Insulin is index 0 of treatment vector (which is index 1 of full vector)
            # Our dataset.py says treatment vector indices are [1, 2, 3, 4, 7, 8] (Insulin, Carbs...)
            # So Insulin is index 0 in the future_treat tensor.
            orig_insulin = future_treat[0, :, 0].sum().item()
            new_insulin = cf_treat[0, :, 0].sum().item()
            plt.title(f"Counterfactual Analysis: Insulin {orig_insulin:.2f} -> {new_insulin:.2f}")
            
        plt.xlabel("Time (minutes)")
        plt.ylabel("Glucose (Normalized)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        plt.close()

