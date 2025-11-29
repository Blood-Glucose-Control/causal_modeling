import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy
import pandas as pd
import sys

# Add project root to path to find src module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.crn_transformer import GlucoseTransformerCRN
from src.data.dataset import GlucoseDataset
from src.data.generator import DiabetesAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    args = parser.parse_args()
    
    # 1. Initialize Helper Classes
    print("Initializing Environment...")
    # We use this dataset instance primarily for the Scaler
    # Using small num_days to be fast
    dataset_helper = GlucoseDataset(mode='train', num_days=10) 
    
    analyzer = DiabetesAnalyzer(seed=100) # New seed for evaluation
    
    # 2. Load Model
    print("Loading Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GlucoseTransformerCRN(
        input_dim=9,
        treatment_dim=6,
        output_dim=1
    ).to(device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
        model.eval()
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    # 3. Generate Evaluation Scenarios
    print("Generating Base Scenario...")
    # Generate a month of data to find good intervention points
    base_df = analyzer.generate_patient_data(n_days=30)
    interventions = analyzer.counterfactual_model.list_interventions(base_df)
    
    print(f"Found {len(interventions)} interventions to analyze.")
    
    os.makedirs('analysis/plots', exist_ok=True)
    
    # Analyze first 5 valid interventions
    count = 0
    for i, intervention in enumerate(interventions):
        intervention_id = intervention['id']
        target_time = intervention['timestamp']
        
        # Find the index in the dataframe corresponding to the intervention
        if target_time not in base_df.index:
            continue
        
        # We need enough history (12h) before the intervention
        idx_loc = base_df.index.get_loc(target_time)
        hist_len = dataset_helper.history_window
        
        if idx_loc < hist_len:
            continue
            
        print(f"Analyzing Intervention {i} at {target_time} (Dose: {intervention['dose']:.1f}U)")
        
        # --- Scenario A: Factual (Original Dose) ---
        # We treat the original data as "Scenario A"
        
        # --- Scenario B: Counterfactual (e.g., 1.5x Dose) ---
        dose_factor = 1.5
        cf_df_raw = analyzer.analyze_intervention(
            base_df, 
            intervention_id, 
            analysis_type='dose', 
            dose_factor=dose_factor
        )
        
        # We need to extract the "Counterfactual World" dataframe from this mixed result
        # The generator puts CF values in 'cf{N}_glucose', etc.
        cf_prefix = 'cf1' # Usually the first one
        
        # Reconstruct the clean CF dataframe
        cf_world_df = cf_df_raw.copy()
        cf_world_df['glucose'] = cf_df_raw[f'{cf_prefix}_glucose']
        cf_world_df['insulin'] = cf_df_raw[f'{cf_prefix}_insulin']
        cf_world_df['active_insulin'] = cf_df_raw[f'{cf_prefix}_active_insulin']
        cf_world_df['carb_impact'] = cf_df_raw[f'{cf_prefix}_carb_impact']
        
        # Now we have two dataframes: base_df (Factual) and cf_world_df (Counterfactual Truth)
        # We need to prepare Model Inputs for both
        
        # Prepare Model Inputs
        # We need to slice the window [t - history, t + horizon]
        # Note: Intervention happens at 'target_time' which is usually t=0 in the future window
        # The dataset class logic: mid_idx is the start of prediction.
        # So we want the intervention to be at mid_idx (or slightly after)
        
        # Let's define the window start such that intervention is at step 0 of future
        start_pos = idx_loc - hist_len
        
        # Transform both worlds to tensors
        feat_factual = dataset_helper.transform_new_data(base_df)
        feat_counterfactual = dataset_helper.transform_new_data(cf_world_df)
        
        # Get Window Tensors
        # Both share the same history (mostly, up to t=0)
        # But we use the respective tensors to be safe
        input_factual = dataset_helper.get_window(feat_factual, start_pos)
        input_cf = dataset_helper.get_window(feat_counterfactual, start_pos)
        
        # Run Model Inference
        def predict(inputs):
            hist = inputs['encoder_inputs'].unsqueeze(0).to(device)
            fut_treat = inputs['future_treatments'].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model.predict_counterfactual(hist, fut_treat)
            return pred.cpu().numpy()[0, :, 0] # [Time]
            
        pred_factual = predict(input_factual)
        pred_cf = predict(input_cf)
        
        # Get Ground Truth Outcomes (Normalized or Unnormalized?)
        # The model predicts Normalized Glucose.
        # Let's invert transform for plotting to make it interpretable (mg/dL)
        
        # Helper to inverse transform glucose
        # We know glucose is index 0
        glucose_mean = dataset_helper.scaler.mean_[0]
        glucose_std = dataset_helper.scaler.scale_[0]
        
        def denorm(x): return x * glucose_std + glucose_mean
        
        # Get GT Glucose
        # We need the raw values from the dataframe window
        gt_factual = base_df['glucose'].iloc[idx_loc : idx_loc + 36].values
        gt_cf = cf_world_df['glucose'].iloc[idx_loc : idx_loc + 36].values
        
        # Plot
        plt.figure(figsize=(12, 6))
        t = np.arange(36) * 5 # Minutes
        
        # Plot Ground Truths
        plt.plot(t, gt_factual, 'b--', alpha=0.5, label='GT Factual (1.0x)')
        plt.plot(t, gt_cf, 'r--', alpha=0.5, label=f'GT Counterfactual ({dose_factor}x)')
        
        # Plot Model Predictions (Denormalized)
        plt.plot(t, denorm(pred_factual), 'b-', linewidth=2, label='Model Factual')
        plt.plot(t, denorm(pred_cf), 'r-', linewidth=2, label='Model Counterfactual')
        
        plt.title(f"Model vs Ground Truth: Insulin Dose {intervention['dose']:.1f}U -> {intervention['dose']*dose_factor:.1f}U")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Glucose (mg/dL)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = f"analysis/plots/scenario_{i}.png"
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()
        
        count += 1
        if count >= 5: break
        
    print("Evaluation Complete.")

if __name__ == "__main__":
    main()

