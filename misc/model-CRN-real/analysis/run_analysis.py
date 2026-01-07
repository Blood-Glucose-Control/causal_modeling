import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import sys
from tqdm import tqdm

# Add project root to path to find src module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.crn_transformer import GlucoseTransformerCRN
from src.data.dataset import GlucoseDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--data_path', type=str, default='../output/processed_training_data.csv')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()
    
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load Data
    print(f"Loading data from {args.data_path}...")
    if not os.path.exists(args.data_path):
        print("Data file not found!")
        return
        
    full_df = pd.read_csv(args.data_path)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    full_df = full_df.set_index('timestamp')
    
    # Use only validation split (last 20%)
    split_idx = int(0.8 * len(full_df))
    val_df = full_df.iloc[split_idx:].copy()
    
    # Initialize Dataset (Fit scaler on full data or load saved scaler ideally, 
    # here fitting on full for simplicity of analysis script)
    # Note: In strict ML we should load the scaler from training. 
    # For quick analysis, fitting on full dataset is acceptable approx.
    print("Initializing Dataset...")
    # Create a dummy dataset just to fit the scaler
    temp_features = GlucoseDataset.extract_features(full_df)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(temp_features)
    
    val_ds = GlucoseDataset(val_df, scaler=scaler)
    
    # 3. Load Model
    print("Loading Model...")
    # Must match training config
    model = GlucoseTransformerCRN(
        input_dim=12, 
        treatment_dim=4, 
        output_dim=1
    ).to(device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        print("Checkpoint loaded.")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Using random weights (Expect garbage).")

    # 4. Find Interesting Events (Meal/Insulin spikes)
    # We look for times where insulin > 0
    # 'insulin' is index 1 in the feature matrix (from Dataset logic)
    insulin_idx = 1
    
    # Get all windows
    indices = val_ds.valid_indices
    interesting_indices = []
    
    print("Searching for insulin events in validation set...")
    for idx in indices:
        # Check if there is insulin in the FUTURE window (prediction horizon)
        # Window dict: 'future_treatments' [Batch, 36, 4]
        # We need to look at the raw data before normalization to be sure, 
        # or just check the normalized tensor.
        
        # Let's peek at the raw dataframe corresponding to the future window
        end_pos = idx + val_ds.history_window + val_ds.prediction_horizon
        if end_pos >= len(val_ds.raw_df):
            continue
            
        start_time = val_ds.raw_df.index[idx + val_ds.history_window]
        end_time = val_ds.raw_df.index[end_pos]
        
        window_slice = val_ds.raw_df.loc[start_time:end_time]
        total_insulin = window_slice['insulin'].sum()
        
        if total_insulin > 2.0: # At least 2 units
            interesting_indices.append(idx)
            
    if not interesting_indices:
        print("No significant insulin events found in validation set. Using random indices.")
        interesting_indices = np.random.choice(indices, args.num_samples, replace=False)
    else:
        # Pick random subset of interesting ones
        np.random.shuffle(interesting_indices)
        interesting_indices = interesting_indices[:args.num_samples]

    # 5. Run Counterfactual Analysis
    print(f"\nGenerating plots for {len(interesting_indices)} events...")
    os.makedirs('analysis/plots', exist_ok=True)
    
    # Helper for denormalization
    gluc_mean = scaler.mean_[0]
    gluc_std = scaler.scale_[0]
    def denorm(x): return x * gluc_std + gluc_mean

    for i, idx in enumerate(interesting_indices):
        # Get factual data
        item = val_ds[val_ds.valid_indices.index(idx)] # Map raw index back to dataset index
        
        history = item['encoder_inputs'].unsqueeze(0).to(device)
        future_treat = item['future_treatments'].unsqueeze(0).to(device)
        future_gt = item['future_outcomes'].numpy().flatten()
        
        # Factual Prediction
        with torch.no_grad():
            pred_fact = model.predict_counterfactual(history, future_treat).cpu().numpy()[0, :, 0]
        
        # Counterfactuals: Modify Insulin in Future Treatments
        # future_treat shape: [1, 36, 4] -> Index 0 is Insulin (in the reduced 4-dim tensor)
        # Wait, check dataset.py get_window:
        # future_treatments_indices = [1, 2, 7, 8] (Insulin, Carbs, Sin, Cos)
        # So Index 0 IS Insulin.
        
        # Scenario A: 0% Insulin
        cf_zero = future_treat.clone()
        cf_zero[:, :, 0] = 0.0 # Zero out insulin
        
        # Scenario B: 200% Insulin
        cf_double = future_treat.clone()
        cf_double[:, :, 0] *= 2.0 
        
        with torch.no_grad():
            pred_zero = model.predict_counterfactual(history, cf_zero).cpu().numpy()[0, :, 0]
            pred_double = model.predict_counterfactual(history, cf_double).cpu().numpy()[0, :, 0]
            
        # Plotting
        plt.figure(figsize=(10, 6))
        t = np.arange(len(pred_fact)) * 5 # Minutes
        
        # Plot Ground Truth
        plt.plot(t, denorm(future_gt), 'k-', label='Ground Truth', linewidth=2, alpha=0.5)
        
        # Plot Predictions
        plt.plot(t, denorm(pred_fact), 'b-', label='Model Factual (100% Dose)')
        plt.plot(t, denorm(pred_zero), 'r--', label='CF: 0% Dose')
        plt.plot(t, denorm(pred_double), 'g--', label='CF: 200% Dose')
        
        plt.title(f"Event {i+1}: Insulin Intervention Analysis")
        plt.xlabel("Minutes into Future")
        plt.ylabel("Glucose (mg/dL)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = f"analysis/plots/real_event_{i+1}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
