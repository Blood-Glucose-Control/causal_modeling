import torch
import os
import numpy as np
from src.models.crn_transformer import GlucoseTransformerCRN
from src.data.dataset import GlucoseDataset
from src.utils.evaluation import CounterfactualEvaluator
import argparse

def modify_insulin(treatments, factor=1.2):
    """
    treatments: [Batch, Time, Features]
    Feature 0 is Insulin (based on dataset.py mapping)
    """
    treatments[:, :, 0] = treatments[:, :, 0] * factor
    return treatments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    args = parser.parse_args()
    
    # 1. Load Data (Val set)
    print("Loading Validation Data...")
    # We use a small num_days for quick evaluation
    val_ds = GlucoseDataset(mode='val', num_days=30)
    
    # 2. Load Model
    print("Loading Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Using same dimensions as main.py
    model = GlucoseTransformerCRN(
        input_dim=9,
        treatment_dim=6,
        output_dim=1
    ).to(device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using random weights.")

    evaluator = CounterfactualEvaluator(model, val_ds)
    
    # 3. Run Analysis on a few interesting samples
    # We look for samples where Insulin was actually administered in the future window
    print("Searching for scenarios with insulin events...")
    
    os.makedirs('plots', exist_ok=True)
    
    count = 0
    for i in range(len(val_ds)):
        sample = val_ds[i]
        future_insulin = sample['future_treatments'][:, 0] # Index 0 is insulin
        
        # If total insulin in the next 3 hours > 0.5 (normalized, roughly)
        if future_insulin.sum() > 0.5: 
            print(f"Found interesting scenario at index {i}")
            
            # Scenario A: What if we increased insulin by 50%?
            evaluator.visualize_scenario(
                i, 
                modification_func=lambda t: modify_insulin(t, factor=1.5),
                save_path=f"plots/scenario_{i}_more_insulin.png"
            )
            
            # Scenario B: What if we gave NO insulin?
            evaluator.visualize_scenario(
                i, 
                modification_func=lambda t: modify_insulin(t, factor=0.0),
                save_path=f"plots/scenario_{i}_no_insulin.png"
            )
            
            count += 1
            if count >= 5: break
            
    print("Evaluation complete. Check 'plots/' directory.")

if __name__ == "__main__":
    main()

