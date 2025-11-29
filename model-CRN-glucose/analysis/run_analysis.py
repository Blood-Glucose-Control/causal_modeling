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
from src.data.generator import DiabetesAnalyzer
from src.utils.data_utils import get_data_splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--num_interventions', type=int, default=50, help='Number of interventions to evaluate')
    parser.add_argument('--config', type=str, default='config/best_params.yaml')
    args = parser.parse_args()
    
    # 1. Initialize Helper Classes
    print("Initializing Environment...")
    # We use get_data_splits to get a valid dataset/scaler pair
    # We only need the scaler and one dataset to access helper methods
    train_ds, _, _, scaler = get_data_splits(num_days=10) 
    dataset_helper = train_ds
    
    analyzer = DiabetesAnalyzer(seed=2024) # New seed for robust eval
    
    # 2. Load Model
    print("Loading Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load params from config if available to match training
    import yaml
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model_keys = ['d_model', 'nhead', 'num_encoder_layers', 'num_decoder_layers', 
                      'dim_feedforward', 'dropout', 'br_size']
        model_params = {k: config[k] for k in model_keys if k in config}
    else:
        model_params = {}

    model = GlucoseTransformerCRN(
        input_dim=9, treatment_dim=6, output_dim=1,
        **model_params
    ).to(device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    # 3. Generate Data
    print("Generating Evaluation Data...")
    # Generate enough days to find requested number of interventions
    base_df = analyzer.generate_patient_data(n_days=max(20, args.num_interventions // 2))
    interventions = analyzer.counterfactual_model.list_interventions(base_df)
    
    print(f"Found {len(interventions)} total interventions. Analyzing first {args.num_interventions}...")
    
    # 4. Quantitative Analysis Loop
    results = []
    dose_factors = [0.5, 0.8, 1.2, 1.5] # Test range of counterfactuals
    
    hist_len = dataset_helper.history_window
    
    # Helper for scaling
    glucose_mean = dataset_helper.scaler.mean_[0]
    glucose_std = dataset_helper.scaler.scale_[0]
    def denorm(x): return x * glucose_std + glucose_mean
    
    # Loop
    valid_count = 0
    pbar = tqdm(total=args.num_interventions)
    
    for i, intervention in enumerate(interventions):
        if valid_count >= args.num_interventions:
            break
            
        target_time = intervention['timestamp']
        
        # Validity Checks
        if target_time not in base_df.index: continue
        idx_loc = base_df.index.get_loc(target_time)
        if idx_loc < hist_len: continue
        
        # Prepare Factual History Input (Shared across all CFs)
        start_pos = idx_loc - hist_len
        feat_factual = dataset_helper.transform_new_data(base_df)
        input_factual = dataset_helper.get_window(feat_factual, start_pos)
        hist_tensor = input_factual['encoder_inputs'].unsqueeze(0).to(device)
        
        # Factual Prediction (Baseline Error)
        gt_factual = base_df['glucose'].iloc[idx_loc : idx_loc + 36].values
        fut_treat_fact = input_factual['future_treatments'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_fact_norm = model.predict_counterfactual(hist_tensor, fut_treat_fact).cpu().numpy()[0, :, 0]
        pred_fact = denorm(pred_fact_norm)
        rmse_factual = np.sqrt(np.mean((pred_fact - gt_factual)**2))
        
        # Iterate Counterfactual Factors
        for factor in dose_factors:
            # A. Generate Ground Truth Counterfactual (Simulator)
            cf_df_raw = analyzer.analyze_intervention(
                base_df, intervention['id'], analysis_type='dose', dose_factor=factor
            )
            
            # Extract GT Glucose
            gt_cf = cf_df_raw[f'cf1_glucose'].iloc[idx_loc : idx_loc + 36].values
            
            # B. Generate Model Counterfactual
            # We need the Modified Future Treatments Tensor
            # We construct a temp DF just to extract features correctly
            cf_world_df = cf_df_raw.copy()
            cf_world_df['glucose'] = cf_df_raw['cf1_glucose'] # Though model shouldn't see this
            cf_world_df['insulin'] = cf_df_raw['cf1_insulin'] # Crucial: The modified insulin
            cf_world_df['active_insulin'] = cf_df_raw['cf1_active_insulin']
            cf_world_df['carb_impact'] = cf_df_raw['cf1_carb_impact']
            
            feat_cf = dataset_helper.transform_new_data(cf_world_df)
            input_cf = dataset_helper.get_window(feat_cf, start_pos)
            fut_treat_cf = input_cf['future_treatments'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_cf_norm = model.predict_counterfactual(hist_tensor, fut_treat_cf).cpu().numpy()[0, :, 0]
            pred_cf = denorm(pred_cf_norm)
            
            # C. Calculate Error (RMSE)
            rmse_cf = np.sqrt(np.mean((pred_cf - gt_cf)**2))
            
            # D. Calculate Treatment Effect Error (ATE Error)
            true_effect = np.mean(gt_cf - gt_factual)
            pred_effect = np.mean(pred_cf - pred_fact)
            effect_error = abs(pred_effect - true_effect)
            
            results.append({
                'intervention_idx': i,
                'dose_original': intervention['dose'],
                'factor': factor,
                'dose_new': intervention['dose'] * factor,
                'rmse_factual': rmse_factual,
                'rmse_cf': rmse_cf,
                'true_effect_mean': true_effect,
                'pred_effect_mean': pred_effect,
                'effect_error': effect_error
            })
            
        valid_count += 1
        pbar.update(1)
        
    pbar.close()
    
    # 5. Summarize Results
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("COUNTERFACTUAL EVALUATION REPORT")
    print("="*50)
    
    print(f"\nOverall Factual RMSE: {df_res['rmse_factual'].mean():.2f} mg/dL")
    
    print("\nPerformance by Dose Factor:")
    summary = df_res.groupby('factor')[['rmse_cf', 'effect_error']].agg(['mean', 'std'])
    print(summary.round(2))
    
    # Save CSV
    os.makedirs('analysis/results', exist_ok=True)
    df_res.to_csv('analysis/results/cf_metrics.csv', index=False)
    print("\nDetailed metrics saved to analysis/results/cf_metrics.csv")

if __name__ == "__main__":
    main()
