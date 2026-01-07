import os
import argparse
import torch
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data.dataset import GlucoseDataset
from src.models.crn_transformer import GlucoseTransformerCRN
from src.training.trainer import CRNTrainer
from src.data.prep_real_data import load_and_validate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--raw_data', type=str, default='../output/merged_health_data.csv')
    parser.add_argument('--proc_data', type=str, default='../output/processed_training_data.csv')
    args = parser.parse_args()

    # Default Config
    config = {
        'batch_size': 64,
        'lr': 1e-3,
        'epochs': 20,
        'lambda_treatment': 1.0,
        'save_dir': 'checkpoints'
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print("=== 1. Data Pipeline ===")
    if os.path.exists(args.proc_data):
        print(f"Loading pre-processed data from {args.proc_data}...")
        full_df = pd.read_csv(args.proc_data)
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        full_df = full_df.set_index('timestamp')
    else:
        print(f"Processing raw data from {args.raw_data}...")
        try:
            full_df = load_and_validate(args.raw_data)
            full_df.to_csv(args.proc_data)
            print(f"Saved processed data to {args.proc_data}")
        except Exception as e:
            print(f"CRITICAL ERROR in Data Prep: {e}")
            return

    print(f"   Total rows: {len(full_df)}")
    print(f"   Columns: {len(full_df.columns)} columns verified.")

    # Train/Val Split (80/20 by time)
    train_size = int(0.8 * len(full_df))
    train_df = full_df.iloc[:train_size].copy()
    val_df = full_df.iloc[train_size:].copy()
    
    print(f"   Train size: {len(train_df)}")
    print(f"   Val size:   {len(val_df)}")

    print("=== 2. Initializing Datasets ===")
    
    # Fit Scaler ONLY on training data
    print("   Fitting StandardScaler...")
    train_features = GlucoseDataset.extract_features(train_df)
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    train_ds = GlucoseDataset(train_df, scaler=scaler)
    val_ds = GlucoseDataset(val_df, scaler=scaler)
    
    print("=== 3. Initializing Model ===")
    # Input Dim = 12 (Glucose, Insulin, Carbs, Ex, Stress, ActIns, CarbImp, Sin, Cos, HRV, Sleep, SPO2)
    # Treatment Dim = 4 (Insulin, Carbs, Sin, Cos)
    model = GlucoseTransformerCRN(
        input_dim=12,
        treatment_dim=4, 
        output_dim=1
    )
    
    print("=== 4. Starting Training ===")
    trainer = CRNTrainer(model, train_ds, val_ds, config)
    trainer.train()

if __name__ == "__main__":
    main()
