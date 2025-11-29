import os
import argparse
import torch
import yaml
from src.utils.data_utils import get_data_splits
from src.models.crn_transformer import GlucoseTransformerCRN
from src.training.trainer import CRNTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/best_params.yaml')
    args = parser.parse_args()

    # Load Config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {args.config}")
    else:
        print(f"Config file {args.config} not found. Using defaults.")
        config = {
            'batch_size': 64,
            'lr': 1e-3,
            'epochs': 20,
            'lambda_treatment': 1.0,
            'save_dir': 'checkpoints',
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'br_size': 32
        }
    
    os.makedirs(config.get('save_dir', 'checkpoints'), exist_ok=True)
    
    print("Initializing Datasets...")
    # Use the utility that creates proper splits
    train_ds, val_ds, test_ds, scaler = get_data_splits(num_days=365)
    
    print("Initializing Model...")
    # Filter config for model params
    model_keys = ['d_model', 'nhead', 'num_encoder_layers', 'num_decoder_layers', 
                  'dim_feedforward', 'dropout', 'br_size']
    model_params = {k: config[k] for k in model_keys if k in config}
    
    # Input Dim = 9 (Glucose, Insulin, Carbs, Ex, Stress, ActIns, CarbImp, Sin, Cos)
    model = GlucoseTransformerCRN(
        input_dim=9,
        treatment_dim=6, 
        output_dim=1,
        **model_params
    )
    
    print(f"Model created with params: {model_params}")
    
    print("Starting Training...")
    trainer = CRNTrainer(model, train_ds, val_ds, config)
    trainer.train()

if __name__ == "__main__":
    main()
