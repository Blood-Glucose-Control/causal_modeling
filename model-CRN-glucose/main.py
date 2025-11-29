import os
import argparse
import torch
import yaml
from src.data.dataset import GlucoseDataset
from src.models.crn_transformer import GlucoseTransformerCRN
from src.training.trainer import CRNTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
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
    
    print("Initializing Datasets...")
    # Generate smaller datasets for testing
    train_ds = GlucoseDataset(mode='train', num_days=100)
    val_ds = GlucoseDataset(mode='val', num_days=20)
    
    print("Initializing Model...")
    # Input Dim = 9 (Glucose, Insulin, Carbs, Ex, Stress, ActIns, CarbImp, Sin, Cos)
    model = GlucoseTransformerCRN(
        input_dim=9,
        treatment_dim=6, 
        output_dim=1
    )
    
    print("Starting Training...")
    trainer = CRNTrainer(model, train_ds, val_ds, config)
    trainer.train()

if __name__ == "__main__":
    main()

