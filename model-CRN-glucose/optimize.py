import os
import argparse
import optuna
import torch
import yaml
import numpy as np
from src.utils.data_utils import get_data_splits
from src.models.causal_transformer import GlucoseTransformerCRN
from src.training.trainer import CRNTrainer

def objective(trial):
    # Hyperparameters to optimize
    config = {
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'epochs': 10, # Keep low for speed during search
        'lambda_treatment': trial.suggest_float('lambda_treatment', 0.1, 5.0),
        'save_dir': 'checkpoints/optuna_temp'
    }
    
    # Model Architecture Hyperparams
    model_params = {
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
        'nhead': trial.suggest_categorical('nhead', [2, 4]),
        'num_encoder_layers': trial.suggest_int('num_encoder_layers', 1, 4),
        'num_decoder_layers': trial.suggest_int('num_decoder_layers', 1, 4),
        'dim_feedforward': trial.suggest_categorical('dim_feedforward', [64, 128, 256]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'br_size': trial.suggest_categorical('br_size', [16, 32, 64])
    }
    
    # Ensure d_model is divisible by nhead
    if model_params['d_model'] % model_params['nhead'] != 0:
        raise optuna.exceptions.TrialPruned()

    # Initialize Model
    model = GlucoseTransformerCRN(
        input_dim=9,
        treatment_dim=6,
        output_dim=1,
        **model_params
    )
    
    # Train
    # Re-init trainer with new batch size
    trainer = CRNTrainer(model, train_ds, val_ds, config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # Short Training Loop
    for epoch in range(config['epochs']):
        model.train()
        # Training ... (simplified for speed)
        for batch in trainer.train_loader:
            history = batch['encoder_inputs'].to(device)
            future_treat = batch['future_treatments'].to(device)
            future_y = batch['future_outcomes'].to(device)
            
            optimizer.zero_grad()
            
            # GRL Schedule
            p = epoch / config['epochs']
            alpha = 2. / (1. + np.exp(-10. * p)) - 1
            
            out = model(history, future_treat, alpha=alpha)
            loss = criterion(out['pred_outcomes'], future_y) + config['lambda_treatment'] * criterion(out['pred_treatment'], future_treat[:,0,:])
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in trainer.val_loader:
                history = batch['encoder_inputs'].to(device)
                future_treat = batch['future_treatments'].to(device)
                future_y = batch['future_outcomes'].to(device)
                
                # No alpha for val
                out = model(history, future_treat, alpha=0.0)
                val_loss += criterion(out['pred_outcomes'], future_y).item()
        
        val_loss /= len(trainer.val_loader)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_val_loss = min(best_val_loss, val_loss)
        
    return best_val_loss

if __name__ == "__main__":
    # Generate Data ONCE
    # 1 Year of data split 70/15/15
    global train_ds, val_ds, test_ds, scaler
    train_ds, val_ds, test_ds, scaler = get_data_splits(num_days=200) # 200 days for speed
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best params
    os.makedirs("config", exist_ok=True)
    with open("config/best_params.yaml", "w") as f:
        yaml.dump(trial.params, f)
