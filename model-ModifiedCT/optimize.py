import os
import argparse
import optuna
import torch
import yaml
import numpy as np
from src.utils.data_utils import get_data_splits
from src.models.causal_transformer import GlucoseTransformerCRN
from src.training.trainer import CT_Trainer

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization

    Optimizes:
    - Model architecture (d_model, layers, etc.)
    - Training hyperparams (lr, batch_size)
    - Balance loss weight (lambda_balance)
    """

    # Model Architecture Hyperparams
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])

    # Ensure d_model is divisible by nhead
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    model_params = {
        'd_model': d_model,
        'nhead': nhead,
        'num_encoder_layers': trial.suggest_int('num_encoder_layers', 1, 4),
        'num_decoder_layers': trial.suggest_int('num_decoder_layers', 1, 3),
        'dim_feedforward': trial.suggest_categorical('dim_feedforward', [64, 128, 256]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'br_size': trial.suggest_categorical('br_size', [16, 32, 64])
    }

    # Training Hyperparameters
    config = {
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'lambda_balance': trial.suggest_float('lambda_balance', 0.01, 1.0, log=True),
        'epochs': 20,  # Keep low for speed during search
        'patience': 5,  # Early stopping
        'save_dir': 'checkpoints/optuna_temp'
    }

    # Create temp directory for this trial
    os.makedirs(config['save_dir'], exist_ok=True)

    # Initialize Model
    model = GlucoseTransformerCRN(
        input_dim=9,  # glucose, insulin, carbs, exercise, stress, active_insulin, carb_impact, sin_hour, cos_hour
        treatment_dim=6,  # insulin, carbs, exercise, stress, sin_hour, cos_hour
        output_dim=1,  # glucose prediction
        **model_params
    )

    # Train using proper CT_Trainer (handles IPW + HSIC automatically)
    trainer = CT_Trainer(model, train_ds, val_ds, config)

    # Custom training loop for Optuna pruning
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Run one epoch of training (simplified - just validation for pruning)
        # We could call trainer.train() but we want per-epoch control for pruning

        # Manually run training for this epoch
        trainer.model.train()
        for batch in trainer.train_loader:
            history = batch['encoder_inputs'].to(trainer.device)
            future_treat = batch['future_treatments'].to(trainer.device)
            future_y = batch['future_outcomes'].to(trainer.device)

            # Train propensity network
            trainer.optimizer_propensity.zero_grad()
            propensity_scores = trainer.model.compute_propensity(history)
            target_next_treatment = future_treat[:, 0, :]
            loss_propensity = torch.mean(trainer.mse_loss(propensity_scores, target_next_treatment))
            loss_propensity.backward()
            trainer.optimizer_propensity.step()

            # Train main model
            trainer.optimizer_main.zero_grad()
            outputs = trainer.model(history, future_treat)
            pred_y = outputs['pred_outcomes']
            balanced_rep = outputs['balanced_rep']

            # IPW-weighted outcome loss
            with torch.no_grad():
                from src.training.trainer import compute_ipw_weights
                propensity_scores_detached = trainer.model.compute_propensity(history)
                ipw_weights = compute_ipw_weights(propensity_scores_detached, target_next_treatment)

            sample_losses = trainer.mse_loss(pred_y, future_y).mean(dim=(1, 2))
            loss_outcome = torch.mean(ipw_weights * sample_losses)

            # HSIC balance loss
            from src.models.causal_transformer import compute_hsic_balance_loss
            loss_balance = compute_hsic_balance_loss(balanced_rep, target_next_treatment)

            # Total loss
            loss_total = loss_outcome + config['lambda_balance'] * loss_balance
            loss_total.backward()
            trainer.optimizer_main.step()

        # Validation
        val_loss = trainer.validate()

        # Report to Optuna for pruning
        trial.report(val_loss, epoch)

        # Prune if needed
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break  # Early stopping

    return best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for Glucose CRN")
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials')
    parser.add_argument('--num_days', type=int, default=180,
                       help='Number of days of data to generate (smaller = faster)')
    parser.add_argument('--study_name', type=str, default='glucose_crn_optimization',
                       help='Name for Optuna study')
    parser.add_argument('--output', type=str, default='config/best_params_glucose.yaml',
                       help='Output file for best parameters')
    args = parser.parse_args()

    print("="*80)
    print("GLUCOSE CRN HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Trials: {args.n_trials}")
    print(f"Data: {args.num_days} days")
    print(f"Study: {args.study_name}")
    print("="*80)

    # Generate Data ONCE (shared across all trials)
    print("\nGenerating glucose data...")
    global train_ds, val_ds, test_ds, scaler
    train_ds, val_ds, test_ds, scaler = get_data_splits(num_days=args.num_days)

    print(f"Train: {len(train_ds)} windows")
    print(f"Val: {len(val_ds)} windows")
    print(f"Test: {len(test_ds)} windows")

    # Create Optuna study
    print(f"\nStarting optimization with {args.n_trials} trials...")
    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )

    # Optimize
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest trial (#{study.best_trial.number}):")
    print(f"  Validation Loss: {study.best_trial.value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Save best params to YAML
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Add some fixed params that aren't optimized
    best_params = study.best_trial.params.copy()
    best_params['epochs'] = 100  # Use more epochs for final training
    best_params['patience'] = 15
    best_params['save_dir'] = 'checkpoints/glucose'

    with open(args.output, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Best hyperparameters saved to: {args.output}")
    print(f"\nTo use these hyperparameters for training:")
    print(f"  uv run python train_glucose.py --config {args.output}")
    print("="*80)
