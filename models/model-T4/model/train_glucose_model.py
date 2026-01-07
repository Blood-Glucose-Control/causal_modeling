import argparse
import os
import logging
import time
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import load_glucose_data
from seq2seq import Encoder, AttentionDecoder, Seq2Seq
from counterfactual_sim import simulate_counterfactual, evaluate_glucose_control, plot_counterfactual_glucose

# Set up logging
timestamp = datetime.now().strftime('%Y-%m%d_%H%M%S')
date, hour = timestamp.split('_')
os.makedirs(f'log/{date}', exist_ok=True)
os.makedirs(f'checkpoints/{date}', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train glucose counterfactual model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/ml_dataset.csv',
                       help='Path to glucose dataset CSV')
    parser.add_argument('--window_size', type=int, default=12,
                       help='History window size (12 = 1 hour at 5min intervals)')
    parser.add_argument('--pred_horizon', type=int, default=36,
                       help='Prediction horizon (36 = 3 hours at 5min intervals)')
    
    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=32,
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--layer_num', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--vital_num', type=int, default=11,
                       help='Number of vital (feature) dimensions')
    parser.add_argument('--demo_dim', type=int, default=4,
                       help='Number of static (demographic) dimensions')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output parameters
    parser.add_argument('--log_file', type=str, default=f'log/{date}/{hour}_glucose.log',
                       help='Log file path')
    parser.add_argument('--save_model', type=str, default=f'checkpoints/{date}/{hour}_glucose.pt',
                       help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--cf_samples', type=int, default=5,
                       help='Number of counterfactual samples to visualize')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(model, train_dataset, valid_dataset, args, logger):
    """Train the model on glucose data"""
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Define loss function and optimizer
    criterion = F.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    log_interval = 10
    start_time = time.time()
    best_valid_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_ps_loss = 0.0
        
        for idx, (x, x_demo, treatment, treatment_cf, y, death, mask) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Forward pass through model
            output_factual, output_counterfactual, _, ps_output, _ = model(
                x, y, x_demo, treatment, teacher_forcing_ratio=0.7
            )
            
            # Calculate glucose prediction loss (factual)
            loss = criterion(output_factual, y)
            
            # Add propensity score loss (for dosage prediction)
            treatment_target = treatment[:, x.shape[1]:x.shape[1] + y.shape[1]]
            ps_loss = F.binary_cross_entropy_with_logits(ps_output.reshape(-1), 
                                                         treatment_target.reshape(-1))
            
            # Combine losses
            combined_loss = loss + 0.1 * ps_loss
            
            # Backpropagation
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_ps_loss += ps_loss.item()
            
            if idx % log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(f'| epoch {epoch:3d} | {idx:5d}/{len(train_dataloader):5d} batches '
                           f'| loss {loss.item():.4f} | ps_loss {ps_loss.item():.4f}')
                start_time = time.time()
        
        # Validation phase
        valid_loss = evaluate(model, valid_dataloader, criterion, args)
        
        # Update learning rate
        scheduler.step(valid_loss)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            logger.info('Best model. Saving...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'loss': valid_loss
            }, args.save_model)
        
        # Log epoch results
        logger.info('-' * 59)
        logger.info(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:.2f}s | '
                   f'valid loss {valid_loss:.4f} | train loss {train_loss/len(train_dataloader):.4f}')
        logger.info('-' * 59)
        
    logger.info(f'Training completed. Best validation loss: {best_valid_loss:.4f}')
    return best_valid_loss

def evaluate(model, dataloader, criterion, args):
    """Evaluate the model on validation data"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, x_demo, treatment, treatment_cf, y, death, mask in dataloader:
            # Forward pass through model
            output_factual, _, _, _, _ = model(
                x, y, x_demo, treatment, teacher_forcing_ratio=0
            )
            
            # Calculate loss
            loss = criterion(output_factual, y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def visualize_counterfactuals(model, test_dataset, args, logger):
    """Generate and visualize counterfactual examples"""
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Get a batch of data
    x, x_demo, treatment, treatment_cf, y, death, mask = next(iter(test_dataloader))
    
    # Number of samples to visualize
    num_samples = min(args.cf_samples, x.shape[0])
    
    for i in range(num_samples):
        # Extract single sample
        x_i = x[i:i+1]
        x_demo_i = x_demo[i:i+1]
        treatment_i = treatment[i:i+1]
        mask_i = mask[i:i+1]
        
        # Define counterfactual treatments
        # 1. Increase dose by 20%
        new_treatment_1 = treatment_i.clone()
        dose_mask = new_treatment_1 > 0
        new_treatment_1[dose_mask] = new_treatment_1[dose_mask] * 1.2
        new_treatment_1 = torch.clamp(new_treatment_1, 0, 1)
        
        # 2. Decrease dose by 20%
        new_treatment_2 = treatment_i.clone()
        new_treatment_2[dose_mask] = new_treatment_2[dose_mask] * 0.8
        
        # 3. Change timing (earlier by 15 min)
        timing_shift = -15  # minutes
        new_treatment_3 = torch.stack([
            treatment_i.squeeze(),
            torch.ones_like(treatment_i.squeeze()) * timing_shift
        ], dim=-1)
        
        # 4. Change timing (later by 15 min)
        timing_shift = 15  # minutes
        new_treatment_4 = torch.stack([
            treatment_i.squeeze(),
            torch.ones_like(treatment_i.squeeze()) * timing_shift
        ], dim=-1)
        
        # Run counterfactual simulations
        # Original vs increased dose
        results_1 = simulate_counterfactual(
            model, x_i, x_demo_i, treatment_i, new_treatment_1, mask_i, args.device
        )
        
        # Original vs decreased dose
        results_2 = simulate_counterfactual(
            model, x_i, x_demo_i, treatment_i, new_treatment_2, mask_i, args.device
        )
        
        # Original vs earlier timing
        results_3 = simulate_counterfactual(
            model, x_i, x_demo_i, treatment_i, new_treatment_3, mask_i, args.device
        )
        
        # Original vs later timing
        results_4 = simulate_counterfactual(
            model, x_i, x_demo_i, treatment_i, new_treatment_4, mask_i, args.device
        )
        
        # Create directory for visualizations
        os.makedirs(f'visualizations/{date}', exist_ok=True)
        
        # Plot the counterfactuals
        # Dose increase
        fig1 = plot_counterfactual_glucose(
            results_1['factual_trajectories'][0],
            results_1['counterfactual_trajectories'][0],
            treatment_i.squeeze().cpu().numpy(),
            new_treatment_1.squeeze().cpu().numpy()
        )
        fig1.savefig(f'visualizations/{date}/sample_{i}_dose_increase.png')
        
        # Dose decrease
        fig2 = plot_counterfactual_glucose(
            results_2['factual_trajectories'][0],
            results_2['counterfactual_trajectories'][0],
            treatment_i.squeeze().cpu().numpy(),
            new_treatment_2.squeeze().cpu().numpy()
        )
        fig2.savefig(f'visualizations/{date}/sample_{i}_dose_decrease.png')
        
        # Earlier timing
        fig3 = plot_counterfactual_glucose(
            results_3['factual_trajectories'][0],
            results_3['counterfactual_trajectories'][0],
            treatment_i.squeeze().cpu().numpy(),
            treatment_i.squeeze().cpu().numpy(),  # Original dosage shown
            target_range=(-0.5, 0.5)
        )
        fig3.suptitle(f'Earlier Insulin Timing (-15 min)', fontsize=16)
        fig3.savefig(f'visualizations/{date}/sample_{i}_earlier_timing.png')
        
        # Later timing
        fig4 = plot_counterfactual_glucose(
            results_4['factual_trajectories'][0],
            results_4['counterfactual_trajectories'][0],
            treatment_i.squeeze().cpu().numpy(),
            treatment_i.squeeze().cpu().numpy(),  # Original dosage shown
            target_range=(-0.5, 0.5)
        )
        fig4.suptitle(f'Later Insulin Timing (+15 min)', fontsize=16)
        fig4.savefig(f'visualizations/{date}/sample_{i}_later_timing.png')
        
        plt.close('all')
        
    logger.info(f'Counterfactual visualizations saved to visualizations/{date}/')

def main():
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        filename=args.log_file,
        filemode='w',
        datefmt='%m/%d/%Y %I:%M:%S')
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    console.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.addHandler(console)
    
    logger.info(args)
    
    # Set random seed
    set_seed(args.seed)
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load and process glucose data
    features, dataset = load_glucose_data(
        args.data_path, args, device, logger, 
        window_size=args.window_size, 
        prediction_horizon=args.pred_horizon
    )
    
    # Update vital_num based on actual features
    args.vital_num = len(features)
    
    # Split dataset
    num_train = int(len(dataset) * 0.7)
    num_val = int(len(dataset) * 0.15)
    num_test = len(dataset) - num_train - num_val
    
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [num_train, num_val, num_test]
    )
    
    logger.info(f"Dataset split: {num_train} train, {num_val} validation, {num_test} test samples")
    
    # Initialize model
    encoder = Encoder(
        input_dim=args.vital_num,
        output_dim=1,
        x_static_size=args.demo_dim,
        emb_dim=args.emb_dim,
        hid_dim=args.hidden_dim,
        n_layers=args.layer_num,
        dropout=args.dropout,
        device=device
    )
    
    decoder = AttentionDecoder(
        output_dim=1,
        x_static_size=args.demo_dim,
        emb_dim=args.emb_dim,
        hid_dim=args.hidden_dim,
        n_layers=args.layer_num,
        dropout=args.dropout,
        treatment_dim=2  # Support for both dosage and timing
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    logger.info(f"Initialized model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    logger.info("Starting training...")
    train(model, train_dataset, valid_dataset, args, logger)
    
    # Load best model for evaluation
    checkpoint = torch.load(args.save_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test model
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_loss = evaluate(model, test_dataloader, F.mse_loss, args)
    logger.info(f"Test loss: {test_loss:.4f}")
    
    # Generate counterfactual visualizations
    logger.info("Generating counterfactual visualizations...")
    visualize_counterfactuals(model, test_dataset, args, logger)
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 