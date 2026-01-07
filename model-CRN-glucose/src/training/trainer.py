import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import logging

from src.models.causal_transformer import compute_hsic_balance_loss


def get_device():
    """Get best available device (MPS for Mac, CUDA for NVIDIA, CPU fallback)"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def compute_ipw_weights(propensity_scores, actual_treatments, epsilon=1e-6):
    """
    Compute Inverse Propensity Weights for continuous treatments

    Args:
        propensity_scores: [batch, treatment_dim] - predicted treatment values
        actual_treatments: [batch, treatment_dim] - actual treatment values
        epsilon: small constant for numerical stability

    Returns:
        weights: [batch] - IPW weights
    """
    # For continuous treatments, use Gaussian kernel for density estimation
    # Weight = 1 / P(T | X) where P is approximated by distance

    # Compute prediction error (proxy for inverse density)
    mse_per_sample = torch.mean((propensity_scores - actual_treatments) ** 2, dim=1)

    # Convert to weights: smaller error = higher propensity = lower weight
    # Use inverse of (1 + error) for stability
    weights = 1.0 / (1.0 + mse_per_sample + epsilon)

    # Normalize weights to have mean 1
    weights = weights / (weights.mean() + epsilon)

    # Clip extreme weights for stability
    weights = torch.clamp(weights, min=0.1, max=10.0)

    return weights


class CT_Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)

        print(f"Causal Transformer w/ IPW+Cross-Attn using device: {self.device}")

        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Separate optimizers for propensity network and main model
        self.optimizer_propensity = optim.Adam(
            self.model.propensity_net.parameters(),
            lr=config['lr']
        )
        self.optimizer_main = optim.Adam(
            [p for n, p in self.model.named_parameters() if 'propensity_net' not in n],
            lr=config['lr']
        )

        # Losses
        self.mse_loss = nn.MSELoss(reduction='none')  # Use 'none' for IPW weighting

    def train(self):
        best_val_loss = float('inf')
        patience = self.config.get('patience', 10)
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            self.model.train()

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")

            epoch_outcome_loss = 0
            epoch_propensity_loss = 0
            epoch_balance_loss = 0

            for batch in pbar:
                history = batch['encoder_inputs'].to(self.device)
                future_treat = batch['future_treatments'].to(self.device)
                future_y = batch['future_outcomes'].to(self.device)

                # ===== Step 1: Train Propensity Network =====
                self.optimizer_propensity.zero_grad()

                propensity_scores = self.model.compute_propensity(history)
                target_next_treatment = future_treat[:, 0, :]  # Predict immediate next treatment

                # Propensity loss (standard MSE)
                loss_propensity = torch.mean(self.mse_loss(propensity_scores, target_next_treatment))

                loss_propensity.backward()
                self.optimizer_propensity.step()

                # ===== Step 2: Train Main Model with IPW + Balance =====
                self.optimizer_main.zero_grad()

                # Forward pass
                outputs = self.model(history, future_treat)
                pred_y = outputs['pred_outcomes']
                balanced_rep = outputs['balanced_rep']

                # Compute IPW weights (detached propensity scores)
                with torch.no_grad():
                    propensity_scores_detached = self.model.compute_propensity(history)
                    ipw_weights = compute_ipw_weights(
                        propensity_scores_detached,
                        target_next_treatment
                    )

                # IPW-weighted outcome loss
                sample_losses = self.mse_loss(pred_y, future_y).mean(dim=(1, 2))  # [batch]
                loss_outcome = torch.mean(ipw_weights * sample_losses)

                # HSIC balance loss (treatment-invariant representations)
                # Measures statistical independence between reps and treatments
                lambda_balance = self.config.get('lambda_balance', 0.1)
                loss_balance = compute_hsic_balance_loss(balanced_rep, target_next_treatment)

                # Total loss: outcome + balance
                loss_total = loss_outcome + lambda_balance * loss_balance

                loss_total.backward()
                self.optimizer_main.step()

                # Track losses
                epoch_outcome_loss += loss_outcome.item()
                epoch_propensity_loss += loss_propensity.item()
                epoch_balance_loss += loss_balance.item()

                pbar.set_postfix({
                    'Outcome': f'{loss_outcome.item():.4f}',
                    'Propensity': f'{loss_propensity.item():.4f}',
                    'Balance': f'{loss_balance.item():.4f}',
                    'IPW_mean': f'{ipw_weights.mean().item():.3f}'
                })

            # Validation
            val_losses = self.validate()
            print(f"Epoch {epoch+1} - Val Outcome: {val_losses['outcome']:.4f} | Val Propensity: {val_losses['propensity']:.4f} | Val Balance: {val_losses['balance']:.4f}")
            print(f"           Train Outcome: {epoch_outcome_loss/len(self.train_loader):.4f} | Train Propensity: {epoch_propensity_loss/len(self.train_loader):.4f} | Train Balance: {epoch_balance_loss/len(self.train_loader):.4f}")

            if val_losses['outcome'] < best_val_loss:
                best_val_loss = val_losses['outcome']
                self.save_checkpoint('best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"EarlyStopping counter: {patience_counter} out of {patience}")
                if patience_counter >= patience:
                    print("Early stopping")
                    break

    def validate(self):
        self.model.eval()
        total_outcome_loss = 0
        total_propensity_loss = 0
        total_balance_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                history = batch['encoder_inputs'].to(self.device)
                future_treat = batch['future_treatments'].to(self.device)
                future_y = batch['future_outcomes'].to(self.device)

                # Outcome loss
                outputs = self.model(history, future_treat)
                loss_outcome = torch.mean(self.mse_loss(outputs['pred_outcomes'], future_y))
                total_outcome_loss += loss_outcome.item()

                # Propensity loss
                propensity_scores = self.model.compute_propensity(history)
                target_next_treatment = future_treat[:, 0, :]
                loss_propensity = torch.mean(self.mse_loss(propensity_scores, target_next_treatment))
                total_propensity_loss += loss_propensity.item()

                # Balance loss
                balanced_rep = outputs['balanced_rep']
                loss_balance = compute_hsic_balance_loss(balanced_rep, target_next_treatment)
                total_balance_loss += loss_balance.item()

        return {
            'outcome': total_outcome_loss / len(self.val_loader),
            'propensity': total_propensity_loss / len(self.val_loader),
            'balance': total_balance_loss / len(self.val_loader)
        }

    def save_checkpoint(self, filename):
        path = os.path.join(self.config.get('save_dir', '.'), filename)
        torch.save(self.model.state_dict(), path)
