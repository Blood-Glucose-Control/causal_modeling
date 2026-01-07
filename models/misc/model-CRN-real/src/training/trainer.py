import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import logging

class CRNTrainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        # Losses
        self.mse_loss = nn.MSELoss()
        
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            
            # Lambda schedule for GRL (slowly increase adversarial power)
            p = epoch / self.config['epochs']
            alpha = 2. / (1. + np.exp(-10. * p)) - 1
            
            total_loss = 0
            total_outcome_loss = 0
            total_treat_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            
            for batch in pbar:
                history = batch['encoder_inputs'].to(self.device)
                future_treat = batch['future_treatments'].to(self.device)
                future_y = batch['future_outcomes'].to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(history, future_treat, alpha=alpha)
                
                pred_y = outputs['pred_outcomes']
                pred_t = outputs['pred_treatment']
                
                # Outcome Loss (Standard MSE on future glucose)
                loss_outcome = self.mse_loss(pred_y, future_y)
                
                # Treatment Loss (Adversarial)
                # We want to predict the *immediate next* treatment based on history
                # "future_treat" shape is [B, Horizon, TreatDim]. 
                # We take the first step [:, 0, :] as the "next action" the patient took.
                target_next_t = future_treat[:, 0, :] 
                loss_treatment = self.mse_loss(pred_t, target_next_t)
                
                # Total Loss
                # We minimize Outcome Error AND Minimize Treatment Prediction Error 
                # (The GRL inside the model flips the gradient for the Encoder, 
                # so the Encoder actually *Maximizes* this error effectively)
                loss = loss_outcome + self.config['lambda_treatment'] * loss_treatment
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_outcome_loss += loss_outcome.item()
                total_treat_loss += loss_treatment.item()
                
                pbar.set_postfix({'Outcome': loss_outcome.item(), 'Treat': loss_treatment.item()})
                
            # Validation
            val_loss = self.validate()
            print(f"Epoch {epoch+1} Val MSE: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                
    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                history = batch['encoder_inputs'].to(self.device)
                future_treat = batch['future_treatments'].to(self.device)
                future_y = batch['future_outcomes'].to(self.device)
                
                outputs = self.model(history, future_treat, alpha=0.0) # Alpha 0 disabling GRL impact if any
                loss = self.mse_loss(outputs['pred_outcomes'], future_y)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, filename):
        path = os.path.join(self.config['save_dir'], filename)
        torch.save(self.model.state_dict(), path)

