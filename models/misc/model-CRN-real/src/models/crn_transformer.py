import torch
import torch.nn as nn
import math
from .layers import GradientReversalLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class GlucoseTransformerCRN(nn.Module):
    def __init__(self, 
                 input_dim=9,
                 treatment_dim=6, # Insulin, Carbs, Exercise, Stress, SinTime, CosTime
                 output_dim=1,    # Glucose
                 d_model=64,
                 nhead=4,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 dim_feedforward=128,
                 dropout=0.1,
                 br_size=32):     # Balanced Representation Size
        super().__init__()
        
        self.d_model = d_model
        
        # --- Encoder (History Processing) ---
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=dim_feedforward, 
                                                 dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # --- Balancing Representation ---
        # Compress the full history context into a fixed-size representation
        # In original CRN this was the LSTM hidden state. Here we pool the transformer output.
        self.br_head = nn.Sequential(
            nn.Linear(d_model, br_size),
            nn.ELU()
        )
        
        # --- Treatment Prediction (Adversarial Head) ---
        # Predicts the NEXT treatment given the history
        self.grl = GradientReversalLayer()
        self.treatment_predictor = nn.Sequential(
            nn.Linear(br_size, dim_feedforward),
            nn.ELU(),
            nn.Linear(dim_feedforward, treatment_dim) # Predicting continuous treatments
        )
        
        # --- Decoder (Outcome Prediction) ---
        self.treatment_proj = nn.Linear(treatment_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_head = nn.Linear(d_model, output_dim)

    def get_balancing_representation(self, history):
        """
        history: [batch, hist_len, input_dim]
        returns: [batch, br_size]
        """
        # 1. Embed & Encode
        x = self.input_proj(history)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x) # [batch, hist_len, d_model]
        
        # 2. Pooling Strategy:
        # Combine the last state (immediate context) with mean of all states (global context)
        last_state = memory[:, -1, :]
        mean_state = torch.mean(memory, dim=1)
        combined = last_state + mean_state
        
        # 3. Project to Balancing Rep
        br = self.br_head(combined)
        return br, memory

    def forward(self, history, future_treatments, alpha=1.0):
        """
        history: [batch, hist_len, input_dim]
        future_treatments: [batch, pred_len, treatment_dim]
        """
        
        # --- Encoder Pass ---
        br, memory = self.get_balancing_representation(history)
        
        # --- Adversarial Pass (Treatment Prediction) ---
        # GRL applied to the Balancing Representation
        br_reversed = self.grl(br, alpha)
        pred_next_treatment = self.treatment_predictor(br_reversed)
        
        # --- Decoder Pass (Outcome Prediction) ---
        # Embed future treatments (the "query" for the decoder)
        tgt = self.treatment_proj(future_treatments)
        tgt = self.pos_decoder(tgt)
        
        # Decode
        # Memory is the keys/values from encoder
        decoded = self.transformer_decoder(tgt, memory)
        
        # Predict Glucose
        pred_outcomes = self.output_head(decoded)
        
        return {
            'pred_outcomes': pred_outcomes,       # [batch, pred_len, 1]
            'pred_treatment': pred_next_treatment # [batch, treatment_dim] (Only predicts step t+1 ideally, or avg?)
        }
        
    def predict_counterfactual(self, history, future_treatments):
        """Inference mode without GRL"""
        with torch.no_grad():
            # We don't care about the adversarial head during inference
            # Just encode -> decode
            x = self.input_proj(history)
            x = self.pos_encoder(x)
            memory = self.transformer_encoder(x)
            
            tgt = self.treatment_proj(future_treatments)
            tgt = self.pos_decoder(tgt)
            decoded = self.transformer_decoder(tgt, memory)
            return self.output_head(decoded)

