import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gaussian_kernel(x, y, sigma):
    """
    Gaussian (RBF) kernel: k(x,y) = exp(-||x-y||²/(2σ²))

    Args:
        x: [n, d]
        y: [m, d]
        sigma: kernel bandwidth

    Returns:
        K: [n, m] kernel matrix
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1)  # [n, 1, d]
    y = y.unsqueeze(0)  # [1, m, d]

    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)

    kernel_input = (tiled_x - tiled_y).pow(2).sum(2)
    return torch.exp(-kernel_input / (2 * sigma ** 2))


def compute_hsic_balance_loss(balanced_rep, treatments, sigma_rep=1.0, sigma_treat=1.0):
    """
    Hilbert-Schmidt Independence Criterion (HSIC) for covariate balance

    Measures statistical dependence between representations and treatments.
    HSIC = 0 iff representations and treatments are independent.
    Lower HSIC = better balance = less confounding

    HSIC(X,Y) = (1/(n-1)²) * trace(K_X @ H @ K_Y @ H)
    where H = I - (1/n)*11ᵀ is the centering matrix

    References:
    - Gretton et al. (2005): Kernel methods for measuring independence
    - Lopez-Paz et al. (2017): CEVAE uses HSIC for causal inference

    Args:
        balanced_rep: [batch, br_dim] - balanced representation
        treatments: [batch, treatment_dim] - actual treatments (continuous)
        sigma_rep: kernel bandwidth for representations
        sigma_treat: kernel bandwidth for treatments

    Returns:
        hsic: scalar - statistical dependence measure
    """
    n = balanced_rep.size(0)

    if n < 2:
        return torch.tensor(0.0, device=balanced_rep.device)

    # Compute kernel matrices
    K_rep = gaussian_kernel(balanced_rep, balanced_rep, sigma_rep)      # [n, n]
    K_treat = gaussian_kernel(treatments, treatments, sigma_treat)      # [n, n]

    # Centering matrix: H = I - (1/n)*11ᵀ
    H = torch.eye(n, device=balanced_rep.device) - torch.ones(n, n, device=balanced_rep.device) / n

    # HSIC = trace(K_rep @ H @ K_treat @ H) / (n-1)²
    # Efficient computation: trace(AB) = sum(A * Bᵀ)
    K_rep_centered = K_rep @ H
    K_treat_centered = K_treat @ H

    hsic = torch.sum(K_rep_centered * K_treat_centered.t()) / ((n - 1) ** 2)

    return hsic


class RelativePositionalEncoding(nn.Module):
    """Learnable relative positional encoding for time series"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        # Learnable relative position embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len = x.size(0), x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.pos_embedding(positions)


class PropensityNetwork(nn.Module):
    """Estimates propensity scores P(treatment | history) for IPW

    Uses transformer-encoded representations for full temporal context
    """
    def __init__(self, d_model, treatment_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, treatment_dim)
        )

    def forward(self, encoded_history):
        """
        Args:
            encoded_history: [batch, seq_len, d_model] - transformer encoding
        Returns:
            propensity: [batch, treatment_dim] - predicted treatment values
        """
        # Pool encoded sequence (use last timestep + mean for richer representation)
        last_state = encoded_history[:, -1, :]  # [batch, d_model]
        mean_state = torch.mean(encoded_history, dim=1)  # [batch, d_model]
        pooled = last_state + mean_state  # [batch, d_model]

        return self.network(pooled)  # [batch, treatment_dim]


class TreatmentCrossAttention(nn.Module):
    """Cross-attention module for treatment-aware outcome prediction"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, treatment_emb, history_emb):
        """
        Args:
            treatment_emb: [batch, treat_seq_len, d_model] - query
            history_emb: [batch, hist_seq_len, d_model] - key, value
        Returns:
            [batch, treat_seq_len, d_model]
        """
        # Treatment embeddings attend to history
        attn_out, _ = self.cross_attn(
            query=treatment_emb,
            key=history_emb,
            value=history_emb
        )
        # Residual + norm
        return self.norm(treatment_emb + self.dropout(attn_out))


class GlucoseTransformerCRN(nn.Module):
    """
    Three-subnetwork Causal Transformer with IPW:
    1. PropensityNetwork: P(treatment | history)
    2. EncoderNetwork: history -> balanced representation
    3. DecoderNetwork: (representation, future_treatments) -> outcomes
    """
    def __init__(self,
                 input_dim=9,
                 treatment_dim=6,
                 output_dim=1,
                 d_model=64,
                 nhead=4,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 dim_feedforward=128,
                 dropout=0.1,
                 br_size=32):  # kept for config compatibility but not used the same way
        super().__init__()

        self.d_model = d_model
        self.treatment_dim = treatment_dim

        # ===== 1. Propensity Network =====
        # Uses transformer encoder output (shares temporal modeling)
        self.propensity_net = PropensityNetwork(
            d_model=d_model,
            treatment_dim=treatment_dim,
            hidden_dim=dim_feedforward
        )

        # ===== 2. Encoder Network =====
        self.encoder_input_proj = nn.Linear(input_dim, d_model)
        self.encoder_pos_encoding = RelativePositionalEncoding(d_model)

        # Modern transformer with Pre-LN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Balanced Representation Head (for explicit covariate balancing)
        self.br_head = nn.Sequential(
            nn.Linear(d_model, br_size),
            nn.ELU()
        )
        self.br_size = br_size

        # ===== 3. Decoder Network =====
        # Treatment embedding
        self.treatment_proj = nn.Linear(treatment_dim, d_model)
        self.treatment_pos_encoding = RelativePositionalEncoding(d_model)

        # Treatment-aware cross-attention
        self.treatment_cross_attn = TreatmentCrossAttention(d_model, nhead, dropout)

        # Decoder layers with Pre-LN
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Output head
        self.output_head = nn.Linear(d_model, output_dim)

    def encode_history(self, history):
        """
        Encode history through encoder network
        Args:
            history: [batch, hist_len, input_dim]
        Returns:
            memory: [batch, hist_len, d_model]
        """
        x = self.encoder_input_proj(history)
        x = self.encoder_pos_encoding(x)
        memory = self.transformer_encoder(x)
        return memory

    def forward(self, history, future_treatments):
        """
        Forward pass for training
        Args:
            history: [batch, hist_len, input_dim]
            future_treatments: [batch, pred_len, treatment_dim]
        Returns:
            dict with 'pred_outcomes', 'propensity_scores', and 'balanced_rep'
        """
        # 1. Encode history (shared between propensity and outcome prediction)
        memory = self.encode_history(history)  # [batch, hist_len, d_model]

        # 2. Balanced representation for covariate balance
        # Pool encoder output: last state + mean state
        last_state = memory[:, -1, :]  # [batch, d_model]
        mean_state = torch.mean(memory, dim=1)  # [batch, d_model]
        pooled = last_state + mean_state  # [batch, d_model]
        balanced_rep = self.br_head(pooled)  # [batch, br_size]

        # 3. Propensity estimation using encoded history (full temporal context)
        propensity_scores = self.propensity_net(memory)  # [batch, treatment_dim]

        # 4. Embed future treatments
        treatment_emb = self.treatment_proj(future_treatments)  # [batch, pred_len, d_model]
        treatment_emb = self.treatment_pos_encoding(treatment_emb)

        # 5. Treatment-aware cross-attention
        treatment_context = self.treatment_cross_attn(treatment_emb, memory)

        # 6. Decode with treatment context as query and history as memory
        decoded = self.transformer_decoder(treatment_context, memory)

        # 7. Predict outcomes
        pred_outcomes = self.output_head(decoded)  # [batch, pred_len, output_dim]

        return {
            'pred_outcomes': pred_outcomes,
            'propensity_scores': propensity_scores,
            'balanced_rep': balanced_rep
        }

    def compute_propensity(self, history):
        """
        Compute propensity scores for IPW
        Args:
            history: [batch, hist_len, input_dim]
        Returns:
            [batch, treatment_dim]
        """
        memory = self.encode_history(history)
        return self.propensity_net(memory)

    def predict_counterfactual(self, history, future_treatments):
        """Inference mode"""
        with torch.no_grad():
            memory = self.encode_history(history)

            treatment_emb = self.treatment_proj(future_treatments)
            treatment_emb = self.treatment_pos_encoding(treatment_emb)

            treatment_context = self.treatment_cross_attn(treatment_emb, memory)
            decoded = self.transformer_decoder(treatment_context, memory)

            return self.output_head(decoded)
