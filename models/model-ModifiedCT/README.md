# Causal Transformer for Glucose Prediction

A three-subnetwork transformer architecture for counterfactual glucose prediction using Inverse Propensity Weighting (IPW).

## Architecture Overview

### Three-Subnetwork Design

```
┌─────────────────────┐
│ Propensity Network  │  Estimates P(treatment | history)
└──────────┬──────────┘
           │ (IPW weights)
           ↓
┌─────────────────────┐
│  Encoder Network    │  Processes history → balanced representation
│  + Pre-LN           │  Modern transformer with Pre-Layer Normalization
│  + Relative PE      │  Learnable relative positional encoding
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Decoder Network    │  Predicts outcomes from representation + treatments
│  + Treatment Attn   │  Cross-attention over treatment sequences
└─────────────────────┘
```

### Key Features

1. **Inverse Propensity Weighting (IPW)**: Classical causal inference approach for treatment balancing
   - More stable than Gradient Reversal
   - Interpretable propensity scores
   - Separate propensity network prevents gradient conflicts

2. **Modern Transformer Components**:
   - Pre-Layer Normalization (`norm_first=True`) for training stability
   - Relative positional encoding for better time series modeling
   - Treatment-aware cross-attention for context-dependent predictions

3. **Separate Optimization**:
   - Propensity network trained independently
   - Main model uses IPW-weighted losses
   - No gradient reversal tricks

## Setup

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

## Training

To train the model:

```bash
uv run python main.py
```

Or with custom config:

```bash
uv run python main.py --config config/my_config.yaml
```

## Loss Functions

### Propensity Loss
```python
loss_propensity = MSE(P(T|X), T_actual)
```

### IPW-Weighted Outcome Loss
```python
weights = compute_ipw(propensity_scores, actual_treatments)
loss_outcome = mean(weights * MSE(pred_Y, Y_actual))
```

## Citation

This architecture combines insights from:
- Classical causal inference (IPW, propensity scores)
- Modern transformer architectures (Pre-LN, relative PE)
- Counterfactual prediction literature (Causal Transformer, counterfactual Recurrent Network)