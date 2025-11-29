# Model CRN Glucose

This project implements a Counterfactual Recurrent Network (CRN) using a Transformer architecture to estimate counterfactual glucose outcomes. It adapts the original CRN concept (gradient reversal for balanced representations) to continuous glucose monitoring data.

## Structure

- `src/data/`: Data generation and PyTorch dataset wrappers.
- `src/models/`: Transformer-based CRN implementation with Gradient Reversal.
- `src/training/`: Training loops.

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

## Architecture

The model consists of:
1.  **Encoder**: Processes 12 hours of history (Glucose, Insulin, Carbs, etc.).
2.  **Balancing Head**: Compresses history into a representation invariant to the next treatment.
3.  **Adversarial Head**: Tries to predict the next treatment from the representation (Gradient Reversal ensures the representation *fails* at this).
4.  **Decoder**: Predicts the next 3 hours of glucose given the balanced representation and future treatments.

