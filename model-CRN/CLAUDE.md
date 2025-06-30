# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements the Counterfactual Recurrent Network (CRN), a causal inference method for estimating treatment effects over time from observational data. The model was published at ICLR 2020 and uses adversarially balanced representations to handle time-dependent confounding in sequential treatment assignment scenarios.

## Core Architecture

The CRN consists of two main components:

1. **Encoder Network** (`CRN_model.py`): Builds treatment-invariant representations of patient history using LSTM cells with dropout and gradient reversal for domain adaptation
2. **Decoder Network** (`CRN_model.py`): Estimates outcomes under intended treatment sequences while updating balanced representations

Key architectural elements:
- Uses TensorFlow 1.15 with compatibility mode for TF 2.x
- Implements gradient reversal via `utils/flip_gradient.py` for adversarial training
- Cancer simulation data generation in `utils/cancer_simulation.py`
- Evaluation utilities in `utils/evaluation_utils.py`

## Commands

### Training and Evaluation
```bash
# Basic training with default hyperparameters
python test_crn.py --chemo_coeff=2 --radio_coeff=2 --model_name=crn_test_2

# Training with hyperparameter optimization (takes ~8 hours on GPU)
python test_crn.py --chemo_coeff=2 --radio_coeff=2 --model_name=crn_test_2 --b_encoder_hyperparm_tuning=True --b_decoder_hyperparm_tuning=True

# Custom results directory
python test_crn.py --results_dir=custom_results --model_name=my_model
```

### Environment Setup
```bash
# Install dependencies (Python 3.6 required)
pip install -r requirements.txt

# Note: Requires tensorflow-gpu==1.15.0 for optimal performance
```

## Key Parameters

- `chemo_coeff` / `radio_coeff`: Control time-dependent confounding strength (1-5 range)
- `model_name`: Used for saving trained models and hyperparameters
- `b_encoder_hyperparm_tuning` / `b_decoder_hyperparm_tuning`: Enable hyperparameter optimization

## Data and Results

- Synthetic datasets are generated on-the-fly (>1GB each) using pharmacokinetic-pharmacodynamic tumor growth simulation
- Models saved to `results/crn_models/` directory
- Hyperparameters saved as text files in results directory
- Outputs RMSE for both one-step-ahead and five-step-ahead counterfactual prediction

## Development Notes

- The codebase uses TensorFlow 1.x patterns with `tf.compat.v1` for TF 2.x compatibility
- Model training involves two phases: encoder training followed by decoder training
- Hyperparameter optimization uses random search over 50 (encoder) and 30 (decoder) simulations
- No traditional unit tests - evaluation is done through synthetic data experiments