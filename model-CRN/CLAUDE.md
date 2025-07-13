# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

This project uses `uv` for Python environment management. Install dependencies and activate virtual environment:

```bash
uv sync
source .venv/bin/activate
```

The project requires Python ≥3.7 and key dependencies:
- TensorFlow GPU 1.15.0 (uses tf.compat.v1 for compatibility)
- NumPy 1.18.2
- Pandas 1.0.4
- SciPy 1.1.0
- scikit-learn 0.22.2

## Project Context

This codebase implements the **Counterfactual Recurrent Network (CRN)** from the ICLR 2020 paper "Estimating counterfactual treatment outcomes over time through adversarially balanced representations". The model was originally designed for cancer treatment planning with one-hot key encoded treatments, but it in the process of being adapted for **diabetes modeling** where the treatment is either insulin dose timing relative to meal, or it's insulin dosage strength, and the outcome is blood glucose over time, with relevant confounders in the time-series data being de-confounded. The training data will consist of historical data per patient, and evaluation on each training set will be done on the patient itself. I.e. the purpose is a personalized model per patient per their own history and data.

## Common Commands

### Training and Testing
```bash
# Basic test with default hyperparameters
uv run python test_crn.py --chemo_coeff=2 --radio_coeff=2 --model_name=crn_test_2

# Run with hyperparameter tuning (takes ~8 hours on GPU)
uv run python test_crn.py --chemo_coeff=2 --radio_coeff=2 --model_name=crn_test_2 --b_encoder_hyperparm_tuning=True --b_decoder_hyperparm_tuning=True
```

### Evaluation
```bash
# Evaluate encoder separately
uv run python CRN_encoder_evaluate.py

# Evaluate decoder separately
uv run python CRN_decoder_evaluate.py
```

## Architecture Overview

The CRN implements a causal inference method for estimating treatment effects over time from observational data using **adversarially balanced representations**.

### Core Innovation

The key innovation is using **domain adversarial training** to handle time-dependent confounding instead of traditional inverse probability weighting (IPTW). This creates treatment-invariant representations that break the association between patient history and treatment assignments.

### Two-Phase Architecture

1. **Encoder Phase**: 
   - Builds treatment-invariant representations of patient history using LSTM
   - Uses adversarial training with gradient reversal layer (`utils/flip_gradient.py`)
   - Maximizes treatment classifier loss while minimizing outcome prediction loss
   - Creates balanced representations: `P(Φ(H_t)|A_t=A_1) = ... = P(Φ(H_t)|A_t=A_K)`

2. **Decoder Phase**:
   - Initialized with encoder's balanced representations
   - Predicts counterfactual outcomes for sequences of future treatments
   - Auto-regressive prediction for treatment planning scenarios

### Key Components

- **CRN_model.py**: Main model with encoder/decoder architecture
- **Domain Adversarial Training**: Uses gradient reversal to create treatment-invariant representations
- **Sequence-to-Sequence**: Handles variable-length patient histories and future treatment sequences
- **Cancer Simulation** (`utils/cancer_simulation.py`): Generates synthetic data with controllable confounding for the cancer usecase

### Medical Applications

The model addresses critical clinical questions:
- **Treatment Selection**: Which treatment to give at each timestep
- **Treatment Timing**: When to start/stop treatments
- **Treatment Sequencing**: Optimal sequences of multiple treatments over time

### Key Parameters

- `chemo_coeff`/`radio_coeff`: Control time-dependent confounding strength (1-5)
- `br_size`: Balanced representation dimensionality
- `lambda`: Trade-off between domain discrimination and outcome prediction
- `max_sequence_length`: Maximum timesteps for sequence modeling

### Data Generation

The cancer simulation model generates >1GB synthetic datasets per configuration with:
- Realistic pharmacokinetic-pharmacodynamic tumor growth dynamics
- Configurable selection bias and time-dependent confounding
- Ground truth counterfactuals for evaluation

### Model Persistence

- Trained models saved in `results/crn_models/` as TensorFlow checkpoints
- Hyperparameters logged in `results/` as text files
- Separate encoder/decoder model files with naming convention: `{encoder/decoder}_{model_name}_final.ckpt.*`

## Treatment Encodings and Data Architecture

### Treatment Encoding System

The CRN uses **one-hot encoding** for treatments with 4 possible treatment options:

```python
# Treatment encoding in utils/evaluation_utils.py:73-88
[1, 0, 0, 0] = No treatment
[0, 1, 0, 0] = Chemotherapy only  
[0, 0, 1, 0] = Radiotherapy only
[0, 0, 0, 1] = Combined chemotherapy + radiotherapy
```

The model architecture expects:
- `num_treatments` = 4 (configurable parameter)
- Treatment tensors: `[batch_size, max_sequence_length, num_treatments]`
- Previous and current treatment placeholders in `CRN_model.py:40-41`

### Data-Specific Components (Cancer Domain)

**Located in `utils/cancer_simulation.py`:**
- **Treatment Assignment Logic**: Sigmoid probabilities based on tumor diameter over 15-day windows
- **Pharmacokinetic Model**: Chemotherapy concentration with exponential decay (half-life modeling)
- **Tumor Growth Dynamics**: Gompertz growth model with treatment effects
- **Patient Heterogeneity**: Cancer stage distributions (I, II, IIIA, IIIB, IV) 
- **Confounding Parameters**: `chemo_coeff`/`radio_coeff` control treatment selection bias
- **Outcome Variables**: Tumor volume (continuous), death threshold at 13cm diameter

**Domain-Specific Constants:**
- `tumour_cell_density = 5.8 * 10^8 cells/cm³`
- `tumour_death_threshold = calc_volume(13)` 
- Cancer stage size distributions from medical literature
- Drug half-life and dosing parameters

### General/Reusable Components

**Core ML Architecture (`CRN_model.py`):**
- **Sequence-to-Sequence Framework**: Encoder-decoder with LSTM cells
- **Domain Adversarial Training**: Gradient reversal layer implementation
- **Balancing Representation Builder**: Treatment-invariant representation learning
- **Hyperparameter System**: Configurable architecture (br_size, rnn_hidden_units, etc.)

**General Utilities:**
- **`utils/flip_gradient.py`**: Domain adaptation gradient reversal (universally applicable)
- **`utils/evaluation_utils.py`**: Model loading, data preprocessing patterns
- **Training Infrastructure**: Encoder/decoder training phases, hyperparameter tuning

### Adaptation Guidelines for New Domains

To adapt CRN for other applications (e.g., diabetes, medication management):

1. **Replace `utils/cancer_simulation.py`** with domain-specific data generation
2. **Modify treatment encoding** in preprocessing (change `num_treatments` and one-hot mapping)
3. **Update outcome variables** (e.g., glucose levels instead of tumor volume)
4. **Adjust confounding mechanisms** (domain-specific treatment assignment policies)
5. **Retain core architecture** (`CRN_model.py`, adversarial training, sequence modeling)

The **general CRN architecture** (adversarial balancing + sequence modeling) is domain-agnostic and can be applied to any sequential treatment effect estimation problem.

## Theoretical Foundation

The method is theoretically grounded in:
- **Potential Outcomes Framework**: Extended to time-varying treatments
- **Sequential Strong Ignorability**: Assumes no hidden confounders
- **H-divergence Minimization**: Builds representations indistinguishable across treatment domains
- **Adversarial Training**: Proven to achieve treatment-invariant representations when global minimum is reached