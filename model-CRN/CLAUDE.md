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

## Data-API Integration for Diabetes Modeling

### Overview

The **data-api** module (`data-api/main.py`) provides a synthetic diabetes data generator and counterfactual analysis system designed specifically to support the CRN model's adaptation to diabetes management. This serves two critical roles in our research pipeline:

1. **Training Data Generation**: Creates realistic patient glucose data for training the CRN model
2. **Ground Truth Simulation**: Provides counterfactual scenarios for model evaluation and validation

### Core Components

The data-api consists of three main classes:

- **`GlucoseSimulator`**: Generates realistic patient glucose patterns with meals, insulin, exercise, and stress
- **`CounterfactualAnalyzer`**: Creates "what-if" scenarios for insulin dose and timing modifications  
- **`DiabetesAnalyzer`**: Main interface combining data generation and counterfactual analysis

### Data Generation Process

#### Command Usage

```bash
# Basic usage (runs demo with 30 days of data)
uv run python data-api/main.py

# Programmatic usage
from data-api.main import DiabetesAnalyzer
analyzer = DiabetesAnalyzer(seed=42)
patient_data = analyzer.generate_patient_data(n_days=30, start_date='2024-01-01') # creates  30 day training data
# <add here how to add counterfactuals to the training data for ground truth evaluations>
```

#### Generated Data Structure

The simulator produces time-series data at **5-minute intervals** with the following columns:

**Core Variables:**
- `glucose` (int64): Blood glucose levels (mg/dL), range 40-400
- `carbs` (int64): Carbohydrate intake (grams) 
- `insulin` (float64): Insulin doses (units)
- `exercise` (int64): Exercise periods (binary)
- `stress` (float64): Stress levels (0-1 scale)

**Derived Variables:**
- `active_insulin` (float64): Current insulin activity from past doses
- `carb_impact` (float64): Current glucose impact from carb absorption
- `meal_insulin_delay` (int64): Timing difference between meal and insulin (minutes)

**Metadata:**
- `intervention_id` (object): Unique UUID for each insulin intervention

### Counterfactual Analysis Capabilities

#### Dose Counterfactuals

Analyze "what if we changed the insulin dose?" scenarios:

```python
# Example: 20% more insulin
result = analyzer.analyze_intervention(
    patient_data,
    intervention_id="uuid-string",
    analysis_type='dose',
    dose_factor=1.2,  # 20% more insulin
    before_minutes=120,
    after_minutes=180
)
```

#### Timing Counterfactuals

Analyze "what if we gave insulin earlier/later?" scenarios:

```python
# Example: 30 minutes earlier
result = analyzer.analyze_intervention(
    patient_data,
    intervention_id="uuid-string", 
    analysis_type='timing',
    timing_shift_minutes=-30,  # 30 min earlier
    before_minutes=120,
    after_minutes=180
)
```

#### Counterfactual Data Format

Counterfactual analysis adds new columns to the original data:

- `cf{N}_insulin`: Modified insulin doses
- `cf{N}_glucose`: Counterfactual glucose trajectories
- `cf{N}_active_insulin`: Modified insulin activity
- `cf{N}_carb_impact`: Modified carb impacts

Metadata is stored in `DataFrame.attrs` with complete intervention details.

### Integration with CRN Model

#### Current Challenge: Treatment Encoding

The CRN model expects **discrete one-hot encoded treatments**:
```python
# Current cancer model (4 discrete treatments)
[1, 0, 0, 0] = No treatment
[0, 1, 0, 0] = Chemotherapy only  
[0, 0, 1, 0] = Radiotherapy only
[0, 0, 0, 1] = Combined treatment
```

However, diabetes management involves **continuous treatment variables**:
- **Insulin dose**: Continuous values (e.g., 4.9 units, 5.6 units)
- **Insulin timing**: Continuous time shifts (e.g., -30 min, +15 min)

#### Adaptation Strategy

To integrate data-api with CRN, we need to:

1. **Discretize Continuous Treatments**: Convert continuous insulin doses/timing into discrete bins
2. **Modify Treatment Encoding**: Update `num_treatments` parameter and one-hot mapping
3. **Replace Cancer Simulation**: Use data-api instead of `utils/cancer_simulation.py`
4. **Update Outcome Variables**: Change from tumor volume to glucose trajectories

#### Ground Truth Evaluation

The data-api serves as the **ground truth simulator** for model evaluation:

- **Training**: CRN learns from historical patient data (glucose, insulin, meals, etc.)
- **Evaluation**: Compare CRN counterfactual predictions against data-api ground truth
- **Metrics**: Glucose trajectory accuracy, hypoglycemia/hyperglycemia prediction

### Clinical Applications

The integrated system will enable personalized diabetes management:

- **Insulin Dosing**: Optimize insulin-to-carb ratios for individual patients
- **Meal Timing**: Analyze optimal insulin timing relative to meals
- **Exercise Planning**: Predict glucose impact of exercise and adjust insulin accordingly
- **Stress Management**: Account for stress-induced glucose variations

### Next Steps for Integration

1. **Treatment Discretization**: Implement binning strategy for continuous insulin variables
2. **Data Preprocessing**: Create diabetes-specific data loading utilities  
3. **Model Architecture**: Adapt CRN input layers for diabetes feature set
4. **Evaluation Pipeline**: Implement ground truth comparison metrics using data-api

## Theoretical Foundation

The method is theoretically grounded in:
- **Potential Outcomes Framework**: Extended to time-varying treatments
- **Sequential Strong Ignorability**: Assumes no hidden confounders
- **H-divergence Minimization**: Builds representations indistinguishable across treatment domains
- **Adversarial Training**: Proven to achieve treatment-invariant representations when global minimum is reached