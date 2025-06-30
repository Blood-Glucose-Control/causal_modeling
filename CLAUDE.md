# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a comprehensive pipeline for causal modeling in healthcare, specifically focusing on understanding how insulin timing and dosage decisions affect blood glucose levels. The project combines multiple advanced causal inference methods with real-world health data processing capabilities.

## Repository Architecture

The codebase is organized into distinct modules, each implementing different causal modeling approaches:

### Core Components

1. **Health Data Processing** (`data_standardization/`)
   - Standardizes multi-source health data (Gluroo + FitBit)
   - Handles varying data frequencies and file structures
   - Manages event-based vs. regular interval data

2. **Causal Modeling Frameworks**:
   - **CRN** (`model-CRN/`) - Counterfactual Recurrent Network
   - **Causal Transformer** (`model-CausalTransformer/`) - Transformer-based causal inference
   - **T4** (`model-T4/`) - Time-to-treatment modeling framework
   - **ITS** (`model-ITS-causalimpact/`) - Interrupted Time Series analysis

3. **Synthetic Data Generation** (`synthetic_data/`)
   - Glucose simulation and counterfactual generation
   - Insulin dose and timing modification tools

## Common Development Commands

### Data Processing
```bash
# Standardize health data from multiple sources
python data_standardization/standardize_health_data.py

# Validate processed data quality
python data_standardization/validate_health_data.py

# Visualize health data patterns
python data_standardization/visualize_health_data.py
```

### Testing Commands

#### CRN Model Testing
```bash
# Run CRN tests with specific parameters
python model-CRN/test_crn.py --chemo_coeff 2 --radio_coeff 2 --results_dir results --model_name "crn_test"
```

#### ITS Model Testing
```bash
# Interactive test pipeline
python model-ITS-causalimpact/test_pipeline.py

# Single analysis
python model-ITS-causalimpact/run_its_models.py --analysis single --data-path synthetic_data/data/ml_dataset.csv --output-dir output/single_analysis --max-events 5

# Counterfactual analysis
python model-ITS-causalimpact/run_its_models.py --analysis counterfactual --data-path synthetic_data/data/dose_counterfactuals --output-dir output/counterfactual_analysis --max-events 5
```

#### Causal Transformer Testing
```bash
# Test specific datasets
python model-CausalTransformer/tests/test_ct_gluroo.py
python model-CausalTransformer/tests/test_ct_cancer.py
```

### Training Commands

#### T4 Model
```bash
# Run complete T4 training pipeline
cd model-T4 && ./run.sh [TAU_VALUE]

# Train glucose-specific model
python model-T4/train_glucose_model.py
```

#### Causal Transformer Training
```bash
# Various training configurations available
python model-CausalTransformer/runnables/train_multi.py      # Multi-task training
python model-CausalTransformer/runnables/train_gnet.py       # G-Net training
python model-CausalTransformer/runnables/train_enc_dec.py    # Encoder-decoder training
```

### Environment Setup

#### Virtual Environments
```bash
# CRN environment (if available)
source model-CRN/crn_env/bin/activate

# Install Causal Transformer package
cd model-CausalTransformer && pip install -e .
```

#### Dependencies
Each model has its own requirements file:
- `model-CRN/requirements.txt` - Older TensorFlow stack (TF 1.15)
- `model-CausalTransformer/requirements.txt` - PyTorch Lightning setup

### Synthetic Data Generation
```bash
# Generate base glucose simulation data
python synthetic_data/simple_glucose_gen.py

# Generate counterfactual scenarios
python synthetic_data/insulin_dose_counterfactual_gen.py
python synthetic_data/insulin_timing_counterfactual_gen.py
```

## Data Structure Understanding

### Health Data Sources
- **Gluroo**: Blood glucose (5min intervals), insulin doses (event-based), meals (event-based)
- **FitBit**: Sleep scores (daily), stress (5min), temperature (1min), SpO2 (1min), HRV (daily/event)

### File Organization Patterns
- **Time-range files**: `{user_id}_5th-7th.csv` (Gluroo data)
- **Daily files**: `Daily Readiness Score - 2024-04-01.csv`
- **Monthly files**: `Minute SpO2 - 2024-01.csv`

### Key Challenges Addressed
1. **Time Deconfounding**: Capturing exact causal effects in time series
2. **Autocorrelation**: Managing recursive effects in sequential data
3. **Individual Treatment Effects**: Person-specific rather than population-average effects
4. **Continuous Treatments**: Non-binary insulin dosing and timing decisions

## Model-Specific Notes

### CRN (Counterfactual Recurrent Network)
- Focuses on cancer treatment simulation but architecture applies to glucose modeling
- Encoder-decoder architecture with domain adversarial training
- Evaluation includes both decoder and encoder components

### Causal Transformer
- Uses attention mechanisms for temporal causal relationships
- Supports multiple training paradigms (MSM, G-Net, RMSN)
- Configuration managed via Hydra YAML files

### T4 Framework
- Specifically designed for time-to-treatment analysis
- Shell script orchestrates multi-seed training
- CUDA-optimized training pipeline

### ITS (Interrupted Time Series)
- Uses causalimpact package for intervention analysis
- Supports both single event and counterfactual analysis modes
- Generates comprehensive visualization outputs

## Output Directories
- `output/` - Main results directory
- `model-*/results/` - Model-specific results
- `model-*/test_output/` - Test run outputs with visualizations

## Important Implementation Details

### Data Processing Pipeline
The standardization process handles multiple data frequencies and formats, requiring careful timestamp alignment and duplicate detection across overlapping time periods.

### Causal Inference Methods
Each model implements different approaches to the fundamental challenge of estimating "what would have happened" under different treatment decisions, using techniques from domain adversarial training to transformer attention mechanisms.

### Visualization Capabilities
Most models include comprehensive plotting and analysis tools, particularly important for validating causal effect estimates and understanding model behavior.