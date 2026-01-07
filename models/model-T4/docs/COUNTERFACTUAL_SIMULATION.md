# Counterfactual Simulation Process

This document describes the process of simulating counterfactual trajectories using the trained T4 model. The process involves loading a trained model, preparing patient data, simulating alternative treatment sequences, and visualizing the results.

## Overview

The counterfactual simulation allows us to answer questions like:
- "What would have happened if we had given a different treatment?"
- "What is the optimal treatment sequence for this patient?"
- "How do different treatment strategies affect patient outcomes?"

## Process Steps

### 1. Model Loading

```python
model = load_model(checkpoint_path, args, device)
```

- Loads a trained T4 model from a checkpoint
- Initializes both encoder and decoder components
- Handles different checkpoint formats (state dict or full model)
- Moves model to specified device (CPU/GPU)

### 2. Data Preparation

```python
features, dataset = load_and_process_data(args, device, logger, dataset_name)
```

The data includes:
- Temporal features (x): [batch_size, seq_len, features]
- Static features (x_demo): [batch_size, demo_features]
- Treatment sequences: [batch_size, seq_len]
- Masks for valid timesteps: [batch_size, seq_len]

### 3. Sequence Length Handling

The process handles different sequence lengths:
- Input sequence length (e.g., 20 timesteps)
- Prediction window (e.g., 4 timesteps)
- Treatment sequence length (padded/truncated as needed)

Example dimensions:
```
x shape: [32, 20, 20]        # 32 patients, 20 timesteps, 20 features
y shape: [32, 4]             # 32 patients, 4 prediction timesteps
treatment shape: [32, 23]    # 32 patients, 23 treatment timesteps
mask shape: [32, 20, 20]     # 32 patients, 20 timesteps, 20 features
```

### 4. Counterfactual Simulation Types

#### A. Opposite Treatment Simulation
```python
new_treatment = 1 - original_treatment  # Flip all treatments
```
- Takes the original treatment sequence
- Inverts all treatment decisions (0→1, 1→0)
- Simulates patient trajectories under these opposite treatments

#### B. Optimal Treatment Search
```python
results = get_optimal_treatment(model, x, x_demo, mask, device)
```
- Generates all possible treatment combinations
- Simulates outcomes for each combination
- Identifies the treatment sequence with the best outcome

### 5. Simulation Process

For each simulation:
1. Prepare input data and treatment sequences
2. Run model forward pass:
   ```python
   output_factual, _, patient_rep, _, _ = model(
       x_batch, y_batch, x_demo_batch, treatment_batch,
       teacher_forcing_ratio=0
   )
   ```
3. Calculate treatment effects:
   ```python
   treatment_effects = counterfactual_trajectories - factual_trajectories
   ```
4. Estimate uncertainty in predictions

### 6. Visualization

Two main types of visualizations are generated:

#### A. Multiple Patient Trajectories
```
results/counterfactuals/opposite_treatment/patient_*.png
```
- Shows factual vs counterfactual trajectories for multiple patients
- Includes treatment sequences and changes
- Highlights points of treatment difference

#### B. Optimal Treatment Comparison
```
results/counterfactuals/optimal_treatment.png
```
- Compares baseline (no treatment) with optimal treatment
- Shows predicted outcomes under each scenario
- Visualizes treatment decisions and their timing

### 7. Output Format

Each visualization includes:
1. Top panel:
   - Blue line: Factual trajectory
   - Red dashed line: Counterfactual trajectory
   - Red shaded area: Uncertainty bounds

2. Bottom panel:
   - Blue line: Original treatment sequence
   - Red dashed line: Counterfactual treatment sequence
   - Gray vertical lines: Points of treatment change

3. Side panel:
   - Text box showing intervention details
   - Timing and nature of treatment changes

### 8. Numerical Results

Results are saved as numpy arrays:
```
optimal_treatment.npy  # Optimal treatment sequence
optimal_trajectory.npy # Predicted trajectory under optimal treatment
```

## Usage

Run the simulation with:
```powershell
python simulate_counterfactuals.py --checkpoint checkpoints/[DATE]/[TIME]_[TAU]_[RATIO]_[SEED].pt
```

Key parameters:
- `--checkpoint`: Path to trained model checkpoint
- `--num_patients`: Number of patients to visualize (default: 5)
- `--output_dir`: Directory for saving results
- `--batch_size`: Batch size for processing

## Limitations and Considerations

1. **Sequence Length**:
   - Treatment sequences must be padded/truncated to match required length
   - Prediction window size affects required sequence length

2. **Uncertainty Estimation**:
   - Based on variation in treatment effects
   - May not capture all sources of uncertainty

3. **Optimal Treatment Search**:
   - Computational complexity grows exponentially with sequence length
   - Limited to shorter sequences for practical reasons

4. **Visualization**:
   - Shows one feature at a time
   - Treatment sequences are binary (0/1)

## Future Improvements

1. Support for multi-dimensional outcomes
2. More sophisticated uncertainty estimation
3. Efficient optimal treatment search for longer sequences
4. Interactive visualization options
5. Support for non-binary treatments ✓ Now implemented! See [NON_BINARY_TREATMENTS.md](NON_BINARY_TREATMENTS.md)

## Non-Binary Treatment Support

The T4 model now supports non-binary treatment values (e.g., 0.5) for more nuanced counterfactual analysis. This allows for:
- Partial treatment effects (e.g., half-dose represented as 0.5)
- Treatment intensity analysis (dose-response curves)
- More realistic counterfactual scenarios

To use this feature, see the dedicated documentation in [NON_BINARY_TREATMENTS.md](NON_BINARY_TREATMENTS.md) and the example script `non_binary_counterfactual_demo.py`. 