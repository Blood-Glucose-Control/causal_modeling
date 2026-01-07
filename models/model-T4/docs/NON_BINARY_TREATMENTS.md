# Non-Binary Treatment Values in Counterfactual Generation

This document explains how to use non-binary treatment values (e.g., 0.5) in counterfactual generation with the T4 model. This feature allows for more nuanced "what-if" scenarios beyond just binary on/off treatments.

## Overview

The T4 model has been extended to support continuous treatment values between 0 and 1, allowing for:
- Partial treatment effects (e.g., half-dose represented as 0.5)
- Treatment intensity analysis (e.g., dose-response curves)
- More realistic counterfactual scenarios

## How It Works

The model architecture uses treatment values as interpolation weights between two output branches:

```python
prediction = treatment_next * self.fc_out_1(output) + (1-treatment_next) * self.fc_out_0(output)
```

With this formulation:
- Treatment = 0.0: Full output from fc_out_0 branch
- Treatment = 0.5: Equal weighting (50%) from both branches
- Treatment = 1.0: Full output from fc_out_1 branch

## Key Functions

### 1. Create Specific Counterfactual

```python
from model.counterfactual_sim import create_specific_counterfactual

results = create_specific_counterfactual(
    model=model,
    x=x_batch,
    x_demo=x_demo_batch,
    original_treatment=original_treatment,
    mask=mask_batch,
    treatment_value=0.5,  # Specify any value between 0 and 1
    device=device
)
```

This function creates a counterfactual with a specific treatment value for all time steps.

### 2. Get Optimal Treatment with Non-Binary Values

```python
from model.counterfactual_sim import get_optimal_treatment

# Define treatment values to try
treatment_values = [0.0, 0.25, 0.5, 0.75, 1.0]

optimal_results = get_optimal_treatment(
    model=model,
    x=x_batch,
    x_demo=x_demo_batch,
    mask=mask_batch,
    device=device,
    treatment_values=treatment_values  # Specify the values to try
)
```

This function finds the optimal treatment sequence by trying all combinations of the specified treatment values.

### 3. Simulate Custom Counterfactual

For more complex scenarios with varying treatment values at different time steps:

```python
from model.counterfactual_sim import simulate_counterfactual

# Create a custom treatment sequence with varying intensities
new_treatment = np.zeros_like(original_treatment)
new_treatment[:, 0:2] = 0.25  # 0.25 for first two timesteps
new_treatment[:, 2:4] = 0.5   # 0.5 for next two timesteps
new_treatment[:, 4:] = 0.75   # 0.75 for remaining timesteps

results = simulate_counterfactual(
    model=model,
    x_batch=x_batch,
    x_demo_batch=x_demo_batch,
    treatment_batch=original_treatment,
    new_treatment_batch=new_treatment,
    mask_batch=mask_batch,
    device=device
)
```

## Example Usage

A complete demonstration script is provided in `non_binary_counterfactual_demo.py`. Run it with:

```bash
python non_binary_counterfactual_demo.py --checkpoint checkpoints/your_model.pt
```

The script demonstrates:
1. Generating counterfactuals with different treatment intensities
2. Visualizing treatment intensity comparisons
3. Plotting treatment effect vs. intensity curves
4. Finding optimal treatment with non-binary values

## Visualizations

The demo script generates several visualizations:

### 1. Treatment Intensity Comparison
Shows how different treatment intensities affect patient trajectories:
- Historical data and factual trajectory
- Multiple counterfactual trajectories with different intensities
- Treatment sequences in the bottom panel

### 2. Treatment Effect Curve
Shows the relationship between treatment intensity and treatment effect:
- X-axis: Treatment intensity (0.0 to 1.0)
- Y-axis: Mean treatment effect
- Helps identify optimal treatment intensity

## Limitations and Considerations

1. **Interpretation**: Non-binary values represent treatment intensity or probability, not necessarily a physical dose.

2. **Model Training**: The model should ideally be trained with some non-binary treatment examples for best results.

3. **Computational Complexity**: Using many treatment values increases the search space exponentially when finding optimal treatments.

4. **Extrapolation**: The model may not extrapolate well to treatment values it hasn't seen during training.

## Future Improvements

1. Time-varying treatment intensities optimization
2. Multi-treatment interaction analysis
3. Personalized optimal treatment intensity prediction
4. Uncertainty quantification for non-binary treatments 