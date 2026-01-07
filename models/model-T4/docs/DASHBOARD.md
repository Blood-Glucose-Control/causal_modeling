# T4 Counterfactual Dashboard

This interactive dashboard allows you to explore counterfactual scenarios with the T4 model. You can select a patient, zoom in on a time range, select a counterfactual intervention, and visualize the effects over a configurable number of timesteps.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the dashboard with:

```bash
python counterfactual_dashboard.py --checkpoint checkpoints/your_model.pt --device cpu --port 8050
```

Arguments:
- `--checkpoint`: Path to the model checkpoint (required)
- `--device`: Device to run the model on (`cpu` or `cuda`, default: `cpu`)
- `--port`: Port to run the dashboard on (default: 8050)

## Dashboard Features

The dashboard is divided into two main sections:

### Control Panel (Left)

1. **Patient Selection**
   - Select a patient from the dropdown
   - Select a feature to visualize

2. **Time Range**
   - Set the historical data range to zoom in on specific time periods
   - Set the intervention time (when the treatment is applied)
   - Configure the number of prediction steps after the intervention

3. **Treatment Configuration**
   - Set the treatment intensity (0.0 to 1.0)
   - Click "Generate Counterfactual" to update the visualization

### Visualization Panel (Right)

1. **Patient Trajectory and Counterfactual**
   - Top subplot: Shows historical data, factual trajectory, and counterfactual trajectory
   - Bottom subplot: Shows original treatment and new treatment sequences
   - Vertical line indicates the intervention point

2. **Treatment Effect**
   - Bar chart showing the magnitude of the treatment effect at each time step
   - Positive values indicate beneficial effects, negative values indicate harmful effects

## Example Scenarios

### Scenario 1: Immediate Intervention

1. Select a patient and feature
2. Set intervention time to the last historical data point
3. Set treatment intensity to 1.0
4. Click "Generate Counterfactual"

### Scenario 2: Delayed Intervention

1. Select a patient and feature
2. Set intervention time to a future time point (after historical data)
3. Set treatment intensity to 0.5
4. Click "Generate Counterfactual"

### Scenario 3: Comparing Different Intensities

1. Select a patient and feature
2. Set intervention time
3. Generate counterfactuals with different treatment intensities (e.g., 0.25, 0.5, 0.75, 1.0)
4. Compare the resulting trajectories and effects

## Limitations

1. The dashboard currently supports one feature at a time
2. Treatment intensities are limited to the range [0, 1]
3. The intervention is applied at a single time point and maintained thereafter

## Future Improvements

1. Support for multiple features in the same visualization
2. Time-varying treatment intensities
3. Uncertainty visualization
4. Comparison of multiple counterfactual scenarios side by side
5. Export of results and visualizations 