# T4 Glucose Counterfactual Model

This extension of the T4 model enables counterfactual analysis of blood glucose trajectories under different insulin dosing and timing scenarios. It allows for predicting "what-if" scenarios with continuous treatment values (insulin dosages) and timing shifts.

## Key Modifications

We've modified the original T4 model in several important ways:

### 1. Non-Binary Treatment Support

The original T4 model only supported binary treatments (0 or 1). Our modifications enable:

- **Continuous insulin dosages**: Values between 0 and 1 representing different dose amounts
- **Timing shifts**: Representation of insulin timing differences (-30 to +30 minutes)
- **Multi-dimensional treatments**: Simultaneous modeling of both dosage and timing effects

### 2. Model Architecture Changes

- **AttentionDecoder**: Modified to handle multi-dimensional treatments with separate branches for dosage and timing effects
- **Treatment impact**: Changed from binary switching to continuous interpolation between effect branches
- **Timing representation**: Added specialized handling for timing shifts and their effects on glucose trajectories

### 3. Data Processing

- New `load_glucose_data()` function to process time-series glucose data
- Treatment representation that captures both insulin dosages and timing information
- Sliding window approach for handling long continuous glucose monitoring data

### 4. Counterfactual Simulation

- Specialized functions for insulin dose counterfactuals
- Support for insulin timing counterfactuals
- Dose-response analysis for optimizing insulin doses
- Advanced visualization of counterfactual outcomes

## Usage

### Data Preparation

Generate synthetic glucose data:
```bash
python simulation/simple_glucose_gen.py
```

This generates:
- `data/full_dataset.csv`: Raw time-series data
- `data/ml_dataset.csv`: Feature-engineered data ready for modeling

### Training

Train the model with:
```bash
python model/train_glucose_model.py --data_path data/ml_dataset.csv --epochs 50
```

Key parameters:
- `--window_size`: Historical window length (default: 12 = 1 hour at 5min intervals)
- `--pred_horizon`: Prediction horizon (default: 36 = 3 hours at 5min intervals)
- `--cf_samples`: Number of counterfactual examples to visualize

### Counterfactual Generation

After training, you can generate counterfactuals using:

```python
from model.counterfactual_sim import create_dose_counterfactual, create_timing_counterfactual

# Generate dose counterfactual
results = create_dose_counterfactual(
    model=model,
    x=x_batch,
    x_demo=x_demo_batch,
    original_treatment=original_treatment,
    mask=mask_batch,
    dose_value=0.5,  # Half dose
    device=device
)

# Generate timing counterfactual
results = create_timing_counterfactual(
    model=model,
    x=x_batch,
    x_demo=x_demo_batch,
    original_treatment=original_treatment,
    original_timing=original_timing,
    mask=mask_batch,
    timing_shift=-15,  # 15 minutes earlier
    device=device
)
```

### Optimization

Find optimal insulin dosages:

```python
from model.counterfactual_sim import get_optimal_dose

results = get_optimal_dose(
    model=model,
    x=x_batch,
    x_demo=x_demo_batch,
    mask=mask_batch,
    device=device,
    dose_values=[0.0, 0.25, 0.5, 0.75, 1.0]
)
```

## Visualization

The model generates several visualizations:

1. **Dose comparison**: Shows how different insulin doses affect glucose trajectories
2. **Timing comparison**: Shows how insulin timing affects glucose trajectories
3. **Dose-response curves**: Shows the relationship between insulin doses and glucose outcomes

Visualizations are saved to `visualizations/{date}/` directory.

## Implementation Details

### Key Files

- `model/seq2seq.py`: Modified T4 model architecture
- `model/dataset.py`: Data processing for glucose monitoring data
- `model/counterfactual_sim.py`: Counterfactual simulation functions
- `model/train_glucose_model.py`: Training script for glucose model
- `simulation/simple_glucose_gen.py`: Synthetic glucose data generator

### Model Parameters

- **Glucose history window**: 12 timesteps (1 hour at 5min intervals)
- **Prediction horizon**: 36 timesteps (3 hours at 5min intervals)
- **Feature dimensions**: 11 (glucose, carbs, insulin, exercise, etc.)
- **Treatment dimensions**: 2 (insulin dosage and timing)

## Limitations and Future Work

1. **Data limitations**: Currently using synthetic data; real-world data may have different dynamics
2. **Model complexity**: The current approach handles timing as a separate dimension, but integrating it more deeply into the physiological model could improve accuracy
3. **Patient variability**: The current model doesn't account for inter-patient variability in insulin sensitivity
4. **Multiple treatments**: Future work could model multiple treatment types (e.g., different insulin types, exercise, carbohydrate intake) 