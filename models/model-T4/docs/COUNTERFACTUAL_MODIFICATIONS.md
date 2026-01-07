# Modifications for Counterfactual Analysis

This document outlines the key modifications made to the original T4 model structure to enable counterfactual trajectory generation.

## 1. Model Architecture Adaptations

### Original Structure
The original T4 model was designed to:
- Encode patient history and current state
- Predict future outcomes based on observed treatments
- Output a single trajectory

### Key Modifications
1. **Dual Output in Seq2Seq**:
   ```python
   # Original output
   output, hidden, patient_rep = model(x, y, x_demo, treatment)

   # Modified to include counterfactual
   output_factual, output_counterfactual, patient_rep, ps_output, attention = model(
       x, y, x_demo, treatment, teacher_forcing_ratio=0
   )
   ```
   - Added parallel trajectory generation
   - Enabled simultaneous factual and counterfactual predictions
   - Maintained patient representation for consistency

2. **Treatment Sequence Handling**:
   ```python
   # Added treatment sequence padding/truncation
   if treatment_batch.size(1) < total_len:
       pad_size = total_len - treatment_batch.size(1)
       last_treatment = treatment_batch[:, -1:].repeat(1, pad_size)
       treatment_batch = torch.cat([treatment_batch, last_treatment], dim=1)
   ```
   - Ensures treatment sequences match required lengths
   - Maintains temporal alignment with predictions

## 2. Data Processing Changes

1. **Dataset Creation**:
   ```python
   def create_counterfactual_dataset(x, x_demo, original_treatment, mask, device):
       # Convert to tensors if not already
       if not torch.is_tensor(x):
           x = torch.tensor(x, dtype=torch.float32)
       # ... similar for other inputs
       
       # Create dummy y and death tensors (not used in inference)
       dummy_y = torch.zeros_like(original_treatment)
       dummy_death = torch.zeros(len(x), dtype=torch.float32)
       
       dataset = TensorDataset(x, x_demo, original_treatment, dummy_y, dummy_y, 
                             dummy_death, mask)
       return dataset
   ```
   - Added support for counterfactual data processing
   - Created placeholder structures for unused variables
   - Maintained compatibility with original data format

2. **Treatment Manipulation**:
   ```python
   # For opposite treatment analysis
   new_treatment = 1 - original_treatment  # Flip all treatments
   
   # For optimal treatment search
   possible_treatments = np.array([
       [int(b) for b in format(i, f'0{seq_len}b')]
       for i in range(2 ** seq_len)
   ])
   ```
   - Added treatment sequence generation
   - Enabled systematic exploration of treatment options

## 3. Simulation Components

1. **Counterfactual Simulation Function**:
   ```python
   def simulate_counterfactual(model, x, x_demo, original_treatment, new_treatment, 
                             mask, device, batch_size=32):
       # Create dataset and dataloader
       dataset = create_counterfactual_dataset(...)
       
       # Run simulations
       factual_trajectories = []
       counterfactual_trajectories = []
       
       with torch.no_grad():
           for batch in dataloader:
               # Get predictions for original treatment
               output_factual, _, patient_rep, _, _ = model(...)
               
               # Get predictions for new treatment
               _, output_counterfactual, _, _, _ = model(...)
   ```
   - Added dedicated simulation pipeline
   - Handles batch processing for efficiency
   - Maintains model state consistency

2. **Optimal Treatment Search**:
   ```python
   def get_optimal_treatment(model, x, x_demo, mask, device, 
                           possible_treatments=None, batch_size=32):
       # Generate treatment combinations
       if possible_treatments is None:
           seq_len = x.shape[1]
           possible_treatments = generate_treatment_combinations(seq_len)
       
       # Evaluate each treatment sequence
       for treatment_seq in possible_treatments:
           results = simulate_counterfactual(...)
           outcome = evaluate_outcome(results)
   ```
   - Added systematic treatment exploration
   - Implemented outcome evaluation
   - Optimized for shorter sequences

## 4. Visualization Enhancements

1. **Trajectory Visualization**:
   ```python
   def plot_trajectories(factual_trajectory, counterfactual_trajectory, 
                        original_treatment, new_treatment, ...):
       # Plot both trajectories
       plt.plot(factual_trajectory, 'b-', label='Factual')
       plt.plot(counterfactual_trajectory, 'r--', label='Counterfactual')
       
       # Show treatment differences
       diff_mask = original_treatment != new_treatment
       plt.vlines(diff_times, -0.1, 1.1, colors='gray', alpha=0.3)
   ```
   - Added comparative visualization
   - Highlighted treatment changes
   - Included uncertainty bounds

## 5. Key Considerations

1. **Model Behavior**:
   - Maintains consistency in patient representations
   - Ensures treatment effects are plausible
   - Handles temporal dependencies appropriately

2. **Computational Efficiency**:
   - Batch processing for faster simulation
   - Optimized treatment sequence generation
   - Efficient memory usage

3. **Limitations**:
   - Binary treatment decisions only
   - Limited sequence lengths for optimal search
   - Assumes treatment independence

## Usage Example

```python
# Load model and data
model = load_model(checkpoint_path, args, device)
features, dataset = load_and_process_data(args, device, logger, dataset_name)

# Generate counterfactual trajectories
new_treatment = 1 - original_treatment  # Flip treatments
results = simulate_counterfactual(
    model, x, x_demo, original_treatment, new_treatment, 
    mask, device, batch_size=32
)

# Analyze results
plot_trajectories(
    results['factual_trajectories'],
    results['counterfactual_trajectories'],
    original_treatment,
    new_treatment
)
``` 