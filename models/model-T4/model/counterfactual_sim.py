import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import logging

logger = logging.getLogger(__name__)

def create_counterfactual_dataset(x, x_demo, original_treatment, mask, device):
    """
    Create a dataset for counterfactual simulation
    
    Args:
        x: Input features [batch_size, seq_len, features]
        x_demo: Static features [batch_size, demo_features]
        original_treatment: Original treatment sequence [batch_size, seq_len]
        mask: Mask for valid timesteps [batch_size, seq_len, features]
        device: PyTorch device
        
    Returns:
        dataset: TensorDataset for counterfactual simulation
    """
    # Convert to tensors if not already
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
    if not torch.is_tensor(x_demo):
        x_demo = torch.tensor(x_demo, dtype=torch.float32).to(device)
    if not torch.is_tensor(original_treatment):
        original_treatment = torch.tensor(original_treatment, dtype=torch.float32).to(device)
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, dtype=torch.float32).to(device)
    
    # Create dummy y and death tensors (not used in inference)
    dummy_y = torch.zeros(x.shape[0], original_treatment.shape[1], 1, dtype=torch.float32).to(device)
    dummy_death = torch.zeros(x.shape[0], dtype=torch.float32).to(device)
    
    dataset = TensorDataset(x, x_demo, original_treatment, dummy_y, dummy_y, dummy_death, mask)
    return dataset

def simulate_counterfactual(model, x_batch, x_demo_batch, treatment_batch, new_treatment_batch, 
                           mask_batch, device, batch_size=32):
    """
    Simulate counterfactual outcomes with new treatment values
    
    Args:
        model: Trained T4 model
        x_batch: Input features [batch_size, seq_len, features]
        x_demo_batch: Static features [batch_size, demo_features]
        treatment_batch: Original treatment sequence [batch_size, seq_len+pred_len]
        new_treatment_batch: New treatment sequence [batch_size, seq_len+pred_len]
        mask_batch: Mask for valid timesteps [batch_size, seq_len, features]
        device: PyTorch device
        batch_size: Batch size for processing
        
    Returns:
        dict: Dictionary containing factual and counterfactual trajectories
    """
    # Create dataset and dataloader
    dataset = create_counterfactual_dataset(x_batch, x_demo_batch, treatment_batch, mask_batch, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    factual_trajectories = []
    counterfactual_trajectories = []
    patient_reps = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, x_demo, treatment, _, _, _, mask = batch
            
            # Get predictions for original treatment
            output_factual, _, patient_rep, _, _ = model(
                x, torch.zeros_like(x[:, :, 0:1]), x_demo, treatment, teacher_forcing_ratio=0
            )
            
            # Convert new_treatment to tensor
            if not torch.is_tensor(new_treatment_batch):
                new_treatment = torch.tensor(new_treatment_batch, dtype=torch.float32).to(device)
            else:
                new_treatment = new_treatment_batch.to(device)
            
            # Get predictions for new treatment
            _, output_counterfactual, _, _, _ = model(
                x, torch.zeros_like(x[:, :, 0:1]), x_demo, new_treatment, teacher_forcing_ratio=0
            )
            
            factual_trajectories.append(output_factual.cpu().numpy())
            counterfactual_trajectories.append(output_counterfactual.cpu().numpy())
            patient_reps.append(patient_rep.cpu().numpy())
    
    # Concatenate results
    factual_trajectories = np.concatenate(factual_trajectories, axis=0)
    counterfactual_trajectories = np.concatenate(counterfactual_trajectories, axis=0)
    patient_reps = np.concatenate(patient_reps, axis=0)
    
    return {
        'factual_trajectories': factual_trajectories,
        'counterfactual_trajectories': counterfactual_trajectories,
        'patient_representations': patient_reps
    }

def create_dose_counterfactual(model, x, x_demo, original_treatment, mask, 
                               dose_value, device, batch_size=32):
    """
    Create a counterfactual with a specific insulin dose value
    
    Args:
        model: Trained T4 model
        x: Input features [batch_size, seq_len, features]
        x_demo: Static features [batch_size, demo_features]
        original_treatment: Original treatment sequence [batch_size, seq_len+pred_len]
        mask: Mask for valid timesteps [batch_size, seq_len, features]
        dose_value: New insulin dose value (between 0 and 1)
        device: PyTorch device
        batch_size: Batch size for processing
        
    Returns:
        dict: Dictionary containing factual and counterfactual trajectories
    """
    # Create new treatment with specified dose value
    new_treatment = original_treatment.clone() if torch.is_tensor(original_treatment) else np.copy(original_treatment)
    
    # Set all non-zero values to the new dose value
    if torch.is_tensor(new_treatment):
        mask = original_treatment > 0
        new_treatment[mask] = dose_value
    else:
        mask = original_treatment > 0
        new_treatment[mask] = dose_value
    
    # Run simulation
    return simulate_counterfactual(
        model, x, x_demo, original_treatment, new_treatment, mask, device, batch_size
    )

def create_timing_counterfactual(model, x, x_demo, original_treatment, original_timing,
                                mask, timing_shift, device, batch_size=32):
    """
    Create a counterfactual with shifted insulin timing
    
    Args:
        model: Trained T4 model
        x: Input features [batch_size, seq_len, features]
        x_demo: Static features [batch_size, demo_features]
        original_treatment: Original treatment sequence [batch_size, seq_len+pred_len]
        original_timing: Original timing values [batch_size, seq_len+pred_len]
        mask: Mask for valid timesteps [batch_size, seq_len, features]
        timing_shift: Minutes to shift insulin timing (-30 to +30)
        device: PyTorch device
        batch_size: Batch size for processing
        
    Returns:
        dict: Dictionary containing factual and counterfactual trajectories
    """
    # We need to adjust the timing values
    new_timing = original_timing.clone() if torch.is_tensor(original_timing) else np.copy(original_timing)
    
    # Add the timing shift (only where insulin is given)
    if torch.is_tensor(new_timing):
        mask = original_treatment > 0
        new_timing[mask] = new_timing[mask] + timing_shift
    else:
        mask = original_treatment > 0
        new_timing[mask] = new_timing[mask] + timing_shift
    
    # Combine dose and timing into a multi-dimensional treatment
    if torch.is_tensor(original_treatment):
        new_treatment = torch.stack([original_treatment, new_timing], dim=-1)
    else:
        new_treatment = np.stack([original_treatment, new_timing], axis=-1)
    
    # Run simulation with multi-dimensional treatment
    return simulate_counterfactual(
        model, x, x_demo, original_treatment, new_treatment, mask, device, batch_size
    )

def get_optimal_dose(model, x, x_demo, mask, device, dose_values=None, batch_size=32):
    """
    Find the optimal insulin dosage for each patient
    
    Args:
        model: Trained T4 model
        x: Input features [batch_size, seq_len, features]
        x_demo: Static features [batch_size, demo_features]
        mask: Mask for valid timesteps [batch_size, seq_len, features]
        device: PyTorch device
        dose_values: List of dose values to try (default: [0.0, 0.25, 0.5, 0.75, 1.0])
        batch_size: Batch size for processing
        
    Returns:
        dict: Dictionary containing optimal treatment and outcomes
    """
    if dose_values is None:
        dose_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Get original treatment (we'll only use this as a template)
    seq_len = x.shape[1]
    pred_len = 36  # Default prediction length (3 hours)
    
    # Create dummy original treatment (all zeros)
    original_treatment = torch.zeros(x.shape[0], seq_len + pred_len).to(device)
    
    # Find the latest time with glucose data (for intervention)
    intervention_time = seq_len - 1
    
    best_outcomes = np.inf * np.ones(x.shape[0])
    best_doses = np.zeros(x.shape[0])
    all_trajectories = []
    
    # Try each dose value
    for dose in dose_values:
        # Create new treatment with this dose at intervention time
        new_treatment = original_treatment.clone()
        new_treatment[:, intervention_time] = dose
        
        # Simulate outcome
        results = simulate_counterfactual(
            model, x, x_demo, original_treatment, new_treatment, mask, device, batch_size
        )
        
        # Extract counterfactual trajectories
        cf_trajectories = results['counterfactual_trajectories']
        all_trajectories.append(cf_trajectories)
        
        # Evaluate outcome (e.g., minimize glucose deviation from target)
        target_glucose = 0.0  # Normalized target glucose (0 = mean)
        outcome_metric = np.mean(np.abs(cf_trajectories - target_glucose), axis=1)
        
        # Update if better than current best
        better_mask = outcome_metric < best_outcomes
        best_outcomes[better_mask] = outcome_metric[better_mask]
        best_doses[better_mask] = dose
    
    return {
        'optimal_doses': best_doses,
        'optimal_outcomes': best_outcomes,
        'all_trajectories': np.array(all_trajectories)
    }

def evaluate_glucose_control(trajectories, target_range=(-0.5, 0.5)):
    """
    Evaluate glucose control based on time in range
    
    Args:
        trajectories: Glucose trajectories [batch_size, seq_len]
        target_range: Target glucose range in normalized units (default: -0.5 to 0.5)
        
    Returns:
        float: Percentage of time in range
    """
    low, high = target_range
    in_range = np.logical_and(trajectories >= low, trajectories <= high)
    time_in_range = np.mean(in_range)
    return time_in_range

def plot_counterfactual_glucose(factual, counterfactual, original_treatment, new_treatment, 
                               timestamps=None, target_range=(-0.5, 0.5)):
    """
    Plot factual and counterfactual glucose trajectories
    
    Args:
        factual: Factual glucose trajectory [seq_len]
        counterfactual: Counterfactual glucose trajectory [seq_len]
        original_treatment: Original treatment sequence [seq_len]
        new_treatment: New treatment sequence [seq_len]
        timestamps: Timestamps for x-axis
        target_range: Target glucose range in normalized units
        
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    if timestamps is None:
        timestamps = np.arange(len(factual)) * 5  # 5-minute intervals
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot glucose trajectories
    ax1.plot(timestamps, factual, 'b-', label='Factual')
    ax1.plot(timestamps, counterfactual, 'r--', label='Counterfactual')
    
    # Add target range
    low, high = target_range
    ax1.axhspan(low, high, color='green', alpha=0.2, label='Target Range')
    
    # Highlight treatment differences
    diff_mask = original_treatment != new_treatment
    diff_times = timestamps[diff_mask]
    for t in diff_times:
        ax1.axvline(t, color='gray', alpha=0.3)
    
    ax1.set_ylabel('Normalized Glucose')
    ax1.set_title('Glucose Counterfactual Trajectory')
    ax1.legend()
    
    # Plot treatment in bottom panel
    ax2.step(timestamps, original_treatment, 'b-', where='post', label='Original Treatment')
    ax2.step(timestamps, new_treatment, 'r--', where='post', label='Counterfactual Treatment')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Insulin Dose')
    ax2.set_ylim([-0.1, 1.1])
    ax2.legend()
    
    plt.tight_layout()
    return fig

def analyze_dose_response(model, x, x_demo, mask, device, 
                         dose_range=np.linspace(0, 1, 11), batch_size=32):
    """
    Analyze the dose-response relationship for glucose control
    
    Args:
        model: Trained T4 model
        x: Input features [batch_size, seq_len, features] 
        x_demo: Static features [batch_size, demo_features]
        mask: Mask for valid timesteps [batch_size, seq_len, features]
        device: PyTorch device
        dose_range: Range of doses to analyze
        batch_size: Batch size for processing
        
    Returns:
        dict: Dictionary containing dose-response results
    """
    seq_len = x.shape[1]
    pred_len = 36  # Default prediction length (3 hours)
    
    # Create dummy original treatment (all zeros)
    original_treatment = torch.zeros(x.shape[0], seq_len + pred_len).to(device)
    
    # Find the latest time with glucose data (for intervention)
    intervention_time = seq_len - 1
    
    # Track outcomes for each dose
    dose_outcomes = []
    
    # Try each dose value
    for dose in dose_range:
        # Create new treatment with this dose at intervention time
        new_treatment = original_treatment.clone()
        new_treatment[:, intervention_time] = dose
        
        # Simulate outcome
        results = simulate_counterfactual(
            model, x, x_demo, original_treatment, new_treatment, mask, device, batch_size
        )
        
        # Extract counterfactual trajectories
        cf_trajectories = results['counterfactual_trajectories']
        
        # Evaluate outcomes
        in_range = evaluate_glucose_control(cf_trajectories)
        mean_glucose = np.mean(cf_trajectories, axis=1)
        glucose_var = np.var(cf_trajectories, axis=1)
        
        dose_outcomes.append({
            'dose': dose,
            'time_in_range': in_range,
            'mean_glucose': np.mean(mean_glucose),
            'glucose_variance': np.mean(glucose_var)
        })
    
    return {
        'dose_range': dose_range,
        'outcomes': dose_outcomes
    } 