import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Import the EnhancedGlucoseGenerator class from the simple_glucose_gen script
from simple_glucose_gen import EnhancedGlucoseGenerator, plot_glucose_data

def generate_insulin_timing_counterfactuals(days=7, timing_shifts=[-60, -30, 0, 30, 60], seed=42):
    """
    Generate multiple datasets with insulin doses at different times while keeping doses constant.
    
    Args:
        days: Number of days to simulate
        timing_shifts: List of time shifts in minutes (negative = earlier, positive = later)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of DataFrames with different insulin timings
    """
    print(f"Generating {len(timing_shifts)} counterfactual scenarios over {days} days...")
    
    # Create base generator with fixed seed
    base_generator = EnhancedGlucoseGenerator(seed=seed)
    
    # Generate base data first (no modifications)
    base_data = base_generator.generate_data(days=days)
    
    # Dictionary to store all datasets
    datasets = {"0": base_data.copy()}  # 0 is our baseline/factual scenario (no shift)
    
    # For each timing shift, create a modified dataset
    for shift in timing_shifts:
        if shift == 0:
            continue  # We already have the baseline
            
        print(f"Generating scenario with insulin timing shift: {shift} minutes...")
        
        # Create a new dataset with adjusted times for insulin doses
        modified_data = base_data.copy()
        
        # Find all insulin doses
        insulin_doses = modified_data[modified_data['insulin'] > 0].copy()
        
        # Remove original insulin doses
        modified_data['insulin'] = 0.0
        
        # Add insulin doses at shifted times
        shift_delta = timedelta(minutes=shift)
        for idx, row in insulin_doses.iterrows():
            # Calculate new time for the insulin dose
            new_time = idx + shift_delta
            
            # Make sure the new time exists in our dataframe
            if new_time in modified_data.index:
                modified_data.at[new_time, 'insulin'] = row['insulin']
            else:
                # Find the closest time point if exact match isn't found
                closest_idx = modified_data.index[modified_data.index.get_indexer([new_time], method='nearest')[0]]
                modified_data.at[closest_idx, 'insulin'] = row['insulin']
        
        # Recalculate glucose dynamics with the shifted insulin doses
        # Reset glucose to baseline
        modified_data['glucose'] = base_generator.params['basal_glucose']
        
        # Simulate glucose dynamics with time lags and interactions
        glucose = np.array(modified_data['glucose'])
        insulin_activity = np.zeros(len(modified_data))
        carb_impact = np.zeros(len(modified_data))
        
        # Pre-calculate all effects
        for t in range(1, len(modified_data)):
            current_time = modified_data.index[t]
            
            # Calculate lagged insulin effects
            for past_t in range(max(0, t - base_generator.params['insulin_duration']//5), t):
                if modified_data['insulin'].iloc[past_t] > 0:
                    time_diff = (t - past_t) * 5  # Convert steps to minutes
                    insulin_activity[t] += base_generator._insulin_curve(time_diff, modified_data['insulin'].iloc[past_t])
            
            # Calculate lagged carb effects
            for past_t in range(max(0, t - base_generator.params['carb_duration']//5), t):
                if modified_data['carbs'].iloc[past_t] > 0:
                    time_diff = (t - past_t) * 5  # Convert steps to minutes
                    carb_impact[t] += base_generator._carb_curve(time_diff, modified_data['carbs'].iloc[past_t])
            
            # Calculate current glucose with all effects
            exercise_effect = 1 - (modified_data['exercise'].iloc[t] * base_generator.params['exercise_sensitivity'] / 100)
            stress_effect = modified_data['stress'].iloc[t] * base_generator.params['stress_effect']
            dawn_effect = base_generator._dawn_effect(current_time.hour + current_time.minute/60)
            
            # Combine all effects with appropriate scaling and momentum
            target_glucose = (
                base_generator.params['basal_glucose']
                + carb_impact[t] * base_generator.params['carb_impact']
                - insulin_activity[t] * base_generator.params['insulin_sensitivity'] * exercise_effect
                + stress_effect
                + dawn_effect
                + base_generator.rng.normal(0, base_generator.params['noise_level'])
            )
            
            # Add momentum (glucose doesn't change instantly)
            glucose[t] = 0.9 * glucose[t-1] + 0.1 * target_glucose
        
        # Store all calculated values
        modified_data['glucose'] = np.clip(glucose, 40, 400)
        modified_data['active_insulin'] = insulin_activity
        modified_data['carb_impact'] = carb_impact
        
        # Add to datasets dictionary
        datasets[f"{shift}"] = modified_data
    
    return datasets

def plot_timing_counterfactual_comparison(datasets, title="Insulin Timing Counterfactual Comparison"):
    """
    Create a comparison plot of glucose levels across different insulin timing scenarios.
    """
    fig = go.Figure()
    
    # Define colors for different timing shifts
    colors = {
        "-60": "purple",    # Very early - purple
        "-30": "blue",      # Early - blue
        "0": "green",       # On time - green
        "30": "orange",     # Late - orange
        "60": "red",        # Very late - red
    }
    
    # Add a trace for each dataset
    for shift, df in datasets.items():
        color = colors.get(shift, "gray")
        label = "On time" if shift == "0" else f"{shift} min {'earlier' if int(shift) < 0 else 'later'}"
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['glucose'],
            name=label,
            line=dict(color=color, width=2 if shift == "0" else 1.5)
        ))
    
    # Add markers for meals (from the baseline scenario)
    baseline_df = datasets.get("0")
    if baseline_df is not None:
        meal_times = baseline_df[baseline_df['carbs'] > 0].index
        meal_values = baseline_df.loc[meal_times, 'glucose']
        meal_sizes = baseline_df.loc[meal_times, 'carbs']
        
        fig.add_trace(go.Scatter(
            x=meal_times,
            y=meal_values,
            mode='markers',
            name='Meals',
            marker=dict(
                color='green',
                symbol='triangle-up',
                size=meal_sizes/3 + 8,
            ),
            text=[f'{c}g carbs' for c in meal_sizes],
            hovertemplate='%{text}<br>Glucose: %{y:.0f} mg/dL'
        ))
        
        # Add markers for insulin doses (from the baseline scenario)
        insulin_times = baseline_df[baseline_df['insulin'] > 0].index
        insulin_values = baseline_df.loc[insulin_times, 'glucose'] 
        insulin_sizes = baseline_df.loc[insulin_times, 'insulin']
        
        fig.add_trace(go.Scatter(
            x=insulin_times,
            y=insulin_values,
            mode='markers',
            name='Baseline Insulin',
            marker=dict(
                color='red',
                symbol='triangle-down',
                size=insulin_sizes*2 + 8,  # Scale to make visible
            ),
            text=[f'{i:.1f} units' for i in insulin_sizes],
            hovertemplate='%{text}<br>Glucose: %{y:.0f} mg/dL'
        ))
    
    # Add range guidelines
    fig.add_hline(y=180, line=dict(color='red', dash='dash', width=1))
    fig.add_hline(y=70, line=dict(color='red', dash='dash', width=1))
    fig.add_hline(y=100, line=dict(color='green', dash='dot', width=1))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Glucose (mg/dL)',
        hovermode='x unified',
        showlegend=True,
        yaxis=dict(range=[40, 300]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def generate_timing_metrics(datasets):
    """
    Generate metrics comparing the different insulin timing scenarios.
    """
    metrics = []
    
    for shift, df in datasets.items():
        metrics.append({
            'timing_shift_min': shift,
            'mean_glucose': df['glucose'].mean(),
            'std_glucose': df['glucose'].std(),
            'min_glucose': df['glucose'].min(),
            'max_glucose': df['glucose'].max(),
            'time_in_range': (df['glucose'].between(70, 180)).mean() * 100,
            'time_below_range': (df['glucose'] < 70).mean() * 100,
            'time_above_range': (df['glucose'] > 180).mean() * 100,
        })
    
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

def save_timing_counterfactual_datasets(datasets, output_dir="./synthetic_data/data/timing_counterfactuals"):
    """
    Save all counterfactual datasets to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for shift, df in datasets.items():
        # Format filename differently for negative shifts
        if int(shift) < 0:
            filename = f"insulin_timing_minus{abs(int(shift))}_min.csv"
        else:
            filename = f"insulin_timing_plus{shift}_min.csv"
            
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath)
        print(f"Saved dataset to {filepath}")

if __name__ == "__main__":
    # Create output directories
    base_dir = "./synthetic_data"
    vis_dir = os.path.join(base_dir, "visualizations", "timing_counterfactuals")
    data_dir = os.path.join(base_dir, "data", "timing_counterfactuals")
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate counterfactual datasets with different insulin timing
    timing_shifts = [-60, -30, 0, 30, 60]  # Minutes before/after baseline
    counterfactual_datasets = generate_insulin_timing_counterfactuals(
        days=7,  # One week of data
        timing_shifts=timing_shifts,
        seed=42  # Fixed seed for reproducibility
    )
    
    # Save datasets
    save_timing_counterfactual_datasets(counterfactual_datasets, output_dir=data_dir)
    
    # Generate comparison visualization
    print("Generating comparison visualizations...")
    full_comparison = plot_timing_counterfactual_comparison(counterfactual_datasets)
    full_comparison.write_html(os.path.join(vis_dir, 'insulin_timing_counterfactuals_comparison.html'))
    
    # Generate a 24-hour sample for detailed view
    sample_start = list(counterfactual_datasets.values())[0].index[0] + timedelta(days=3)
    sample_end = sample_start + timedelta(days=1)
    
    sample_datasets = {}
    for shift, df in counterfactual_datasets.items():
        sample_datasets[shift] = df[sample_start:sample_end]
    
    sample_comparison = plot_timing_counterfactual_comparison(
        sample_datasets, 
        title="24-Hour Insulin Timing Comparison (Day 4)"
    )
    sample_comparison.write_html(os.path.join(vis_dir, 'insulin_timing_counterfactuals_24h_sample.html'))
    
    # Generate metrics report
    print("Calculating metrics...")
    metrics = generate_timing_metrics(counterfactual_datasets)
    metrics.to_csv(os.path.join(data_dir, 'insulin_timing_counterfactual_metrics.csv'), index=False)
    
    print("\nTiming counterfactual data generation complete!")
    print(f"Generated {len(counterfactual_datasets)} scenarios with timing shifts: {', '.join(map(str, timing_shifts))} minutes")
    print(f"Each scenario contains {len(list(counterfactual_datasets.values())[0])} data points over 7 days")
    
    print("\nFiles saved:")
    print("  Visualizations:")
    print(f"    - {os.path.join(vis_dir, 'insulin_timing_counterfactuals_comparison.html')}")
    print(f"    - {os.path.join(vis_dir, 'insulin_timing_counterfactuals_24h_sample.html')}")
    print("  Data:")
    print(f"    - {os.path.join(data_dir, 'insulin_timing_counterfactual_metrics.csv')}")
    for shift in timing_shifts:
        if shift < 0:
            filename = f"insulin_timing_minus{abs(shift)}_min.csv"
        else:
            filename = f"insulin_timing_plus{shift}_min.csv"
        print(f"    - {os.path.join(data_dir, filename)}")
