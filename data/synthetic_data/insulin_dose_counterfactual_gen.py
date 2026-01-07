import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Import the EnhancedGlucoseGenerator class from the simple_glucose_gen script
from simple_glucose_gen import EnhancedGlucoseGenerator, plot_glucose_data

def generate_insulin_counterfactuals(days=7, insulin_factors=[0.8, 0.9, 1.0, 1.1, 1.2], seed=42):
    """
    Generate multiple datasets with different insulin dosages while keeping other factors constant.
    
    Args:
        days: Number of days to simulate
        insulin_factors: List of factors to multiply insulin doses by
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of DataFrames with different insulin doses
    """
    print(f"Generating {len(insulin_factors)} counterfactual scenarios over {days} days...")
    
    # Create base generator with fixed seed
    base_generator = EnhancedGlucoseGenerator(seed=seed)
    
    # Generate base data first (no modifications)
    base_data = base_generator.generate_data(days=days)
    
    # Dictionary to store all datasets
    datasets = {"1.0": base_data.copy()}  # 1.0 is our baseline/factual scenario
    
    # For each insulin factor, create a modified dataset
    for factor in insulin_factors:
        if factor == 1.0:
            continue  # We already have the baseline
            
        print(f"Generating scenario with insulin factor: {factor:.1f}...")
        
        # Create a copy of the base data
        modified_data = base_data.copy()
        
        # Scale all insulin doses by the factor
        modified_data['insulin'] = modified_data['insulin'] * factor
        
        # Recalculate glucose dynamics with the new insulin doses
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
        datasets[f"{factor:.1f}"] = modified_data
    
    return datasets

def plot_counterfactual_comparison(datasets, title="Insulin Dose Counterfactual Comparison"):
    """
    Create a comparison plot of glucose levels across different insulin dosing scenarios.
    """
    fig = go.Figure()
    
    colors = {
        "0.8": "red",       # Low insulin - red (danger)
        "0.9": "orange",    # Slightly low insulin - orange
        "1.0": "green",     # Normal insulin - green
        "1.1": "blue",      # Slightly high insulin - blue
        "1.2": "purple"     # High insulin - purple
    }
    
    # Add a trace for each dataset
    for factor, df in datasets.items():
        color = colors.get(factor, "gray")
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['glucose'],
            name=f"Insulin × {factor}",
            line=dict(color=color, width=2 if factor == "1.0" else 1.5)
        ))
    
    # Add markers for meals (from the baseline scenario)
    baseline_df = datasets.get("1.0")
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
            name='Insulin Doses',
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

def generate_metrics(datasets):
    """
    Generate metrics comparing the different insulin dosing scenarios.
    """
    metrics = []
    
    for factor, df in datasets.items():
        metrics.append({
            'insulin_factor': factor,
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

def save_counterfactual_datasets(datasets, output_dir="./synthetic_data/data/dose_counterfactuals"):
    """
    Save all counterfactual datasets to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for factor, df in datasets.items():
        filename = f"insulin_factor_{factor.replace('.', '_')}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath)
        print(f"Saved dataset to {filepath}")

if __name__ == "__main__":
    # Create output directories
    base_dir = "./synthetic_data"
    vis_dir = os.path.join(base_dir, "visualizations", "dose_counterfactuals")
    data_dir = os.path.join(base_dir, "data", "dose_counterfactuals")
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate counterfactual datasets with different insulin factors
    insulin_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    counterfactual_datasets = generate_insulin_counterfactuals(
        days=7,  # One week of data
        insulin_factors=insulin_factors,
        seed=42  # Fixed seed for reproducibility
    )
    
    # Save datasets
    save_counterfactual_datasets(counterfactual_datasets, output_dir=data_dir)
    
    # Generate comparison visualization
    print("Generating comparison visualizations...")
    full_comparison = plot_counterfactual_comparison(counterfactual_datasets)
    full_comparison.write_html(os.path.join(vis_dir, 'insulin_counterfactuals_comparison.html'))
    
    # Generate a 24-hour sample for detailed view
    sample_start = list(counterfactual_datasets.values())[0].index[0] + timedelta(days=3)
    sample_end = sample_start + timedelta(days=1)
    
    sample_datasets = {}
    for factor, df in counterfactual_datasets.items():
        sample_datasets[factor] = df[sample_start:sample_end]
    
    sample_comparison = plot_counterfactual_comparison(
        sample_datasets, 
        title="24-Hour Insulin Dose Comparison (Day 4)"
    )
    sample_comparison.write_html(os.path.join(vis_dir, 'insulin_counterfactuals_24h_sample.html'))
    
    # Generate metrics report
    print("Calculating metrics...")
    metrics = generate_metrics(counterfactual_datasets)
    metrics.to_csv(os.path.join(data_dir, 'insulin_counterfactual_metrics.csv'), index=False)
    
    print("\nCounterfactual data generation complete!")
    print(f"Generated {len(counterfactual_datasets)} scenarios with factors: {', '.join(map(str, insulin_factors))}")
    print(f"Each scenario contains {len(list(counterfactual_datasets.values())[0])} data points over 7 days")
    
    print("\nFiles saved:")
    print("  Visualizations:")
    print(f"    - {os.path.join(vis_dir, 'insulin_counterfactuals_comparison.html')}")
    print(f"    - {os.path.join(vis_dir, 'insulin_counterfactuals_24h_sample.html')}")
    print("  Data:")
    print(f"    - {os.path.join(data_dir, 'insulin_counterfactual_metrics.csv')}")
    for factor in insulin_factors:
        print(f"    - {os.path.join(data_dir, f'insulin_factor_{str(factor).replace('.', '_')}.csv')}")
