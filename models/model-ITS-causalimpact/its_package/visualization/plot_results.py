import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_comparison_across_factors(summary_df, x_col='insulin_factor', y_col='avg_effect_mean', 
                                  err_col='avg_effect_std', title='', 
                                  xlabel='Insulin Factor', ylabel='Effect',
                                  output_dir=None, filename=None):
    """
    Plot comparison of effects across different factors
    
    Parameters:
    -----------
    summary_df : pandas.DataFrame
        Summary dataframe with results
    x_col : str
        Column name for x-axis values
    y_col : str
        Column name for y-axis values
    err_col : str
        Column name for error bars
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    output_dir : str
        Directory to save output
    filename : str
        Filename to save plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with plot
    """
    fig = plt.figure(figsize=(10, 6))
    
    plt.errorbar(
        summary_df[x_col].astype(float),
        summary_df[y_col],
        yerr=summary_df[err_col] if err_col in summary_df else None,
        fmt='o-',
        capsize=5
    )
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    
    # Save plot if directory and filename provided
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
    
    return fig

def plot_event_data(data, event_time, target_col='glucose', 
                   window_start=None, window_end=None,
                   figsize=(12, 6), title=None, 
                   output_dir=None, filename=None):
    """
    Plot data around an event
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to plot
    event_time : datetime-like
        Time of the event
    target_col : str
        Column name to plot
    window_start : datetime-like
        Start of window to plot
    window_end : datetime-like
        End of window to plot
    figsize : tuple
        Size of the figure
    title : str
        Plot title
    output_dir : str
        Directory to save output
    filename : str
        Filename to save plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with plot
    """
    fig = plt.figure(figsize=figsize)
    
    # Filter data to window if specified
    if window_start is not None and window_end is not None:
        plot_data = data.loc[window_start:window_end]
    else:
        plot_data = data
    
    # Plot target variable
    plt.plot(plot_data.index, plot_data[target_col], 'b-', label=target_col)
    
    # Mark event time
    plt.axvline(x=event_time, color='r', linestyle='--', label='Event')
    
    # Add title and labels
    if title:
        plt.title(title)
    else:
        plt.title(f'Data around event at {event_time}')
    
    plt.ylabel(target_col)
    plt.legend()
    
    # Save plot if directory and filename provided
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
    
    return fig

def plot_counterfactual_comparison(datasets, event_time=None, target_col='glucose',
                                 window_hours=12, figsize=(12, 6),
                                 title='Counterfactual Comparison',
                                 output_dir=None, filename=None):
    """
    Plot comparison of counterfactual datasets
    
    Parameters:
    -----------
    datasets : dict
        Dictionary mapping factors to datasets
    event_time : datetime-like
        Time of the event to center around (optional)
    target_col : str
        Column name to plot
    window_hours : float
        Hours to plot before and after event
    figsize : tuple
        Size of the figure
    title : str
        Plot title
    output_dir : str
        Directory to save output
    filename : str
        Filename to save plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with plot
    """
    fig = plt.figure(figsize=figsize)
    
    # Define color map for factors
    colors = {
        "0.8": "red", "0.9": "orange", "1.0": "green", 
        "1.1": "blue", "1.2": "purple"
    }
    
    # Define window if event_time is provided
    if event_time:
        window_start = event_time - pd.Timedelta(hours=window_hours)
        window_end = event_time + pd.Timedelta(hours=window_hours)
    
    # Plot each dataset
    for factor, df in datasets.items():
        # Filter data to window if event_time is provided
        if event_time:
            plot_data = df.loc[window_start:window_end]
        else:
            plot_data = df
            
        color = colors.get(factor, "gray")
        label = f"Factor {factor}"
        plt.plot(
            plot_data.index, 
            plot_data[target_col], 
            label=label, 
            color=color, 
            linewidth=2 if factor == "1.0" else 1.5
        )
    
    # Mark event time if provided
    if event_time:
        plt.axvline(x=event_time, color='k', linestyle='--', label='Event')
    
    # Add guidelines for glucose ranges
    plt.axhline(y=180, color='red', linestyle='--', alpha=0.7, label='High threshold')
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Low threshold')
    plt.axhline(y=100, color='green', linestyle=':', alpha=0.7, label='Target')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(f'{target_col} (mg/dL)')
    plt.legend()
    
    # Save plot if directory and filename provided
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
    
    return fig
