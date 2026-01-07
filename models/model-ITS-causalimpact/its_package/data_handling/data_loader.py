import pandas as pd
import numpy as np
import os
import glob
from ..utils.time_utils import to_timestamp, convert_index_to_timestamp

def load_csv_data(file_path, index_col=0, convert_index=True):
    """
    Load data from a CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    index_col : int or str
        Index column in the CSV
    convert_index : bool
        Whether to convert the index to datetime
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    df = pd.read_csv(file_path, index_col=index_col)
    
    if convert_index:
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
    
    return df

def load_counterfactual_datasets(data_path, file_pattern="insulin_factor_*.csv"):
    """
    Load multiple counterfactual datasets
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing datasets
    file_pattern : str
        Pattern to match files
        
    Returns:
    --------
    dict
        Dictionary mapping factors to datasets
    """
    # Get list of matching files
    dataset_files = glob.glob(os.path.join(data_path, file_pattern))
    
    datasets = {}
    for file in dataset_files:
        # Extract factor from filename
        factor = os.path.basename(file).replace(
            "insulin_factor_", "").replace("_", ".").replace(".csv", "")
        
        # Load data
        df = pd.read_csv(file, index_col=0)
        df.index = pd.to_datetime(df.index)
        
        # Store dataset
        datasets[factor] = df
    
    return datasets

def prepare_window_data(data, event_time, pre_window, post_window, 
                       ffill=True, convert_timestamps=False):
    """
    Extract data around an event
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to extract window from
    event_time : datetime-like
        Time of the event
    pre_window : str
        Window size before event (e.g., '45min')
    post_window : str
        Window size after event (e.g., '30min')
    ffill : bool
        Whether to forward-fill missing values
    convert_timestamps : bool
        Whether to convert datetime index to timestamp integers
        
    Returns:
    --------
    pandas.DataFrame
        Data in the window around the event
    """
    # Define window boundaries
    window_start = event_time - pd.Timedelta(pre_window)
    window_end = event_time + pd.Timedelta(post_window)
    
    # Extract window data
    window_data = data.loc[window_start:window_end].copy()
    
    # Forward fill missing values if needed
    if ffill:
        window_data = window_data.ffill()
    
    # Convert index to timestamp if requested
    if convert_timestamps:
        window_data = convert_index_to_timestamp(window_data)
    
    return window_data

def clean_data_for_causalimpact(window_data, enforce_variability=True):
    """
    Clean data for CausalImpact analysis
    
    Parameters:
    -----------
    window_data : pandas.DataFrame
        Data to clean
    enforce_variability : bool
        Whether to enforce variability in columns
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned data
    """
    # Replace infinities with NaN
    ci_data = window_data.copy()
    ci_data = ci_data.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN
    ci_data = ci_data.dropna()
    
    # Keep only columns with variation (more than one unique value)
    if enforce_variability:
        ci_data = ci_data.loc[:, ci_data.nunique() > 1]
    
    return ci_data
