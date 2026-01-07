import pandas as pd
import numpy as np
from datetime import timedelta

def to_timestamp(dt):
    """
    Convert datetime to timestamp integer for indexing
    
    Parameters:
    -----------
    dt : datetime-like object
        Datetime to convert
        
    Returns:
    --------
    int
        Timestamp in seconds since epoch
    """
    if isinstance(dt, pd.Timestamp):
        return dt.value // 10**9
    return pd.Timestamp(dt).value // 10**9

def from_timestamp(ts):
    """
    Convert timestamp integer to datetime
    
    Parameters:
    -----------
    ts : int
        Timestamp in seconds since epoch
        
    Returns:
    --------
    pandas.Timestamp
        Datetime object
    """
    return pd.Timestamp(ts, unit='s')

def convert_index_to_timestamp(df):
    """
    Convert a DataFrame's datetime index to timestamp integers
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with timestamp index
    """
    # First make a copy of the original datetime index
    df = df.copy()
    df['datetime'] = df.index.copy()
    
    # Convert index to timestamp
    df.index = df.index.astype(np.int64) // 10**9
    
    return df

def convert_index_from_timestamp(df):
    """
    Convert a DataFrame's timestamp index back to datetime
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with timestamp index
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with datetime index
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, unit='s')
    return df

def find_filtered_events(data, event_column, event_threshold=0, min_gap_seconds=3600):
    """
    Find events in the data with a minimum time gap between them
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data containing events
    event_column : str
        Column name containing event values
    event_threshold : float
        Minimum value to consider as an event
    min_gap_seconds : int
        Minimum time gap between events in seconds
        
    Returns:
    --------
    list
        List of event times
    """
    # Find all event times
    all_events = data.index[data[event_column] > event_threshold]
    
    # Filter events by minimum time gap
    filtered_events = []
    last_event = None
    
    for event in all_events:
        if last_event is None or (event - last_event).total_seconds() >= min_gap_seconds:
            filtered_events.append(event)
            last_event = event
    
    return filtered_events
