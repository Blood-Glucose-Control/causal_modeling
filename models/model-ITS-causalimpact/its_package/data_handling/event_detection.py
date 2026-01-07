import pandas as pd
import numpy as np
from ..utils.time_utils import find_filtered_events

def detect_insulin_events(data, insulin_column='insulin', threshold=0, min_gap_hours=1):
    """
    Detect insulin administration events in the data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data containing insulin values
    insulin_column : str
        Column name for insulin values
    threshold : float
        Minimum insulin value to consider as an event
    min_gap_hours : float
        Minimum gap between events in hours
        
    Returns:
    --------
    list
        List of event times
    """
    return find_filtered_events(
        data=data,
        event_column=insulin_column,
        event_threshold=threshold,
        min_gap_seconds=min_gap_hours * 3600
    )

def detect_meal_events(data, meal_column='carbs', threshold=0, min_gap_hours=1):
    """
    Detect meal events in the data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data containing meal values
    meal_column : str
        Column name for meal values
    threshold : float
        Minimum meal value to consider as an event
    min_gap_hours : float
        Minimum gap between events in hours
        
    Returns:
    --------
    list
        List of event times
    """
    return find_filtered_events(
        data=data,
        event_column=meal_column,
        event_threshold=threshold,
        min_gap_seconds=min_gap_hours * 3600
    )

def get_event_values(data, events, column):
    """
    Get values of a column at event times
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data containing values
    events : list
        List of event times
    column : str
        Column name to get values from
        
    Returns:
    --------
    pandas.Series
        Series mapping event times to values
    """
    values = {}
    for event in events:
        if event in data.index:
            values[event] = data.loc[event, column]
        else:
            # Find closest time point if exact match not found
            closest_idx = data.index.get_indexer([event], method='nearest')[0]
            if closest_idx >= 0 and closest_idx < len(data.index):
                values[event] = data.iloc[closest_idx][column]
    
    return pd.Series(values)
