import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from .base import BaseITSModel
import os
import pickle
from datetime import datetime
import warnings

# Suppress specific statsmodels warnings
warnings.filterwarnings("ignore", message="Unknown keyword arguments", category=FutureWarning)

class StatsmodelsITSModel(BaseITSModel):
    """ITS model using statsmodels."""
    
    def __init__(self, output_dir=None):
        """Initialize the statsmodels ITS model."""
        super().__init__(name="StatsmodelsITSModel", output_dir=output_dir)
        self.model = None
        self.results = None
        self.data = None
        self.target_col = None
        self.intervention_time = None
    
    def fit(self, data, pre_period, post_period, target_col='glucose', time_unit='minutes'):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to fit the model to
        pre_period : list
            [start, end] of pre-intervention period
        post_period : list
            [start, end] of post-intervention period
        target_col : str
            Name of the target column
        time_unit : str
            Unit of time for the time variable
            
        Returns:
        --------
        self
        """
        # Handle edge cases first
        if data.empty:
            raise ValueError("Empty dataset provided")
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Reset index to prepare for statsmodels
        window_data = data.reset_index()
        
        # Detect intervention time
        self.intervention_time = pre_period[1]
        
        # Create time variables for ITS analysis
        if 'index' in window_data.columns:
            # Calculate time as seconds from start, converted to specified time unit
            divisor = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}.get(time_unit, 60)
            window_data['time'] = (window_data['index'] - window_data['index'].min()).dt.total_seconds() / divisor
        else:
            # If no index column, create sequential time
            window_data['time'] = np.arange(len(window_data))
            
        # Create intervention indicator and interaction term
        window_data['post'] = (window_data['index'] > self.intervention_time).astype(int)
        window_data['time_post'] = window_data['time'] * window_data['post']
        
        # Store processed data and target column
        self.data = window_data
        self.target_col = target_col
        
        # Fit the model
        formula = f"{target_col} ~ time + post + time_post"
        model = smf.ols(formula, data=window_data)
        self.results = model.fit()
        
        return self
    
    def get_results(self):
        """
        Get the results of the model.
        
        Returns:
        --------
        dict
            Dictionary of results
        """
        if self.results is None:
            raise ValueError("Model not fitted yet")
        
        results = {
            'level_change': self.results.params['post'],
            'slope_change': self.results.params['time_post'],
            'pvalue_level': self.results.pvalues['post'],
            'pvalue_slope': self.results.pvalues['time_post'],
            'r_squared': self.results.rsquared,
            'adj_r_squared': self.results.rsquared_adj,
            'model_summary': self.results.summary()
        }
        
        return results
    
    def plot(self, fig=None, figsize=(12, 6)):
        """
        Plot the results of the model.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to plot on
        figsize : tuple
            Size of the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with plot
        """
        if self.results is None or self.data is None:
            raise ValueError("Model not fitted yet")
        
        # Generate plot
        if fig is None:
            fig = plt.figure(figsize=figsize)
        
        # Plot observed values
        plt.plot(self.data['index'], self.data[self.target_col], 'o-', 
                 alpha=0.7, label=f'Observed {self.target_col}')
        
        # Plot fitted values
        self.data['fitted'] = self.results.fittedvalues
        plt.plot(self.data['index'], self.data['fitted'], 'r--', linewidth=2, label='Fitted')
        
        # Mark intervention
        plt.axvline(x=self.intervention_time, color='k', linestyle='--', label='Intervention')
        
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel(self.target_col)
        plt.title('Interrupted Time Series Analysis')
        plt.legend(loc='best')
        
        return fig
    
    def predict(self, counterfactual=False):
        """
        Generate predictions, optionally creating counterfactual predictions
        
        Parameters:
        -----------
        counterfactual : bool
            If True, generate counterfactual predictions without intervention effects
            
        Returns:
        --------
        pandas.Series
            Series of predictions
        """
        if self.results is None or self.data is None:
            raise ValueError("Model not fitted yet")
        
        if counterfactual:
            # For counterfactual, set post and time_post to 0
            cf_data = self.data.copy()
            cf_data['post'] = 0
            cf_data['time_post'] = 0
            return self.results.predict(cf_data)
        else:
            return self.results.predict()
    
    def save_model(self, filename=None):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filename : str, optional
            Path to save the model. If None, a default name will be used.
        """
        if self.results is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statsmodels_its_model_{timestamp}.pkl"
        
        model_data = {
            'model': self.results,
            'type': 'statsmodels_its',
            'info': {
                'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'target_col': self.target_col,
                'intervention_time': self.intervention_time,
                'formula': self.formula
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
        return filename
