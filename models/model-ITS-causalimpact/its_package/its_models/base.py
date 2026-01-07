import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os

class BaseITSModel(ABC):
    """Base class for Interrupted Time Series models."""
    
    def __init__(self, name="BaseITSModel", output_dir=None):
        """
        Initialize the base ITS model.
        
        Parameters:
        -----------
        name : str
            Name of the model
        output_dir : str
            Directory to save outputs
        """
        self.name = name
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    @abstractmethod
    def fit(self, data, pre_period, post_period):
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
        
        Returns:
        --------
        self
        """
        pass
    
    @abstractmethod
    def get_results(self):
        """
        Get the results of the model.
        
        Returns:
        --------
        dict
            Dictionary of results
        """
        pass
    
    @abstractmethod
    def plot(self, fig=None, figsize=(12, 8)):
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
        pass
    
    def save_plot(self, filename):
        """
        Save the plot to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file
        """
        if not self.output_dir:
            raise ValueError("Output directory not set")
        
        fig = self.plot()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)
        return filepath
