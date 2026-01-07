import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class InterventionPredictor:
    """
    Class for making predictions with pre-trained causal models.
    This allows using models in production to forecast post-intervention values.
    """

    def __init__(self, model_path=None):
        """
        Initialize the predictor with an optional pre-trained model.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to saved model file (.pkl)
        """
        self.model = None
        self.model_type = None
        self.prediction_horizon = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a saved model from file.
        
        Parameters:
        -----------
        model_path : str
            Path to saved model file (.pkl)
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['type']
        self.model_info = model_data.get('info', {})
        
        print(f"Loaded {self.model_type} model from {model_path}")
        return self
    
    def predict(self, pre_period_data, intervention_time, post_period_length, 
                intervention_value=None, time_frequency='5min'):
        """
        Predict post-intervention values based on pre-period data.
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Time series data for the pre-intervention period
        intervention_time : datetime or str
            Time of intervention
        post_period_length : str or int
            Length of post period (e.g., '1h', '2d' or number of points)
        intervention_value : float, optional
            Value of intervention (for models that use this)
        time_frequency : str, optional
            Frequency of time series for generated timestamps
            
        Returns:
        --------
        DataFrame
            Predicted post-intervention data
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Convert intervention_time to datetime if it's a string
        if isinstance(intervention_time, str):
            intervention_time = pd.to_datetime(intervention_time)
            
        # Generate post-period timepoints
        if isinstance(post_period_length, str):
            post_end_time = intervention_time + pd.Timedelta(post_period_length)
            post_period_index = pd.date_range(
                start=intervention_time, 
                end=post_end_time, 
                freq=time_frequency
            )
        else:
            post_period_index = pd.date_range(
                start=intervention_time, 
                periods=post_period_length+1,  # +1 to include intervention time
                freq=time_frequency
            )
        
        # Create empty dataframe for predictions
        predictions = pd.DataFrame(index=post_period_index)
        
        # Call the appropriate prediction method based on model type
        if self.model_type == 'causalimpact':
            return self._predict_with_causalimpact(pre_period_data, intervention_time, predictions)
        
        elif self.model_type == 'statsmodels_its':
            return self._predict_with_statsmodels(pre_period_data, intervention_time, predictions, intervention_value)
        
        else:
            raise ValueError(f"Prediction with model type '{self.model_type}' not implemented")
    
    def _predict_with_causalimpact(self, pre_period_data, intervention_time, predictions):
        """
        Generate predictions using a CausalImpact model.
        
        For CausalImpact, we:
        1. Use the learned relationships between response & covariates
        2. Project those into the post-period using post-period covariates
        3. Return counterfactual predictions
        """
        # Extract model components
        bsts_model = self.model.model
        
        # We need covariates for the post period
        # Here we assume they're provided or we could forecast them
        # For simplicity, we'll just use the last values from pre period
        # In production, you'd want to provide real covariate values or forecasts
        
        # Create post-period covariates (simplified approach)
        post_period_covariates = pd.DataFrame(
            index=predictions.index,
            columns=pre_period_data.columns.drop('response')  # All columns except response
        )
        
        # Fill with last values from pre-period (very simple approach)
        for col in post_period_covariates.columns:
            post_period_covariates[col] = pre_period_data[col].iloc[-1]
        
        # Combine pre-period and post-period 
        combined_data = pd.concat([
            pre_period_data,
            pd.DataFrame(index=predictions.index, columns=pre_period_data.columns)
        ]).sort_index()
        
        # Fill post-period covariates
        for col in combined_data.columns:
            if col != 'response':
                combined_data.loc[predictions.index, col] = post_period_covariates[col]
                
        # Use the original model to predict
        # This is a simplified approach - in a real implementation, you would
        # properly use the BSTS model's predict function
        pred_result = self.model.model.predict(
            combined_data, 
            combined_data.index.min(), 
            intervention_time
        )
        
        # Extract predictions for post period
        predictions['predicted'] = pred_result.iloc[-len(predictions):]
        predictions['lower'] = predictions['predicted'] * 0.9  # Simplified
        predictions['upper'] = predictions['predicted'] * 1.1  # Simplified
        
        return predictions
        
    def _predict_with_statsmodels(self, pre_period_data, intervention_time, predictions, intervention_value=None):
        """
        Generate predictions using a Statsmodels ITS model.
        
        This uses the fitted regression model to predict post-intervention values.
        """
        # For statsmodels, we need to:
        # 1. Create a dataframe with post-period time values
        # 2. Set the intervention indicator
        # 3. Use the model to predict
        
        # Prepare prediction data
        pred_data = pd.DataFrame(index=predictions.index)
        
        # Add time variable (minutes since start)
        start_time = pre_period_data.index.min()
        pred_data['time'] = (pred_data.index - start_time).total_seconds() / 60
        
        # Add post-intervention indicator
        pred_data['post'] = 1  # All points are post-intervention
        
        # Add interaction term
        pred_data['time_post'] = pred_data['time'] * pred_data['post']
        
        # Make predictions
        predictions['predicted'] = self.model.predict(pred_data)
        
        # Add confidence intervals (simplified)
        predictions['lower'] = predictions['predicted'] - 10
        predictions['upper'] = predictions['predicted'] + 10
        
        return predictions
    
    def plot_prediction(self, pre_period_data, predictions, intervention_time, 
                        target_col='glucose', figsize=(12, 6), output_path=None):
        """
        Plot pre-period data and post-period predictions
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Original pre-period data
        predictions : DataFrame
            Predicted post-period data
        intervention_time : datetime
            Time of intervention
        target_col : str
            Name of target column
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot pre-period data
        if target_col in pre_period_data.columns:
            plt.plot(pre_period_data.index, pre_period_data[target_col], 'b-', label='Historical Data')
        else:
            plt.plot(pre_period_data.index, pre_period_data['response'], 'b-', label='Historical Data')
        
        # Plot predictions
        plt.plot(predictions.index, predictions['predicted'], 'r--', label='Predicted')
        
        # Plot confidence intervals if available
        if 'lower' in predictions.columns and 'upper' in predictions.columns:
            plt.fill_between(
                predictions.index,
                predictions['lower'],
                predictions['upper'],
                color='r', alpha=0.2,
                label='Prediction Interval'
            )
        
        # Mark intervention
        plt.axvline(x=intervention_time, color='k', linestyle='-', label='Intervention')
        
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.title('Predicted Post-Intervention Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
    
    def predict_dose_counterfactuals(self, pre_period_data, intervention_time, post_period_length, 
                                    actual_dose, counterfactual_doses, time_frequency='5min'):
        """
        Predict glucose trends under different insulin doses.
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Time series data for the pre-intervention period
        intervention_time : datetime or str
            Time of intervention
        post_period_length : str or int
            Length of post period (e.g., '1h', '2d' or number of points)
        actual_dose : float
            The actual insulin dose used in the training data
        counterfactual_doses : list of float
            List of alternative doses to simulate
        time_frequency : str, optional
            Frequency of time series for generated timestamps
            
        Returns:
        --------
        dict
            Dictionary of predicted glucose trends for each dose
        """
        results = {}
        
        # First get the baseline prediction with actual dose
        baseline_prediction = self.predict(
            pre_period_data, intervention_time, post_period_length, 
            intervention_value=actual_dose, time_frequency=time_frequency
        )
        results['actual'] = baseline_prediction
        
        # Process each counterfactual dose
        for dose in counterfactual_doses:
            if self.model_type == 'causalimpact':
                # For CausalImpact, scale the effect based on dose ratio
                cf_prediction = self._predict_causalimpact_dose_counterfactual(
                    pre_period_data, intervention_time, post_period_length, 
                    actual_dose, dose, time_frequency
                )
                
            elif self.model_type == 'statsmodels_its':
                # For StatsModels, adjust coefficients based on dose ratio
                cf_prediction = self._predict_statsmodels_dose_counterfactual(
                    pre_period_data, intervention_time, post_period_length, 
                    actual_dose, dose, time_frequency
                )
                
            else:
                raise ValueError(f"Counterfactual dose prediction not implemented for model type '{self.model_type}'")
            
            results[f'dose_{dose}'] = cf_prediction
            
        return results
    
    def _predict_causalimpact_dose_counterfactual(self, pre_period_data, intervention_time, 
                                                post_period_length, actual_dose, new_dose, 
                                                time_frequency):
        """
        Predict CausalImpact counterfactual for different insulin dose.
        
        This uses a dose-response scaling approach based on the assumption that
        insulin effects are approximately proportional to dose.
        """
        # Get the counterfactual prediction (what would happen without insulin)
        no_intervention = self._get_causalimpact_counterfactual(
            pre_period_data, intervention_time, post_period_length, time_frequency
        )
        
        # Get the actual intervention prediction
        actual_prediction = self.predict(
            pre_period_data, intervention_time, post_period_length, 
            intervention_value=actual_dose, time_frequency=time_frequency
        )
        
        # Calculate the effect size from the actual dose
        effect = actual_prediction['predicted'] - no_intervention['predicted']
        
        # Scale the effect by the dose ratio
        dose_ratio = new_dose / actual_dose if actual_dose != 0 else 0
        scaled_effect = effect * dose_ratio
        
        # Create new prediction by adjusting the counterfactual
        new_prediction = no_intervention.copy()
        new_prediction['predicted'] = no_intervention['predicted'] + scaled_effect
        
        # Adjust confidence intervals
        effect_uncertainty_factor = np.sqrt(dose_ratio) # Uncertainty grows with dose
        new_prediction['lower'] = new_prediction['predicted'] - (actual_prediction['predicted'] - actual_prediction['lower']) * effect_uncertainty_factor
        new_prediction['upper'] = new_prediction['predicted'] + (actual_prediction['upper'] - actual_prediction['predicted']) * effect_uncertainty_factor
        
        return new_prediction
    
    def _get_causalimpact_counterfactual(self, pre_period_data, intervention_time, 
                                       post_period_length, time_frequency):
        """Get the CausalImpact counterfactual prediction (no intervention)"""
        # Generate post-period timepoints
        if isinstance(post_period_length, str):
            post_end_time = intervention_time + pd.Timedelta(post_period_length)
            post_period_index = pd.date_range(
                start=intervention_time, 
                end=post_end_time, 
                freq=time_frequency
            )
        else:
            post_period_index = pd.date_range(
                start=intervention_time, 
                periods=post_period_length+1,  # +1 to include intervention time
                freq=time_frequency
            )
        
        # Create dataframe for predictions
        predictions = pd.DataFrame(index=post_period_index)
        
        # Use the model's counterfactual predictions directly
        # This is what CausalImpact calculates as "point_pred"
        impact = self.model
        
        # The model predicts what would have happened without intervention
        point_preds = impact.inferences['point_pred']
        pred_lower = impact.inferences['point_pred_lower']
        pred_upper = impact.inferences['point_pred_upper']
        
        # Match the post-period timestamps - might need interpolation
        # Here using a simple approach - could be enhanced with proper interpolation
        
        # For simplicity, just extract the prediction part that matches our post period
        predictions['predicted'] = point_preds.reindex(post_period_index, method='nearest')
        predictions['lower'] = pred_lower.reindex(post_period_index, method='nearest')
        predictions['upper'] = pred_upper.reindex(post_period_index, method='nearest')
        
        return predictions
    
    def _predict_statsmodels_dose_counterfactual(self, pre_period_data, intervention_time, 
                                               post_period_length, actual_dose, new_dose, 
                                               time_frequency):
        """
        Predict StatsModels counterfactual for different insulin dose.
        
        This adjusts the level_change and slope_change coefficients proportionally
        to the dose change.
        """
        # Create prediction dataframe
        if isinstance(post_period_length, str):
            post_end_time = intervention_time + pd.Timedelta(post_period_length)
            post_period_index = pd.date_range(
                start=intervention_time, 
                end=post_end_time, 
                freq=time_frequency
            )
        else:
            post_period_index = pd.date_range(
                start=intervention_time, 
                periods=post_period_length+1,
                freq=time_frequency
            )
            
        predictions = pd.DataFrame(index=post_period_index)
        
        # Calculate time variable
        start_time = pre_period_data.index.min()
        predictions['time'] = (predictions.index - start_time).total_seconds() / 60
        
        # Get the original model parameters
        params = self.model.params.copy()
        
        # Scale intervention effects by dose ratio
        dose_ratio = new_dose / actual_dose if actual_dose != 0 else 0
        
        # Create prediction data with scaled coefficients
        pred_data = pd.DataFrame(index=post_period_index)
        pred_data['time'] = (pred_data.index - start_time).total_seconds() / 60
        pred_data['post'] = 1  # All points are post-intervention
        pred_data['time_post'] = pred_data['time'] * pred_data['post']
        
        # Original prediction formula: intercept + time*β₁ + post*β₂ + time_post*β₃
        # We scale β₂ (level change) and β₃ (slope change) by dose ratio
        
        predictions['predicted'] = (params['Intercept'] + 
                                   params['time'] * pred_data['time'] + 
                                   params['post'] * dose_ratio * pred_data['post'] + 
                                   params['time_post'] * dose_ratio * pred_data['time_post'])
        
        # Simple confidence intervals (can be refined with proper error propagation)
        uncertainty = np.sqrt(dose_ratio) * 5  # Simplified approach
        predictions['lower'] = predictions['predicted'] - uncertainty
        predictions['upper'] = predictions['predicted'] + uncertainty
        
        return predictions
        
    def plot_dose_counterfactuals(self, pre_period_data, counterfactual_results, intervention_time, 
                                target_col='glucose', figsize=(12, 8), output_path=None):
        """
        Plot pre-period data and multiple dose counterfactuals
        
        Parameters:
        -----------
        pre_period_data : DataFrame
            Original pre-period data
        counterfactual_results : dict
            Results from predict_dose_counterfactuals
        intervention_time : datetime
            Time of intervention
        target_col : str
            Name of target column
        figsize : tuple
            Figure size
        output_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot pre-period data
        if target_col in pre_period_data.columns:
            plt.plot(pre_period_data.index, pre_period_data[target_col], 'b-', label='Historical Data')
        else:
            plt.plot(pre_period_data.index, pre_period_data['response'], 'b-', label='Historical Data')
        
        # Define a color map for different doses
        colors = ['r', 'g', 'purple', 'orange', 'c', 'm', 'y', 'k']
        color_idx = 0
        
        # Plot each counterfactual prediction
        for label, prediction_df in counterfactual_results.items():
            color = colors[color_idx % len(colors)]
            if label == 'actual':
                linestyle = '-'
                label_text = 'Actual Dose'
                alpha = 0.9
                linewidth = 2
            else:
                linestyle = '--'
                dose = label.split('_')[1]
                label_text = f'Dose {dose}'
                alpha = 0.7
                linewidth = 1.5
                
            plt.plot(prediction_df.index, prediction_df['predicted'], 
                    color=color, linestyle=linestyle, label=label_text,
                    alpha=alpha, linewidth=linewidth)
            
            # Only show confidence intervals for actual prediction to avoid clutter
            if label == 'actual' and 'lower' in prediction_df.columns and 'upper' in prediction_df.columns:
                plt.fill_between(
                    prediction_df.index,
                    prediction_df['lower'],
                    prediction_df['upper'],
                    color=color, alpha=0.2
                )
                
            color_idx += 1
        
        # Mark intervention
        plt.axvline(x=intervention_time, color='k', linestyle='-', label='Intervention')
        
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.title('Predicted Glucose Response to Different Insulin Doses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
