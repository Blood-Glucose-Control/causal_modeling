import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from math import sqrt
import sklearn.metrics as metrics
import warnings

# Suppress scipy stats warnings about small sample sizes
warnings.filterwarnings("ignore", message=".*p-value may be inaccurate with fewer than 20 observations.*")
# Suppress CausalImpact pandas warnings
warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated", category=FutureWarning)
# Suppress statsmodels warnings about unknown parameters
warnings.filterwarnings("ignore", message="Unknown keyword arguments", category=FutureWarning)

from ..its_models.causalimpact_model import CausalImpactModel
from ..its_models.statsmodels_its import StatsmodelsITSModel
from ..data_handling.data_loader import load_csv_data, load_counterfactual_datasets, prepare_window_data, clean_data_for_causalimpact
from ..data_handling.event_detection import detect_insulin_events, detect_meal_events
from ..visualization.plot_results import plot_event_data, plot_counterfactual_comparison, plot_comparison_across_factors
from ..utils.time_utils import to_timestamp

class CausalAnalysisPipeline:
    """Pipeline for causal analysis of time series data."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save outputs
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Initialize results storage
        self.events = []
        self.causal_impact_results = {}
        self.statsmodels_results = {}
        self.counterfactual_results = {}
        
        # Initialize results storage for evaluation
        self.evaluation_results = {
            'causalimpact': {},
            'statsmodels': {}
        }
        
    def run_single_dataset_analysis(self, data_path, max_events=5, 
                                  pre_window='45min', post_window='30min'):
        """
        Run analysis on a single dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to dataset file
        max_events : int
            Maximum number of events to analyze
        pre_window : str
            Time window before event
        post_window : str
            Time window after event
            
        Returns:
        --------
        dict
            Results of the analysis
        """
        print(f"Loading data from {data_path}")
        data = load_csv_data(data_path)
        
        print("Detecting insulin events")
        events = detect_insulin_events(data)
        print(f"Found {len(events)} insulin events")
        
        # Limit number of events
        events = events[:max_events]
        self.events = events
        
        results = []
        ci_evaluation_results = []
        sm_evaluation_results = []
        
        for i, event in enumerate(events):
            print(f"\nProcessing event {i+1}/{len(events)} at {event}")
            
            # Extract window data
            window_data = prepare_window_data(
                data, event, pre_window, post_window, ffill=True
            )
            
            if window_data.empty:
                print(f"Skipping event - no data in window")
                continue
                
            # Define pre/post periods
            closest_pre = window_data.index[window_data.index <= event].max()
            closest_post = window_data.index[window_data.index > event].min()
            
            if pd.isna(closest_pre) or pd.isna(closest_post):
                print(f"Skipping event - cannot establish pre/post boundaries")
                continue
                
            pre_period = [window_data.index.min(), closest_pre]
            post_period = [closest_post, window_data.index.max()]
            
            # Clean data for CausalImpact
            ci_data = clean_data_for_causalimpact(window_data)
            
            # Plot event data
            plot_name = f"event_{i}_data.png"
            plot_event_data(
                window_data, event, 'glucose', 
                output_dir=self.output_dir, 
                filename=plot_name,
                title=f"Event {i+1} at {event}"
            )
            
            # Run CausalImpact analysis
            ci_results = None
            metrics_dict_ci = None
            try:
                ci_model = CausalImpactModel(output_dir=self.output_dir)
                ci_model.fit(ci_data, pre_period, post_period)
                ci_results = ci_model.get_results()
                
                # Evaluate model performance
                metrics_dict_ci = self.evaluate_causalimpact_model(ci_model.impact, pre_period, post_period, event_index=i)
                
                # Save evaluation plots
                if self.output_dir:
                    self.save_evaluation_plots(ci_model.impact, i, metrics_dict_ci, model_type="causalimpact")
                
                # Store evaluation results
                ci_evaluation_results.append(metrics_dict_ci)
                
                # Save plot
                ci_model.save_plot(f"event_{i}_causalimpact.png")
                
                print(f"CausalImpact results:")
                print(f"  Average effect: {ci_results['avg_effect']:.2f}")
                print(f"  Cumulative effect: {ci_results['cum_effect']:.2f}")
                print(f"  Evaluation - MAE: {metrics_dict_ci['MAE']:.4f}, RMSE: {metrics_dict_ci['RMSE']:.4f}, R²: {metrics_dict_ci['R2']:.4f}")
                
                # Store results
                self.causal_impact_results[event] = ci_results
            except Exception as e:
                print(f"Error in CausalImpact analysis: {str(e)}")
                ci_results = None
            
            # Run Statsmodels ITS analysis
            sm_results = None
            metrics_dict_sm = None
            try:
                sm_model = StatsmodelsITSModel(output_dir=self.output_dir)
                sm_model.fit(window_data, pre_period, post_period)
                sm_results = sm_model.get_results()
                
                # Evaluate model performance
                metrics_dict_sm = self.evaluate_statsmodels_model(sm_model, window_data, event, event_index=i)
                
                # Save evaluation plots
                if self.output_dir:
                    self.save_statsmodels_evaluation_plots(sm_model, i, metrics_dict_sm)
                
                # Store evaluation results
                sm_evaluation_results.append(metrics_dict_sm)
                
                # Save plot
                sm_model.save_plot(f"event_{i}_statsmodels.png")
                
                print(f"Statsmodels ITS results:")
                print(f"  Level change: {sm_results['level_change']:.2f}")
                print(f"  Slope change: {sm_results['slope_change']:.4f}")
                print(f"  Evaluation - MAE: {metrics_dict_sm['MAE']:.4f}, RMSE: {metrics_dict_sm['RMSE']:.4f}, R²: {metrics_dict_sm['R2']:.4f}")
                
                # Store results
                self.statsmodels_results[event] = sm_results
            except Exception as e:
                print(f"Error in Statsmodels ITS analysis: {str(e)}")
                sm_results = None
            
            # Combine results - include whichever metrics are available
            event_result = {
                'event_time': event,
                'insulin_dose': window_data.loc[event, 'insulin'] if event in window_data.index else None,
                'ci_results': ci_results,
                'sm_results': sm_results,
                'ci_evaluation': metrics_dict_ci,
                'sm_evaluation': metrics_dict_sm
            }
            
            results.append(event_result)
        
        # Store evaluation results
        self.evaluation_results['causalimpact']['single'] = pd.DataFrame(ci_evaluation_results) if ci_evaluation_results else pd.DataFrame()
        self.evaluation_results['statsmodels']['single'] = pd.DataFrame(sm_evaluation_results) if sm_evaluation_results else pd.DataFrame()
        
        # Print evaluation summaries (one for each model that has results)
        if not self.evaluation_results['causalimpact']['single'].empty:
            print("\nCausalImpact Evaluation Metrics Summary:")
            print(self.evaluation_results['causalimpact']['single'][['event_index', 'MAE', 'RMSE', 'R2', 'MAPE']])
            
            # Calculate averages
            avg_metrics = {
                'MAE': self.evaluation_results['causalimpact']['single']['MAE'].mean(),
                'RMSE': self.evaluation_results['causalimpact']['single']['RMSE'].mean(),
                'R2': self.evaluation_results['causalimpact']['single']['R2'].mean(),
                'MAPE': self.evaluation_results['causalimpact']['single']['MAPE'].mean()
            }
            
            print("\nCausalImpact Average Metrics:")
            print(f"  MAE: {avg_metrics['MAE']:.4f}")
            print(f"  RMSE: {avg_metrics['RMSE']:.4f}")
            print(f"  R²: {avg_metrics['R2']:.4f}")
            print(f"  MAPE: {avg_metrics['MAPE']:.2f}%")
            
        if not self.evaluation_results['statsmodels']['single'].empty:
            print("\nStatsmodels ITS Evaluation Metrics Summary:")
            print(self.evaluation_results['statsmodels']['single'][['event_index', 'MAE', 'RMSE', 'R2', 'MAPE']])
            
            # Calculate averages
            avg_metrics = {
                'MAE': self.evaluation_results['statsmodels']['single']['MAE'].mean(),
                'RMSE': self.evaluation_results['statsmodels']['single']['RMSE'].mean(),
                'R2': self.evaluation_results['statsmodels']['single']['R2'].mean(),
                'MAPE': self.evaluation_results['statsmodels']['single']['MAPE'].mean()
            }
            
            print("\nStatsmodels Average Metrics:")
            print(f"  MAE: {avg_metrics['MAE']:.4f}")
            print(f"  RMSE: {avg_metrics['RMSE']:.4f}")
            print(f"  R²: {avg_metrics['R2']:.4f}")
            print(f"  MAPE: {avg_metrics['MAPE']:.2f}%")
        
        return results
        
    def run_counterfactual_analysis(self, data_path, file_pattern="insulin_factor_*.csv",
                                  max_events=5, pre_window='45min', post_window='30min'):
        """
        Run analysis on counterfactual datasets.
        
        Parameters:
        -----------
        data_path : str
            Path to directory with counterfactual datasets
        file_pattern : str
            Pattern to match counterfactual files
        max_events : int
            Maximum number of events to analyze
        pre_window : str
            Time window before event
        post_window : str
            Time window after event
            
        Returns:
        --------
        dict
            Results of the analysis
        """
        print(f"Loading counterfactual datasets from {data_path}")
        datasets = load_counterfactual_datasets(data_path, file_pattern)
        print(f"Loaded {len(datasets)} counterfactual datasets")
        
        # Use baseline dataset to identify events
        baseline_key = "1.0"
        if baseline_key not in datasets:
            print(f"Warning: Baseline dataset (factor {baseline_key}) not found")
            baseline_key = list(datasets.keys())[0]
            print(f"Using {baseline_key} as baseline")
        
        baseline_df = datasets[baseline_key]
        
        print("Detecting insulin events in baseline dataset")
        events = detect_insulin_events(baseline_df)
        print(f"Found {len(events)} insulin events")
        
        # Limit number of events
        events = events[:max_events]
        self.events = events
        
        # Plot overall comparison
        plot_counterfactual_comparison(
            datasets, 
            output_dir=self.output_dir,
            filename="counterfactual_comparison.png",
            title="Comparison of Counterfactual Datasets"
        )
        
        factor_results = {}
        ci_evaluation_results = {}
        sm_evaluation_results = {}
        
        # Analyze each factor
        for factor, data in datasets.items():
            print(f"\n{'='*50}")
            print(f"Analyzing insulin factor: {factor}")
            
            event_results = []
            ci_factor_eval_results = []
            sm_factor_eval_results = []
            
            for i, event in enumerate(events):
                print(f"\nProcessing event {i+1}/{len(events)} at {event}")
                
                # Extract window data
                window_data = prepare_window_data(
                    data, event, pre_window, post_window, ffill=True
                )
                
                if window_data.empty:
                    print(f"Skipping event - no data in window")
                    continue
                    
                # Define pre/post periods
                closest_pre = window_data.index[window_data.index <= event].max()
                closest_post = window_data.index[window_data.index > event].min()
                
                if pd.isna(closest_pre) or pd.isna(closest_post):
                    print(f"Skipping event - cannot establish pre/post boundaries")
                    continue
                    
                pre_period = [window_data.index.min(), closest_pre]
                post_period = [closest_post, window_data.index.max()]
                
                # Clean data for CausalImpact
                ci_data = clean_data_for_causalimpact(window_data)
                
                # Plot event data
                plot_name = f"factor_{factor}_event_{i}_data.png"
                plot_event_data(
                    window_data, event, 'glucose', 
                    output_dir=self.output_dir, 
                    filename=plot_name,
                    title=f"Factor {factor}, Event {i+1} at {event}"
                )
                
                # Run CausalImpact analysis
                try:
                    ci_model = CausalImpactModel(output_dir=self.output_dir)
                    ci_model.fit(ci_data, pre_period, post_period)
                    ci_results = ci_model.get_results()
                    
                    # Evaluate model performance
                    metrics_dict_ci = self.evaluate_causalimpact_model(ci_model.impact, pre_period, post_period, 
                                                      event_index=i, insulin_factor=factor)
                    
                    # Save evaluation plots
                    if self.output_dir:
                        self.save_evaluation_plots(ci_model.impact, i, metrics_dict_ci, factor=factor, model_type="causalimpact")
                    
                    # Save plot
                    ci_model.save_plot(f"factor_{factor}_event_{i}_causalimpact.png")
                    
                    # Store metrics
                    ci_factor_eval_results.append(metrics_dict_ci)
                    
                    # Store results
                    event_results.append({
                        'event_time': event,
                        'insulin_dose': window_data.loc[event, 'insulin'] if event in window_data.index else None,
                        'avg_effect': ci_results['avg_effect'],
                        'cum_effect': ci_results['cum_effect'],
                        'evaluation': metrics_dict_ci
                    })
                    
                    print(f"Success: Average effect: {ci_results['avg_effect']:.2f}, Cumulative effect: {ci_results['cum_effect']:.2f}")
                    print(f"Evaluation - MAE: {metrics_dict_ci['MAE']:.4f}, RMSE: {metrics_dict_ci['RMSE']:.4f}, R²: {metrics_dict_ci['R2']:.4f}")
                except Exception as e:
                    print(f"Error in CausalImpact analysis: {str(e)}")
                
                # Run Statsmodels ITS analysis
                try:
                    sm_model = StatsmodelsITSModel(output_dir=self.output_dir)
                    sm_model.fit(window_data, pre_period, post_period)
                    sm_results = sm_model.get_results()
                    
                    # Evaluate model performance
                    metrics_dict_sm = self.evaluate_statsmodels_model(sm_model, window_data, event, 
                                                      event_index=i, insulin_factor=factor)
                    
                    # Save evaluation plots
                    if self.output_dir:
                        self.save_statsmodels_evaluation_plots(sm_model, i, metrics_dict_sm, factor=factor)
                    
                    # Store metrics
                    sm_factor_eval_results.append(metrics_dict_sm)
                    
                    # Save plot
                    sm_model.save_plot(f"factor_{factor}_event_{i}_statsmodels.png")
                    
                    print(f"Statsmodels ITS results:")
                    print(f"  Level change: {sm_results['level_change']:.2f}")
                    print(f"  Slope change: {sm_results['slope_change']:.4f}")
                    print(f"  Evaluation - MAE: {metrics_dict_sm['MAE']:.4f}, RMSE: {metrics_dict_sm['RMSE']:.4f}, R²: {metrics_dict_sm['R2']:.4f}")
                    
                except Exception as e:
                    print(f"Error in Statsmodels ITS analysis: {str(e)}")
            
            # Store results for this factor
            if event_results:
                factor_results[factor] = pd.DataFrame(event_results)
                print(f"\nSummary for insulin factor {factor}:")
                print(factor_results[factor])
            
            # Store evaluation results for this factor
            if ci_factor_eval_results:
                if 'counterfactual' not in self.evaluation_results['causalimpact']:
                    self.evaluation_results['causalimpact']['counterfactual'] = {}
                self.evaluation_results['causalimpact']['counterfactual'][factor] = pd.DataFrame(ci_factor_eval_results)
                ci_evaluation_results[factor] = pd.DataFrame(ci_factor_eval_results)
            
            if sm_factor_eval_results:
                if 'counterfactual' not in self.evaluation_results['statsmodels']:
                    self.evaluation_results['statsmodels']['counterfactual'] = {}
                self.evaluation_results['statsmodels']['counterfactual'][factor] = pd.DataFrame(sm_factor_eval_results)
                sm_evaluation_results[factor] = pd.DataFrame(sm_factor_eval_results)
                
                # Print evaluation summary for this factor
                print(f"\nStatsmodels Evaluation Metrics for Factor {factor}:")
                eval_df = pd.DataFrame(sm_factor_eval_results)
                print(eval_df[['event_index', 'MAE', 'RMSE', 'R2', 'MAPE']])
                
                # Calculate averages
                avg_metrics = {
                    'MAE': eval_df['MAE'].mean(),
                    'RMSE': eval_df['RMSE'].mean(),
                    'R2': eval_df['R2'].mean(),
                    'MAPE': eval_df['MAPE'].mean()
                }
                
                print(f"Average Metrics for Factor {factor} (Statsmodels):")
                print(f"  MAE: {avg_metrics['MAE']:.4f}")
                print(f"  RMSE: {avg_metrics['RMSE']:.4f}")
                print(f"  R²: {avg_metrics['R2']:.4f}")
                print(f"  MAPE: {avg_metrics['MAPE']:.2f}%")
        
        # Store the results
        self.counterfactual_results = factor_results
        
        # Create summary dataframe
        summary = []
        ci_eval_summary = []
        sm_eval_summary = []
        
        for factor, df in factor_results.items():
            if not df.empty:
                summary.append({
                    'insulin_factor': factor,
                    'avg_effect_mean': df['avg_effect'].mean(),
                    'avg_effect_std': df['avg_effect'].std(),
                    'cum_effect_mean': df['cum_effect'].mean(),
                    'cum_effect_std': df['cum_effect'].std(),
                    'num_events': len(df)
                })
        
        # Fix evaluation metrics summary - use proper collections
        # CausalImpact evaluation summary
        for factor, df in ci_evaluation_results.items():
            if not df.empty:
                ci_eval_summary.append({
                    'insulin_factor': factor,
                    'MAE_mean': df['MAE'].mean(),
                    'MAE_std': df['MAE'].std(),
                    'RMSE_mean': df['RMSE'].mean(),
                    'RMSE_std': df['RMSE'].std(),
                    'R2_mean': df['R2'].mean(),
                    'R2_std': df['R2'].std(),
                    'MAPE_mean': df['MAPE'].mean(),
                    'MAPE_std': df['MAPE'].std(),
                    'num_events': len(df)
                })
        
        # Statsmodels evaluation summary 
        for factor, df in sm_evaluation_results.items():
            if not df.empty:
                sm_eval_summary.append({
                    'insulin_factor': factor,
                    'MAE_mean': df['MAE'].mean(),
                    'MAE_std': df['MAE'].std(),
                    'RMSE_mean': df['RMSE'].mean(),
                    'RMSE_std': df['RMSE'].std(),
                    'R2_mean': df['R2'].mean(),
                    'R2_std': df['R2'].std(),
                    'MAPE_mean': df['MAPE'].mean(),
                    'MAPE_std': df['MAPE'].std(),
                    'num_events': len(df)
                })
        
        summary_df = pd.DataFrame(summary)
        ci_eval_summary_df = pd.DataFrame(ci_eval_summary)
        sm_eval_summary_df = pd.DataFrame(sm_eval_summary)
        
        # Plot summary
        if not summary_df.empty:
            plot_comparison_across_factors(
                summary_df,
                title='Average Effect by Insulin Factor',
                output_dir=self.output_dir,
                filename="factor_effect_comparison.png"
            )
        
        # Plot evaluation metrics comparison
        if not ci_eval_summary_df.empty and self.output_dir:
            self.plot_metrics_comparison(ci_eval_summary_df, model_type="causalimpact")
        
        if not sm_eval_summary_df.empty and self.output_dir:
            self.plot_metrics_comparison(sm_eval_summary_df, model_type="statsmodels")
        
        # Print evaluation summary across factors for both models
        if not ci_eval_summary_df.empty:
            print("\nCausalImpact Evaluation Metrics Summary Across Factors:")
            print(ci_eval_summary_df[['insulin_factor', 'MAE_mean', 'RMSE_mean', 'R2_mean', 'MAPE_mean']])
            
        if not sm_eval_summary_df.empty:
            print("\nStatsmodels Evaluation Metrics Summary Across Factors:")
            print(sm_eval_summary_df[['insulin_factor', 'MAE_mean', 'RMSE_mean', 'R2_mean', 'MAPE_mean']])
        
        return {
            'factor_results': summary_df,
            'ci_evaluation': ci_eval_summary_df,
            'sm_evaluation': sm_eval_summary_df
        }
    
    def evaluate_causalimpact_model(self, impact, pre_period, post_period, event_index=None, insulin_factor=None):
        """
        Evaluate CausalImpact model performance using various metrics.
        """
        # Renamed from evaluate_model to evaluate_causalimpact_model for clarity
        # Extract actual and predicted values for the post-period
        post_data = impact.inferences.loc[impact.inferences.index >= post_period[0]]
        actual = post_data['response']
        predicted = post_data['point_pred']
        
        # Calculate metrics
        metrics_dict = {
            'event_index': event_index,
            'MAE': metrics.mean_absolute_error(actual, predicted),
            'MSE': metrics.mean_squared_error(actual, predicted),
            'RMSE': sqrt(metrics.mean_squared_error(actual, predicted)),
            'R2': metrics.r2_score(actual, predicted),
            'MAPE': np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100  # Added small constant to avoid division by zero
        }
        
        # Add insulin factor if provided
        if insulin_factor is not None:
            metrics_dict['insulin_factor'] = insulin_factor
        
        return metrics_dict
    
    def evaluate_statsmodels_model(self, model, window_data, event, event_index=None, insulin_factor=None):
        """
        Evaluate Statsmodels ITS model performance using various metrics.
        
        Parameters:
        -----------
        model : StatsmodelsITSModel
            Statsmodels ITS model to evaluate
        window_data : DataFrame
            Original data window
        event : timestamp
            Event time
        event_index : int, optional
            Index of the event
        insulin_factor : str, optional
            Insulin factor for counterfactual analysis
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Get actual and predicted values
        actual = model.data[model.target_col]
        fitted = model.results.fittedvalues
        
        # Calculate metrics
        metrics_dict = {
            'event_index': event_index,
            'MAE': metrics.mean_absolute_error(actual, fitted),
            'MSE': metrics.mean_squared_error(actual, fitted),
            'RMSE': sqrt(metrics.mean_squared_error(actual, fitted)),
            'R2': model.results.rsquared,  # Use the model's R-squared directly
            'MAPE': np.mean(np.abs((actual - fitted) / (actual + 1e-10))) * 100  # Added small constant to avoid division by zero
        }
        
        # Add insulin factor if provided
        if insulin_factor is not None:
            metrics_dict['insulin_factor'] = insulin_factor
        
        return metrics_dict
    
    def save_evaluation_plots(self, impact, event_index, metrics_dict, factor=None, model_type="causalimpact"):
        """
        Save evaluation plots for a CausalImpact model.
        """
        # Create evaluation directory if it doesn't exist
        eval_dir = os.path.join(self.output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Extract post-period data
        post_data = impact.inferences.loc[impact.inferences.index >= impact.params['post_period'][0]]
        actual = post_data['response']
        predicted = post_data['point_pred']
        
        # Add model_type to filename prefix
        prefix = f"{model_type}_"
        if factor:
            prefix += f"factor_{factor}_"
        
        # Create a residual plot
        plt.figure(figsize=(10, 6))
        residuals = actual - predicted
        plt.scatter(predicted, residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        
        title = f'Residual Plot'
        if factor:
            title += f' for Factor {factor}'
        title += f', Event {event_index}'
        
        plt.title(title)
        plt.savefig(os.path.join(eval_dir, f"{prefix}event_{event_index}_residuals.png"))
        plt.close()
        
        # Create a prediction vs actual plot
        plt.figure(figsize=(10, 6))
        plt.plot(post_data.index, actual, 'b-', label='Actual')
        plt.plot(post_data.index, predicted, 'r--', label='Predicted')
        plt.fill_between(
            post_data.index,
            post_data['point_pred_lower'],
            post_data['point_pred_upper'],
            color='r', alpha=0.2
        )
        plt.xlabel('Time')
        plt.ylabel('Glucose Level')
        
        title = f'Actual vs Predicted'
        if factor:
            title += f' for Factor {factor}'
        title += f', Event {event_index}'
        
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(eval_dir, f"{prefix}event_{event_index}_prediction.png"))
        plt.close()
    
    def save_statsmodels_evaluation_plots(self, model, event_index, metrics_dict, factor=None):
        """
        Save evaluation plots for a Statsmodels ITS model.
        
        Parameters:
        -----------
        model : StatsmodelsITSModel
            Statsmodels ITS model
        event_index : int
            Index of the event
        metrics_dict : dict
            Dictionary of evaluation metrics
        factor : str, optional
            Insulin factor for counterfactual analysis
        """
        # Create evaluation directory if it doesn't exist
        eval_dir = os.path.join(self.output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Get actual and predicted values
        data = model.data
        actual = data[model.target_col]
        fitted = model.results.fittedvalues
        
        # Prefix for filenames
        prefix = f"statsmodels_"
        if factor:
            prefix += f"factor_{factor}_"
        
        # Create a residual plot
        plt.figure(figsize=(10, 6))
        residuals = actual - fitted
        plt.scatter(range(len(fitted)), residuals)  # Using range since we need numerical x-values
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Data Point')
        plt.ylabel('Residuals')
        
        title = f'Statsmodels Residual Plot'
        if factor:
            title += f' for Factor {factor}'
        title += f', Event {event_index}'
        
        plt.title(title)
        plt.savefig(os.path.join(eval_dir, f"{prefix}event_{event_index}_residuals.png"))
        plt.close()
        
        # Create a prediction vs actual plot
        plt.figure(figsize=(10, 6))
        plt.plot(data['index'], actual, 'b-', label='Actual')
        plt.plot(data['index'], fitted, 'r--', label='Fitted')
        
        # Mark intervention
        plt.axvline(x=model.intervention_time, color='k', linestyle='--', label='Intervention')
        
        plt.xlabel('Time')
        plt.ylabel(model.target_col)
        
        title = f'Statsmodels Actual vs Fitted'
        if factor:
            title += f' for Factor {factor}'
        title += f', Event {event_index}'
        
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(eval_dir, f"{prefix}event_{event_index}_prediction.png"))
        plt.close()
    
    def plot_metrics_comparison(self, eval_summary_df, model_type="causalimpact"):
        """
        Plot comparison of evaluation metrics across factors.
        
        Parameters:
        -----------
        eval_summary_df : DataFrame
            DataFrame with evaluation metrics summary
        model_type : str
            Type of model ("causalimpact" or "statsmodels")
        """
        eval_dir = os.path.join(self.output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Plot metrics comparison
        metrics_to_plot = [('MAE_mean', 'MAE_std'), ('RMSE_mean', 'RMSE_std'), ('MAPE_mean', 'MAPE_std')]
        
        plt.figure(figsize=(14, 10))
        
        for i, (metric, std) in enumerate(metrics_to_plot):
            plt.subplot(len(metrics_to_plot), 1, i+1)
            plt.errorbar(
                eval_summary_df['insulin_factor'].astype(float),
                eval_summary_df[metric],
                yerr=eval_summary_df[std],
                fmt='o-',
                capsize=5
            )
            plt.title(f'{metric.split("_")[0]} by Insulin Factor')
            plt.xlabel('Insulin Factor')
            plt.ylabel(metric.split("_")[0])
            plt.grid(True, alpha=0.3)
        
        # Add model_type to filename prefix
        prefix = f"{model_type}_"
        
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f"{prefix}metrics_by_factor.png"))
        plt.close()
        
        # Plot R² comparison
        plt.figure(figsize=(8, 6))
        plt.errorbar(
            eval_summary_df['insulin_factor'].astype(float),
            eval_summary_df['R2_mean'],
            yerr=eval_summary_df['R2_std'],
            fmt='o-',
            capsize=5
        )
        plt.title('R² by Insulin Factor')
        plt.xlabel('Insulin Factor')
        plt.ylabel('R²')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(eval_dir, f"{prefix}r2_by_factor.png"))
        plt.close()
        
    def save_results(self, filename=None):
        """
        Save results to file.
        
        Parameters:
        -----------
        filename : str
            Filename to save results to
        """
        if not self.output_dir:
            raise ValueError("Output directory not set")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"causal_analysis_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert results to serializable format
        results = {
            'events': [str(e) for e in self.events],
            'causal_impact_results': {str(k): v for k, v in self.causal_impact_results.items()},
            'statsmodels_results': {str(k): v for k, v in self.statsmodels_results.items()},
        }
        
        # Save counterfactual results if available
        if self.counterfactual_results:
            cf_results = {}
            for factor, df in self.counterfactual_results.items():
                cf_results[factor] = df.to_dict(orient='records')
            results['counterfactual_results'] = cf_results
        
        # Add evaluation results to the output
        if self.evaluation_results:
            results['evaluation_results'] = {
                'causalimpact': {},
                'statsmodels': {}
            }
            
            # CausalImpact evaluation results
            if 'single' in self.evaluation_results['causalimpact'] and not self.evaluation_results['causalimpact']['single'].empty:
                results['evaluation_results']['causalimpact']['single'] = self.evaluation_results['causalimpact']['single'].to_dict(orient='records')
            
            if 'counterfactual' in self.evaluation_results['causalimpact']:
                cf_eval = {}
                for factor, df in self.evaluation_results['causalimpact']['counterfactual'].items():
                    if not df.empty:
                        cf_eval[factor] = df.to_dict(orient='records')
                results['evaluation_results']['causalimpact']['counterfactual'] = cf_eval
                
            # Statsmodels evaluation results
            if 'single' in self.evaluation_results['statsmodels'] and not self.evaluation_results['statsmodels']['single'].empty:
                results['evaluation_results']['statsmodels']['single'] = self.evaluation_results['statsmodels']['single'].to_dict(orient='records')
            
            if 'counterfactual' in self.evaluation_results['statsmodels']:
                cf_eval = {}
                for factor, df in self.evaluation_results['statsmodels']['counterfactual'].items():
                    if not df.empty:
                        cf_eval[factor] = df.to_dict(orient='records')
                results['evaluation_results']['statsmodels']['counterfactual'] = cf_eval
        
        # Save to JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        print(f"Results saved to {filepath}")
