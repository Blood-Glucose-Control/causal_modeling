#!/usr/bin/env python3

"""
Simplified glucose data generator for CRN testing.
Removes plotly dependency and focuses on data generation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SimpleGlucoseGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.params = {
            # Base parameters
            'basal_glucose': 100,
            'noise_level': 2,
            
            # Insulin parameters
            'insulin_sensitivity': 40,    # mg/dL drop per unit
            'insulin_peak_time': 75,      # minutes
            'insulin_duration': 300,      # minutes
            
            # Carb parameters
            'carb_ratio': 10,            # grams per unit of insulin
            'carb_impact': 4,            # mg/dL rise per gram
            'carb_peak_time': 45,        # minutes
            'carb_duration': 180,        # minutes
            
            # Exercise parameters
            'exercise_sensitivity': 20,   # % increase in insulin sensitivity
            'exercise_duration': 240,     # minutes of effect
            
            # Stress parameters
            'stress_effect': 30,         # max mg/dL rise
            'stress_duration': 180,      # minutes
            
            # Time of day effects
            'dawn_effect': 20,           # mg/dL rise
            'dawn_start': 4,             # hour
            'dawn_peak': 7,              # hour
            'dawn_end': 10               # hour
        }
    
    def _insulin_curve(self, t, dose):
        """Model insulin activity using a biexponential curve"""
        if t <= 0:
            return 0
        peak = self.params['insulin_peak_time']
        duration = self.params['insulin_duration']
        t_scaled = t / peak
        decay = np.exp(-((t_scaled - 1) ** 2) * 4)
        tail = np.exp(-t / duration)
        activity = dose * (0.8 * decay + 0.2 * tail)
        return activity * (t < duration)
    
    def _carb_curve(self, t, grams):
        """Model carb absorption using a modified exponential curve"""
        if t <= 0:
            return 0
        peak = self.params['carb_peak_time']
        duration = self.params['carb_duration']
        t_scaled = t / peak
        absorption = grams * t_scaled * np.exp(1 - t_scaled)
        return absorption * (t < duration)
    
    def _dawn_effect(self, hour):
        """Model dawn phenomenon effect"""
        if self.params['dawn_start'] <= hour <= self.params['dawn_end']:
            peak_effect = self.params['dawn_effect']
            center = (hour - self.params['dawn_start']) / (self.params['dawn_end'] - self.params['dawn_start'])
            return peak_effect * np.sin(center * np.pi)
        return 0
    
    def generate_data(self, days=1, start_date='2024-01-01'):
        # Create timestamps (5-minute intervals)
        start = pd.to_datetime(start_date)
        end = start + timedelta(days=days)
        timestamps = pd.date_range(start, end, freq='5min', inclusive='left')
        
        # Initialize dataframe with correct dtypes
        df = pd.DataFrame(index=timestamps)
        df['glucose'] = self.params['basal_glucose']
        df['carbs'] = 0
        df['insulin'] = 0.0  # Initialize as float
        df['exercise'] = 0
        df['stress'] = 0.0  # Initialize as float
        df['meal_insulin_delay'] = 0
        
        # Generate daily patterns
        for day in range(days):
            day_start = start + timedelta(days=day)
            
            # Generate meals with some randomness in timing and size
            meal_schedule = [
                (8, 40, 60),    # Breakfast: 8am ± 30min, 40-60g carbs
                (13, 50, 80),   # Lunch: 1pm ± 30min, 50-80g carbs
                (19, 45, 70)    # Dinner: 7pm ± 30min, 45-70g carbs
            ]
            
            for hour, min_carbs, max_carbs in meal_schedule:
                meal_time = day_start + timedelta(
                    hours=hour, 
                    minutes=int(self.rng.integers(-30, 30))
                )
                closest_meal_time = timestamps[abs(timestamps - meal_time).argmin()]
                
                # Add meal and insulin with some human error in carb counting
                true_carbs = int(self.rng.integers(min_carbs, max_carbs))
                counted_carbs = int(true_carbs * self.rng.normal(1, 0.1))  # 10% error in carb counting
                insulin_dose = counted_carbs / self.params['carb_ratio']
                
                # Add random timing difference between meal and insulin
                timing_diff = int(self.rng.normal(-15, 10))  # Mean: 15 min pre-bolus, SD: 10 min
                insulin_time = meal_time + timedelta(minutes=timing_diff)
                closest_insulin_time = timestamps[abs(timestamps - insulin_time).argmin()]
                
                df.loc[closest_meal_time, 'carbs'] = true_carbs
                df.loc[closest_insulin_time, 'insulin'] = float(insulin_dose)
                df.loc[closest_insulin_time, 'meal_insulin_delay'] = timing_diff
            
            # Add random exercise
            if self.rng.random() < 0.7:  # 70% chance of exercise
                exercise_time = day_start + timedelta(
                    hours=int(self.rng.integers(14, 20))  # Exercise between 2-8pm
                )
                closest_time = timestamps[abs(timestamps - exercise_time).argmin()]
                df.loc[closest_time:closest_time + timedelta(minutes=45), 'exercise'] = 1
            
            # Add random stress periods
            if self.rng.random() < 0.4:  # 40% chance of stress event
                stress_time = day_start + timedelta(
                    hours=int(self.rng.integers(9, 17))  # Stress during work hours
                )
                closest_time = timestamps[abs(timestamps - stress_time).argmin()]
                stress_value = float(self.rng.normal(0.7, 0.2))
                df.loc[closest_time:closest_time + timedelta(minutes=120), 'stress'] = stress_value
        
        # Simulate glucose dynamics with time lags and interactions
        glucose = np.array(df['glucose'])
        insulin_activity = np.zeros(len(df))
        carb_impact = np.zeros(len(df))
        
        # Pre-calculate all effects
        for t in range(1, len(df)):
            current_time = df.index[t]
            minutes_since_midnight = (current_time.hour * 60 + current_time.minute)
            
            # Calculate lagged insulin effects
            for past_t in range(max(0, t - self.params['insulin_duration']//5), t):
                if df['insulin'].iloc[past_t] > 0:
                    time_diff = (t - past_t) * 5  # Convert steps to minutes
                    insulin_activity[t] += self._insulin_curve(time_diff, df['insulin'].iloc[past_t])
            
            # Calculate lagged carb effects
            for past_t in range(max(0, t - self.params['carb_duration']//5), t):
                if df['carbs'].iloc[past_t] > 0:
                    time_diff = (t - past_t) * 5  # Convert steps to minutes
                    carb_impact[t] += self._carb_curve(time_diff, df['carbs'].iloc[past_t])
            
            # Calculate current glucose with all effects
            exercise_effect = 1 - (df['exercise'].iloc[t] * self.params['exercise_sensitivity'] / 100)
            stress_effect = df['stress'].iloc[t] * self.params['stress_effect']
            dawn_effect = self._dawn_effect(current_time.hour + current_time.minute/60)
            
            # Combine all effects with appropriate scaling and momentum
            target_glucose = (
                self.params['basal_glucose']
                + carb_impact[t] * self.params['carb_impact']
                - insulin_activity[t] * self.params['insulin_sensitivity'] * exercise_effect
                + stress_effect
                + dawn_effect
                + self.rng.normal(0, self.params['noise_level'])
            )
            
            # Add momentum (glucose doesn't change instantly)
            glucose[t] = 0.9 * glucose[t-1] + 0.1 * target_glucose
        
        # Store all calculated values
        df['glucose'] = np.clip(glucose, 40, 400)
        df['active_insulin'] = insulin_activity
        df['carb_impact'] = carb_impact
        
        return df


def create_sequences(df, sequence_length=20, prediction_horizon=5):
    """
    Create sequences for CRN training from glucose data.
    """
    
    # Define covariates (features that influence glucose but aren't treatments)
    covariate_cols = ['hour', 'day_of_week', 'carbs', 'exercise', 'stress', 
                     'active_insulin', 'carb_impact', 'meal_insulin_delay', 
                     'is_weekend', 'time_since_last_meal', 'time_since_last_insulin']
    
    # Treatment is insulin dose
    treatment_cols = ['insulin']
    
    # Output is glucose level
    output_cols = ['glucose']
    
    sequences = []
    
    # Create overlapping sequences
    for i in range(len(df) - sequence_length - prediction_horizon + 1):
        seq_data = {}
        
        # Extract sequence data
        seq_df = df.iloc[i:i + sequence_length + prediction_horizon]
        
        # Current covariates (including glucose history for context)
        current_covariates = seq_df[covariate_cols + ['glucose']].iloc[:sequence_length].values
        
        # Previous treatments (shifted by 1 timestep)
        prev_treatments = np.zeros((sequence_length, len(treatment_cols)))
        if sequence_length > 1:
            prev_treatments[1:] = seq_df[treatment_cols].iloc[:sequence_length-1].values
        
        # Current treatments
        current_treatments = seq_df[treatment_cols].iloc[:sequence_length].values
        
        # Outputs (glucose levels) - predict next timestep
        outputs = seq_df[output_cols].iloc[1:sequence_length+1].values  # Shifted by 1 for prediction
        
        # Ensure output shape matches input length
        if outputs.shape[0] != sequence_length:
            outputs = seq_df[output_cols].iloc[:sequence_length].values
        
        # Future outputs for testing counterfactuals
        future_outputs = seq_df[output_cols].iloc[sequence_length:sequence_length + prediction_horizon].values
        
        # Active entries (all 1s since we have complete data)
        active_entries = np.ones((sequence_length, len(output_cols)))
        
        seq_data['current_covariates'] = current_covariates
        seq_data['previous_treatments'] = prev_treatments
        seq_data['current_treatments'] = current_treatments
        seq_data['outputs'] = outputs
        seq_data['future_outputs'] = future_outputs
        seq_data['active_entries'] = active_entries
        
        sequences.append(seq_data)
    
    return sequences


def sequences_to_arrays(sequences):
    """
    Convert list of sequence dictionaries to numpy arrays for CRN model.
    """
    if not sequences:
        return {}
    
    # Stack all sequences
    current_covariates = np.stack([seq['current_covariates'] for seq in sequences])
    previous_treatments = np.stack([seq['previous_treatments'] for seq in sequences])
    current_treatments = np.stack([seq['current_treatments'] for seq in sequences])
    outputs = np.stack([seq['outputs'] for seq in sequences])
    active_entries = np.stack([seq['active_entries'] for seq in sequences])
    
    return {
        'current_covariates': current_covariates,
        'previous_treatments': previous_treatments,
        'current_treatments': current_treatments,
        'outputs': outputs,
        'active_entries': active_entries
    }


def get_scaling_params(training_data):
    """
    Calculate scaling parameters for normalization.
    """
    means = {}
    stds = {}
    
    # Scale glucose (outputs)
    glucose_values = training_data['outputs'].flatten()
    means['glucose'] = np.mean(glucose_values)
    stds['glucose'] = np.std(glucose_values)
    
    # Scale insulin (treatments)
    insulin_values = training_data['current_treatments'].flatten()
    means['insulin'] = np.mean(insulin_values)
    stds['insulin'] = np.std(insulin_values)
    
    # Scale covariates
    for i, covariate_name in enumerate(['hour', 'day_of_week', 'carbs', 'exercise', 'stress', 
                                       'active_insulin', 'carb_impact', 'meal_insulin_delay', 
                                       'is_weekend', 'time_since_last_meal', 'time_since_last_insulin', 'glucose_history']):
        covariate_values = training_data['current_covariates'][:, :, i].flatten()
        means[covariate_name] = np.mean(covariate_values)
        stds[covariate_name] = np.std(covariate_values)
    
    return pd.Series(means), pd.Series(stds)