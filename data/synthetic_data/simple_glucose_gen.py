import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

class EnhancedGlucoseGenerator:
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

def plot_glucose_data(df):
    fig = go.Figure()
    
    # Plot glucose line
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['glucose'],
        name='Glucose',
        line=dict(color='blue', width=2)
    ))
    
    # Plot meal markers
    meal_times = df[df['carbs'] > 0].index
    meal_values = df.loc[meal_times, 'glucose']
    meal_sizes = df.loc[meal_times, 'carbs']
    fig.add_trace(go.Scatter(
        x=meal_times,
        y=meal_values,
        mode='markers',
        name='Meals',
        marker=dict(
            color='green',
            symbol='triangle-up',
            size=meal_sizes/3 + 8,  # Size proportional to meal size
        ),
        text=[f'{c}g carbs' for c in meal_sizes],
        hovertemplate='%{text}<br>Glucose: %{y:.0f} mg/dL'
    ))
    
    # Plot insulin markers
    insulin_times = df[df['insulin'] > 0].index
    insulin_values = df.loc[insulin_times, 'glucose']
    insulin_doses = df.loc[insulin_times, 'insulin']
    fig.add_trace(go.Scatter(
        x=insulin_times,
        y=insulin_values,
        mode='markers',
        name='Insulin',
        marker=dict(
            color='red',
            symbol='triangle-down',
            size=insulin_doses*2 + 8,  # Size proportional to insulin dose
        ),
        text=[f'{d:.1f}u insulin' for d in insulin_doses],
        hovertemplate='%{text}<br>Glucose: %{y:.0f} mg/dL'
    ))
    
    # Add exercise periods
    exercise_periods = df[df['exercise'] > 0].index
    if len(exercise_periods) > 0:
        fig.add_trace(go.Scatter(
            x=exercise_periods,
            y=df.loc[exercise_periods, 'glucose'],
            mode='markers',
            name='Exercise',
            marker=dict(color='purple', symbol='square', size=8)
        ))
    
    # Add stress periods
    stress_periods = df[df['stress'] > 0].index
    if len(stress_periods) > 0:
        fig.add_trace(go.Scatter(
            x=stress_periods,
            y=df.loc[stress_periods, 'glucose'],
            mode='markers',
            name='Stress',
            marker=dict(color='orange', symbol='diamond', size=8)
        ))
    
    # Add range guidelines
    fig.add_hline(y=180, line=dict(color='red', dash='dash', width=1))
    fig.add_hline(y=70, line=dict(color='red', dash='dash', width=1))
    fig.add_hline(y=100, line=dict(color='green', dash='dot', width=1))
    
    fig.update_layout(
        title='Enhanced Synthetic Glucose Data',
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

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Generate larger synthetic dataset (90 days)
    generator = EnhancedGlucoseGenerator()
    data = generator.generate_data(days=90)
    
    print("Generating visualizations...")
    
    # Full timeline visualization
    full_fig = plot_glucose_data(data)
    full_fig.write_html('visualizations/full_timeline.html')
    
    # Sample 3-day period for detailed view
    import random
    sample_start = data.index[0] + timedelta(days=random.randint(0, 85))
    sample_end = sample_start + timedelta(days=3)
    sample_data = data[sample_start:sample_end]
    
    sample_fig = plot_glucose_data(sample_data)
    sample_fig.update_layout(title="3-Day Sample for Counterfactual Analysis")
    sample_fig.write_html('visualizations/sample_period.html')
    
    print("Preparing ML dataset...")
    
    # Create feature-engineered version for ML
    ml_data = pd.DataFrame(index=data.index)
    
    # Basic features
    ml_data['glucose'] = data['glucose']
    ml_data['hour'] = data.index.hour
    ml_data['day_of_week'] = data.index.dayofweek
    ml_data['carbs'] = data['carbs']
    ml_data['insulin'] = data['insulin']
    ml_data['exercise'] = data['exercise']
    ml_data['stress'] = data['stress']
    ml_data['active_insulin'] = data['active_insulin']
    ml_data['carb_impact'] = data['carb_impact']
    ml_data['meal_insulin_delay'] = data['meal_insulin_delay']
    
    # Additional engineered features
    ml_data['is_weekend'] = ml_data['day_of_week'].isin([5, 6]).astype(int)
    ml_data['time_since_last_meal'] = ml_data['carbs'].ne(0).astype(int).groupby(ml_data.index.date).cumsum()
    ml_data['time_since_last_insulin'] = ml_data['insulin'].ne(0).astype(int).groupby(ml_data.index.date).cumsum()
    
    # Save datasets
    print("Saving datasets...")
    data.to_csv('data/full_dataset.csv')
    ml_data.to_csv('data/ml_dataset.csv')
    
    print("\nData generation complete!")
    print(f"Generated {len(data):,} data points over {(data.index[-1] - data.index[0]).days} days")
    print("\nFiles saved:")
    print("  Visualizations:")
    print("    - visualizations/full_timeline.html")
    print("    - visualizations/sample_period.html")
    print("  Data:")
    print("    - data/full_dataset.csv")
    print("    - data/ml_dataset.csv") 