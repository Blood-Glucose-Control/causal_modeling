import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import uuid

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
    
    def _recalculate_glucose(self, df):
        """Recalculate glucose dynamics for modified data"""
        glucose = np.full(len(df), self.params['basal_glucose'])
        insulin_activity = np.zeros(len(df))
        carb_impact = np.zeros(len(df))
        
        for t in range(1, len(df)):
            current_time = df.index[t]
            
            # Calculate lagged insulin effects
            for past_t in range(max(0, t - self.params['insulin_duration']//5), t):
                if df['insulin'].iloc[past_t] > 0:
                    time_diff = (t - past_t) * 5
                    insulin_activity[t] += self._insulin_curve(time_diff, df['insulin'].iloc[past_t])
            
            # Calculate lagged carb effects
            for past_t in range(max(0, t - self.params['carb_duration']//5), t):
                if df['carbs'].iloc[past_t] > 0:
                    time_diff = (t - past_t) * 5
                    carb_impact[t] += self._carb_curve(time_diff, df['carbs'].iloc[past_t])
            
            # Calculate current glucose with all effects
            exercise_effect = 1 - (df['exercise'].iloc[t] * self.params['exercise_sensitivity'] / 100)
            stress_effect = df['stress'].iloc[t] * self.params['stress_effect']
            dawn_effect = self._dawn_effect(current_time.hour + current_time.minute/60)
            
            target_glucose = (
                self.params['basal_glucose']
                + carb_impact[t] * self.params['carb_impact']
                - insulin_activity[t] * self.params['insulin_sensitivity'] * exercise_effect
                + stress_effect
                + dawn_effect
                + self.rng.normal(0, self.params['noise_level'])
            )
            
            glucose[t] = 0.9 * glucose[t-1] + 0.1 * target_glucose
        
        df['glucose'] = np.clip(glucose, 40, 400)
        df['active_insulin'] = insulin_activity
        df['carb_impact'] = carb_impact
        return df

    def generate_data(self, days=1, start_date='2024-01-01'):
        # Create timestamps (5-minute intervals)
        start = pd.to_datetime(start_date)
        end = start + timedelta(days=days)
        timestamps = pd.date_range(start, end, freq='5min', inclusive='left')
        
        # Initialize dataframe with correct dtypes
        df = pd.DataFrame(index=timestamps)
        df['glucose'] = self.params['basal_glucose']
        df['carbs'] = 0
        df['insulin'] = 0.0
        df['exercise'] = 0
        df['stress'] = 0.0
        df['meal_insulin_delay'] = 0
        df['intervention_id'] = ''  # Add intervention ID column
        
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
                counted_carbs = int(true_carbs * self.rng.normal(1, 0.1))
                insulin_dose = counted_carbs / self.params['carb_ratio']
                
                # Add random timing difference between meal and insulin
                timing_diff = int(self.rng.normal(-15, 10))
                insulin_time = meal_time + timedelta(minutes=timing_diff)
                closest_insulin_time = timestamps[abs(timestamps - insulin_time).argmin()]
                
                # Generate unique intervention ID for this insulin dose
                intervention_id = str(uuid.uuid4())
                
                df.loc[closest_meal_time, 'carbs'] = true_carbs
                df.loc[closest_insulin_time, 'insulin'] = float(insulin_dose)
                df.loc[closest_insulin_time, 'meal_insulin_delay'] = timing_diff
                df.loc[closest_insulin_time, 'intervention_id'] = intervention_id
            
            # Add random exercise
            if self.rng.random() < 0.7:
                exercise_time = day_start + timedelta(
                    hours=int(self.rng.integers(14, 20))
                )
                closest_time = timestamps[abs(timestamps - exercise_time).argmin()]
                df.loc[closest_time:closest_time + timedelta(minutes=45), 'exercise'] = 1
            
            # Add random stress periods
            if self.rng.random() < 0.4:
                stress_time = day_start + timedelta(
                    hours=int(self.rng.integers(9, 17))
                )
                closest_time = timestamps[abs(timestamps - stress_time).argmin()]
                stress_value = float(self.rng.normal(0.7, 0.2))
                df.loc[closest_time:closest_time + timedelta(minutes=120), 'stress'] = stress_value
        
        return self._recalculate_glucose(df)

class CounterfactualService:
    def __init__(self, seed=42):
        self.generator = EnhancedGlucoseGenerator(seed)
    
    def _get_intervention_by_id(self, base_data, intervention_id):
        """Get intervention details by ID"""
        insulin_interventions = base_data[base_data['intervention_id'] != '']
        
        for timestamp, row in insulin_interventions.iterrows():
            if row['intervention_id'] == intervention_id:
                return {
                    'id': intervention_id,
                    'timestamp': timestamp,
                    'dose': row['insulin'],
                    'carbs_nearby': row['carbs'] if row['carbs'] > 0 else None
                }
        
        raise ValueError(f"Intervention ID '{intervention_id}' not found")
    
    def generate_dose_counterfactual(self, base_data, intervention_id, new_dose_factor, 
                                   before_minutes=120, after_minutes=180, num_scenarios=1):
        """
        Generate counterfactual for a specific intervention with modified dose
        
        Args:
            base_data: Original patient data
            intervention_id: Unique ID of the specific insulin intervention to modify
            new_dose_factor: Factor to multiply the original dose by
            before_minutes: Minutes before intervention to include in results
            after_minutes: Minutes after intervention to include in results
            num_scenarios: Number of stochastic scenarios to generate
        """
        target_intervention = self._get_intervention_by_id(base_data, intervention_id)
        target_time = target_intervention['timestamp']
        original_dose = target_intervention['dose']
        new_dose = original_dose * new_dose_factor
        
        # Define time window
        start_time = target_time - timedelta(minutes=before_minutes)
        end_time = target_time + timedelta(minutes=after_minutes)
        
        results = []
        
        for scenario in range(num_scenarios):
            # Create new generator with different seed for each scenario
            scenario_generator = EnhancedGlucoseGenerator(seed=42 + scenario)
            
            # Modify the specific insulin dose
            modified_data = base_data.copy()
            modified_data.loc[target_time, 'insulin'] = new_dose
            
            # Recalculate glucose for entire dataset
            full_result = scenario_generator._recalculate_glucose(modified_data)
            
            # Extract the time window around the intervention
            window_result = full_result[start_time:end_time].copy()
            
            # Add metadata
            window_result.attrs = {
                'intervention_id': intervention_id,
                'target_time': target_time,
                'original_dose': original_dose,
                'new_dose': new_dose,
                'dose_factor': new_dose_factor,
                'before_minutes': before_minutes,
                'after_minutes': after_minutes,
                'scenario': scenario
            }
            
            results.append(window_result)
        
        return results
    
    def generate_timing_counterfactual(self, base_data, intervention_id, timing_shift_minutes,
                                     before_minutes=120, after_minutes=180, num_scenarios=1):
        """
        Generate counterfactual for a specific intervention with modified timing
        
        Args:
            base_data: Original patient data
            intervention_id: Unique ID of the specific insulin intervention to modify
            timing_shift_minutes: Minutes to shift the intervention (negative = earlier, positive = later)
            before_minutes: Minutes before intervention to include in results
            after_minutes: Minutes after intervention to include in results
            num_scenarios: Number of stochastic scenarios to generate
        """
        target_intervention = self._get_intervention_by_id(base_data, intervention_id)
        original_time = target_intervention['timestamp']
        dose = target_intervention['dose']
        new_time = original_time + timedelta(minutes=timing_shift_minutes)
        
        # Define time window (around the ORIGINAL intervention time for comparison)
        start_time = original_time - timedelta(minutes=before_minutes)
        end_time = original_time + timedelta(minutes=after_minutes)
        
        results = []
        
        for scenario in range(num_scenarios):
            # Create new generator with different seed for each scenario
            scenario_generator = EnhancedGlucoseGenerator(seed=42 + scenario)
            
            # Modify the intervention timing
            modified_data = base_data.copy()
            modified_data.loc[original_time, 'insulin'] = 0.0  # Remove original
            modified_data.loc[original_time, 'intervention_id'] = ''  # Remove ID
            
            # Add at new time (if it exists in the data)
            if new_time in modified_data.index:
                modified_data.loc[new_time, 'insulin'] = dose
                modified_data.loc[new_time, 'intervention_id'] = intervention_id
            else:
                # Find closest time
                closest_idx = modified_data.index[modified_data.index.get_indexer([new_time], method='nearest')[0]]
                modified_data.loc[closest_idx, 'insulin'] = dose
                modified_data.loc[closest_idx, 'intervention_id'] = intervention_id
                new_time = closest_idx  # Update to actual time used
            
            # Recalculate glucose for entire dataset
            full_result = scenario_generator._recalculate_glucose(modified_data)
            
            # Extract the time window around the ORIGINAL intervention time
            window_result = full_result[start_time:end_time].copy()
            
            # Add metadata
            window_result.attrs = {
                'intervention_id': intervention_id,
                'original_time': original_time,
                'new_time': new_time,
                'timing_shift_minutes': timing_shift_minutes,
                'dose': dose,
                'before_minutes': before_minutes,
                'after_minutes': after_minutes,
                'scenario': scenario
            }
            
            results.append(window_result)
        
        return results
    
    def list_interventions(self, base_data):
        """List all available interventions with their details"""
        interventions = []
        insulin_interventions = base_data[base_data['intervention_id'] != '']
        
        for timestamp, row in insulin_interventions.iterrows():
            interventions.append({
                'id': row['intervention_id'],
                'timestamp': timestamp,
                'dose': row['insulin'],
                'carbs_nearby': row['carbs'] if row['carbs'] > 0 else None
            })
        
        return interventions

def test_service():
    global service, base_data, dose_results, dose_results_df
    """Test the counterfactual service"""
    
    # Generate base patient history (7 days)
    service = CounterfactualService()
    base_data = service.generator.generate_data(days=7)
    
    print("Base patient data generated:")
    print(f"Shape (rows, cols): {base_data.shape}")
    print(f"Total insulin doses: {(base_data['insulin'] > 0).sum()}\n")
    
    # List available interventions
    interventions = service.list_interventions(base_data)
    print("Available interventions:")
    for intervention in interventions:
        print(f"ID {intervention['id']}: {intervention['timestamp']} - {intervention['dose']:.1f}u insulin")
    print()
    
    # Test dose counterfactual for first intervention
    if interventions:
        print(f"Dose counterfactual (ID {interventions[0]['id']}, 1.25x dose):")
        dose_results = service.generate_dose_counterfactual(
            base_data, 
            intervention_id=interventions[0]['id'],
            new_dose_factor=1.25,
            before_minutes=60,
            after_minutes=120,
            num_scenarios=2
        )
        dose_results_df = pd.concat(dose_results)
        
        print(f"Generated {len(dose_results)} counterfactual scenarios")
        print(f"Combined result shape: {dose_results_df.shape}")

if __name__ == "__main__":
    test_service() 