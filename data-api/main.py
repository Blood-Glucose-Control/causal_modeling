import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import uuid


# pandas settings
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.max_rows', None) # Show all rows
pd.set_option('display.max_colwidth', None) # Prevent truncation of long values
pd.set_option('display.width', 400) # Set width of the display

class GlucoseSimulator:
    """
    Simulates realistic glucose responses to insulin, food, exercise, and stress.
    
    This creates synthetic patient data that behaves like real diabetes management,
    including realistic insulin curves, carb absorption, and daily patterns.
    """
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

    def generate_patient_data(self, days=1, start_date='2024-01-01'):
        """
        Generate realistic patient glucose data with meals, insulin, exercise, and stress.
        
        Args:
            days: Number of days to simulate
            start_date: Starting date for the simulation
            
        Returns:
            DataFrame with glucose, insulin, carbs, exercise, stress columns
            
        Example:
            simulator = GlucoseSimulator()
            data = simulator.generate_patient_data(days=7, start_date='2024-01-01')
        """
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

class CounterfactualAnalyzer:
    """
    Analyzes 'what-if' scenarios for diabetes management.
    
    This can answer questions like:
    - "What if I had taken 20% more insulin?"
    - "What if I had taken insulin 30 minutes earlier?"
    """
    def __init__(self, seed=42):
        self.generator = GlucoseSimulator(seed)
    
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
    
    def _get_next_counterfactual_number(self, base_data):
        """Get the next counterfactual number based on existing counterfactuals"""
        existing_counterfactuals = [attr for attr in base_data.attrs.keys() if attr.startswith('counterfactual_cf')]
        return len(existing_counterfactuals) + 1
    
    def generate_dose_counterfactual(self, base_data, intervention_id, new_dose_factor, 
                                   before_minutes=120, after_minutes=180, num_scenarios=1):
        """
        Generate dose counterfactual and return baseline data with counterfactual columns
        
        Args:
            base_data: Original patient data (can already contain counterfactual columns)
            intervention_id: Unique ID of the specific insulin intervention to modify
            new_dose_factor: Factor to multiply the original dose by
            before_minutes: Minutes before intervention to include in results
            after_minutes: Minutes after intervention to include in results
            num_scenarios: Number of stochastic scenarios to generate
        
        Returns:
            DataFrame with original data plus counterfactual columns for changed values
        """
        target_intervention = self._get_intervention_by_id(base_data, intervention_id)
        target_time = target_intervention['timestamp']
        original_dose = target_intervention['dose']
        new_dose = original_dose * new_dose_factor
        
        # Define time window
        start_time = target_time - timedelta(minutes=before_minutes)
        end_time = target_time + timedelta(minutes=after_minutes)
        
        # Generate unique counterfactual ID and number
        counterfactual_id = str(uuid.uuid4())
        cf_number = self._get_next_counterfactual_number(base_data)
        cf_prefix = f"cf{cf_number}"
        
        # Start with the baseline data
        result_df = base_data.copy()
        
        # Generate counterfactual scenarios
        counterfactual_results = []
        
        for scenario in range(num_scenarios):
            # Create new generator with different seed for each scenario
            scenario_generator = GlucoseSimulator(seed=42 + scenario)
            
            # Modify the specific insulin dose
            modified_data = base_data.copy()
            modified_data.loc[target_time, 'insulin'] = new_dose
            
            # Recalculate glucose for entire dataset
            full_result = scenario_generator._recalculate_glucose(modified_data)
            counterfactual_results.append(full_result)
        
        # Average across scenarios if multiple scenarios
        if num_scenarios > 1:
            # Average the counterfactual results
            avg_glucose = np.mean([df['glucose'].values for df in counterfactual_results], axis=0)
            avg_active_insulin = np.mean([df['active_insulin'].values for df in counterfactual_results], axis=0)
            avg_carb_impact = np.mean([df['carb_impact'].values for df in counterfactual_results], axis=0)
            
            # Use the first scenario's structure and replace with averages
            final_counterfactual = counterfactual_results[0].copy()
            final_counterfactual['glucose'] = avg_glucose
            final_counterfactual['active_insulin'] = avg_active_insulin
            final_counterfactual['carb_impact'] = avg_carb_impact
        else:
            final_counterfactual = counterfactual_results[0]
        
        # Add counterfactual columns for changed values
        result_df[f'{cf_prefix}_insulin'] = result_df['insulin'].copy()
        result_df.loc[target_time, f'{cf_prefix}_insulin'] = new_dose
        
        # Add counterfactual columns for affected values (only in the time window)
        window_mask = (result_df.index >= start_time) & (result_df.index <= end_time)
        
        result_df[f'{cf_prefix}_glucose'] = result_df['glucose'].astype(float)
        result_df.loc[window_mask, f'{cf_prefix}_glucose'] = final_counterfactual.loc[window_mask, 'glucose']
        
        result_df[f'{cf_prefix}_active_insulin'] = result_df['active_insulin'].astype(float)
        result_df.loc[window_mask, f'{cf_prefix}_active_insulin'] = final_counterfactual.loc[window_mask, 'active_insulin']
        
        result_df[f'{cf_prefix}_carb_impact'] = result_df['carb_impact'].astype(float)
        result_df.loc[window_mask, f'{cf_prefix}_carb_impact'] = final_counterfactual.loc[window_mask, 'carb_impact']
        
        # Add metadata as DataFrame attributes
        result_df.attrs.update({
            f'counterfactual_{cf_prefix}': {
                'counterfactual_id': counterfactual_id,
                'cf_number': cf_number,
                'type': 'dose',
                'intervention_id': intervention_id,
                'target_time': target_time,
                'original_dose': original_dose,
                'new_dose': new_dose,
                'dose_factor': new_dose_factor,
                'before_minutes': before_minutes,
                'after_minutes': after_minutes,
                'num_scenarios': num_scenarios,
                'window_start': start_time,
                'window_end': end_time
            }
        })
        
        return result_df
    
    def generate_timing_counterfactual(self, base_data, intervention_id, timing_shift_minutes,
                                     before_minutes=120, after_minutes=180, num_scenarios=1):
        """
        Generate timing counterfactual and return baseline data with counterfactual columns
        
        Args:
            base_data: Original patient data (can already contain counterfactual columns)
            intervention_id: Unique ID of the specific insulin intervention to modify
            timing_shift_minutes: Minutes to shift the intervention (negative = earlier, positive = later)
            before_minutes: Minutes before intervention to include in results
            after_minutes: Minutes after intervention to include in results
            num_scenarios: Number of stochastic scenarios to generate
        
        Returns:
            DataFrame with original data plus counterfactual columns for changed values
        """
        target_intervention = self._get_intervention_by_id(base_data, intervention_id)
        original_time = target_intervention['timestamp']
        dose = target_intervention['dose']
        new_time = original_time + timedelta(minutes=timing_shift_minutes)
        
        # Define time window (around the ORIGINAL intervention time for comparison)
        start_time = original_time - timedelta(minutes=before_minutes)
        end_time = original_time + timedelta(minutes=after_minutes)
        
        # Generate unique counterfactual ID and number
        counterfactual_id = str(uuid.uuid4())
        cf_number = self._get_next_counterfactual_number(base_data)
        cf_prefix = f"cf{cf_number}"
        
        # Start with the baseline data
        result_df = base_data.copy()
        
        # Generate counterfactual scenarios
        counterfactual_results = []
        
        for scenario in range(num_scenarios):
            # Create new generator with different seed for each scenario
            scenario_generator = GlucoseSimulator(seed=42 + scenario)
            
            # Modify the intervention timing
            modified_data = base_data.copy()
            modified_data.loc[original_time, 'insulin'] = 0.0  # Remove original
            modified_data.loc[original_time, 'intervention_id'] = ''  # Remove ID
            
            # Add at new time (if it exists in the data)
            if new_time in modified_data.index:
                modified_data.loc[new_time, 'insulin'] = dose
                modified_data.loc[new_time, 'intervention_id'] = intervention_id
                actual_new_time = new_time
            else:
                # Find closest time
                closest_idx = modified_data.index[modified_data.index.get_indexer([new_time], method='nearest')[0]]
                modified_data.loc[closest_idx, 'insulin'] = dose
                modified_data.loc[closest_idx, 'intervention_id'] = intervention_id
                actual_new_time = closest_idx
            
            # Recalculate glucose for entire dataset
            full_result = scenario_generator._recalculate_glucose(modified_data)
            counterfactual_results.append((full_result, actual_new_time))
        
        # Average across scenarios if multiple scenarios
        if num_scenarios > 1:
            # Average the counterfactual results
            avg_glucose = np.mean([df[0]['glucose'].values for df in counterfactual_results], axis=0)
            avg_active_insulin = np.mean([df[0]['active_insulin'].values for df in counterfactual_results], axis=0)
            avg_carb_impact = np.mean([df[0]['carb_impact'].values for df in counterfactual_results], axis=0)
            
            # Use the first scenario's structure and replace with averages
            final_counterfactual = counterfactual_results[0][0].copy()
            final_counterfactual['glucose'] = avg_glucose
            final_counterfactual['active_insulin'] = avg_active_insulin
            final_counterfactual['carb_impact'] = avg_carb_impact
            actual_new_time = counterfactual_results[0][1]
        else:
            final_counterfactual, actual_new_time = counterfactual_results[0]
        
        # Add counterfactual columns for changed values
        result_df[f'{cf_prefix}_insulin'] = result_df['insulin'].copy()
        result_df.loc[original_time, f'{cf_prefix}_insulin'] = 0.0  # Remove from original time
        result_df.loc[actual_new_time, f'{cf_prefix}_insulin'] = dose  # Add at new time
        
        # Add counterfactual columns for affected values (only in the time window)
        window_mask = (result_df.index >= start_time) & (result_df.index <= end_time)
        
        result_df[f'{cf_prefix}_glucose'] = result_df['glucose'].astype(float)
        result_df.loc[window_mask, f'{cf_prefix}_glucose'] = final_counterfactual.loc[window_mask, 'glucose']
        
        result_df[f'{cf_prefix}_active_insulin'] = result_df['active_insulin'].astype(float)
        result_df.loc[window_mask, f'{cf_prefix}_active_insulin'] = final_counterfactual.loc[window_mask, 'active_insulin']
        
        result_df[f'{cf_prefix}_carb_impact'] = result_df['carb_impact'].astype(float)
        result_df.loc[window_mask, f'{cf_prefix}_carb_impact'] = final_counterfactual.loc[window_mask, 'carb_impact']
        
        # Add metadata as DataFrame attributes
        result_df.attrs.update({
            f'counterfactual_{cf_prefix}': {
                'counterfactual_id': counterfactual_id,
                'cf_number': cf_number,
                'type': 'timing',
                'intervention_id': intervention_id,
                'original_time': original_time,
                'new_time': actual_new_time,
                'timing_shift_minutes': timing_shift_minutes,
                'dose': dose,
                'before_minutes': before_minutes,
                'after_minutes': after_minutes,
                'num_scenarios': num_scenarios,
                'window_start': start_time,
                'window_end': end_time
            }
        })
        
        return result_df
    
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
    
    def get_counterfactual_summary(self, df):
        """Get a summary of all counterfactuals applied to the DataFrame"""
        counterfactuals = []
        for attr_name, attr_value in df.attrs.items():
            if attr_name.startswith('counterfactual_cf'):
                counterfactuals.append(attr_value)
        return counterfactuals


class DiabetesAnalyzer:
    """
    Main interface for diabetes counterfactual analysis.
    
    This is your one-stop shop for:
    1. Generating realistic patient data
    2. Analyzing 'what-if' scenarios
    3. Comparing different treatment decisions
    
    Example:
        analyzer = DiabetesAnalyzer()
        patient_data = analyzer.generate_patient_data(days=30)
        analysis = analyzer.analyze_dose_change(patient_data, intervention_id, factor=1.2)
    """
    
    def __init__(self, seed=42):
        self.counterfactual_model = CounterfactualAnalyzer(seed)
    
    def generate_patient_data(self, n_days=30, start_date='2024-01-01'):
        """Generate synthetic patient data with realistic glucose patterns"""
        return self.counterfactual_model.generator.generate_patient_data(days=n_days, start_date=start_date)

    def analyze_intervention(self, patient_data, intervention_id, 
                           analysis_type='dose', before_minutes=120, after_minutes=180, **kwargs):
        """
        Analyze what would happen if you changed an insulin intervention.
        
        Args:
            patient_data: Patient glucose data from generate_patient_data()
            intervention_id: ID of the insulin dose to analyze (get from list_interventions())
            analysis_type: 'dose' or 'timing'
            before_minutes: How many minutes before intervention to analyze
            after_minutes: How many minutes after intervention to analyze
            **kwargs: For dose analysis: dose_factor (e.g., 1.2 = 20% more insulin)
                     For timing analysis: timing_shift_minutes (e.g., -30 = 30 min earlier)
        
        Returns:
            DataFrame with original and counterfactual data
        """

        if analysis_type == 'dose':
            if 'dose_factor' not in kwargs:
                raise ValueError("dose_factor is required for dose analysis (e.g., dose_factor=1.2 for 20% more insulin)")
            dose_factor = kwargs['dose_factor']
            return self.counterfactual_model.generate_dose_counterfactual(
                patient_data, intervention_id, dose_factor, before_minutes, after_minutes
            )
        elif analysis_type == 'timing':
            if 'timing_shift_minutes' not in kwargs:
                raise ValueError("timing_shift_minutes is required for timing analysis (e.g., timing_shift_minutes=-30 for 30 min earlier)")
            timing_shift = kwargs['timing_shift_minutes']
            return self.counterfactual_model.generate_timing_counterfactual(
                patient_data, intervention_id, timing_shift, before_minutes, after_minutes
            )
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}. Use 'dose' or 'timing'")


if __name__ == "__main__":
    # Example usage of the ML Evaluation Service

    
    # Initialize the analyzer
    analyzer = DiabetesAnalyzer(seed=42)
    
    # Generate patient history
    print("=== Diabetes Counterfactual Analysis Demo ===")
    print("\n1. Generating 30 days of patient data...")
    patient_data = analyzer.generate_patient_data(n_days=30, start_date='2024-01-01')
    
    print(f"✓ Generated {len(patient_data)} data points ({patient_data.shape[0] / 288:.1f} days)")
    print(f"✓ Found {(patient_data['insulin'] > 0).sum()} insulin interventions")

    # List interventions
    interventions = analyzer.counterfactual_model.list_interventions(patient_data)
    print(f"\n2. Available insulin interventions:")
    for i, intv in enumerate(interventions[:3]):  # Show first 3
        print(f"  {i+1}. {intv['timestamp'].strftime('%Y-%m-%d %H:%M')} - {intv['dose']:.1f} units")
    if len(interventions) > 3:
        print(f"  ... and {len(interventions)-3} more")

    # Analyze what-if scenario
    print(f"\n3. Analyzing: What if we gave 20% more insulin?")
    chosen_intervention_id = interventions[0]['id']
    analysis_result = analyzer.analyze_intervention(
        patient_data,
        intervention_id=chosen_intervention_id,
        analysis_type='dose',
        before_minutes=120,
        after_minutes=180,
        dose_factor=1.2
    )
    
    # Show the results
    cf_meta = analysis_result.attrs['counterfactual_cf1']
    window_mask = (analysis_result.index >= cf_meta['window_start']) & (analysis_result.index <= cf_meta['window_end'])
    comparison_df = analysis_result.loc[window_mask].drop(columns=['meal_insulin_delay', 'active_insulin', 'carb_impact', 'cf1_active_insulin', 'cf1_carb_impact'])
    
    print(f"\n4. Results: Original vs 20% More Insulin")
    print(f"   Original dose: {cf_meta['original_dose']:.1f} units")
    print(f"   New dose: {cf_meta['new_dose']:.1f} units")
    print(f"   Time window: {cf_meta['before_minutes']} min before to {cf_meta['after_minutes']} min after")
    print(f"\n   Comparison data:")
    print(comparison_df.round(1))