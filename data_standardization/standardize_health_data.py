import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import glob
import os

class HealthDataStandardizer:
    def __init__(self):
        self.timezone = pytz.timezone('America/New_York')
        self.data_frames = {}
        
    def _standardize_timestamp(self, df, date_col=None):
        """Standardize timestamps to UTC and ensure consistent format."""
        df = df.copy()  # Make a copy to avoid modifying original
        
        # If date_col is not specified, try to find it
        if date_col is None:
            date_candidates = ['date', 'timestamp', 'creation_date', 'recorded_time', 'sleep_start', 'sleep_end']
            for col in date_candidates:
                if col in df.columns:
                    date_col = col
                    break
            if date_col is None:
                raise ValueError(f"No date column found. Available columns: {df.columns.tolist()}")
        
        original_timestamps = df[date_col].copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], format='mixed')
        
        # Store original timestamps in UTC for comparison
        orig_utc = df[date_col].copy()
        if orig_utc.dt.tz is None:
            orig_utc = orig_utc.dt.tz_localize('UTC')
        else:
            orig_utc = orig_utc.dt.tz_convert('UTC')
        orig_utc = orig_utc.dt.round('min')
        
        # First localize naive timestamps to UTC, then convert any timezone-aware ones
        if df[date_col].dt.tz is None:
            df[date_col] = df[date_col].dt.tz_localize('UTC')
        df[date_col] = df[date_col].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Round to nearest minute and log any adjustments
        df[date_col] = df[date_col].dt.round('min')
        
        # Log timestamp adjustments
        adjustments = []
        for idx, (orig, final) in enumerate(zip(orig_utc, df[date_col])):
            if orig != pd.Timestamp(final).tz_localize('UTC'):
                adjustments.append({
                    'index': idx,
                    'original': original_timestamps[idx],
                    'original_utc': orig,
                    'final': final,
                    'difference_seconds': (pd.Timestamp(final).tz_localize('UTC') - orig).total_seconds()
                })
        
        if adjustments:
            print(f"\nTimestamp adjustments for {date_col}:")
            adj_df = pd.DataFrame(adjustments)
            print("\nSample of adjustments:")
            print(adj_df[['original', 'final', 'difference_seconds']].head())
            print(f"\nTotal adjustments: {len(adjustments)}")
            print(f"Max adjustment: {adj_df['difference_seconds'].abs().max():.1f} seconds")
            
            # Analyze adjustment patterns
            adj_df['adjustment_type'] = adj_df['difference_seconds'].apply(
                lambda x: 'round_up' if x > 0 else 'round_down'
            )
            print("\nAdjustment types:")
            print(adj_df['adjustment_type'].value_counts())
        
        return df
    
    def process_blood_glucose(self, pattern='Data/679372_*.csv'):
        """Process blood glucose data from multiple files."""
        print("\nProcessing blood glucose data...")
        all_bgl = []
        
        for file in glob.glob(pattern):
            print(f"\nReading {file}")
            df = pd.read_csv(file, low_memory=False)
            
            # Store original values for verification
            original_df = df.copy()
            original_bgl = original_df['bgl'].reset_index(drop=True)
            
            # Standardize timestamp only
            df = self._standardize_timestamp(df)
            df = df.sort_values('date')
            current_bgl = df['bgl'].reset_index(drop=True)
            
            # Verify no BGL values were modified
            modified_mask = current_bgl != original_bgl
            if modified_mask.any():
                print("\nWARNING: Blood glucose values were modified:")
                modified_indices = modified_mask[modified_mask].index
                comparison = pd.DataFrame({
                    'original': original_bgl[modified_indices],
                    'modified': current_bgl[modified_indices],
                    'timestamp': df.iloc[modified_indices]['date']
                })
                print("\nSample of modified values:")
                print(comparison.head())
                print(f"\nTotal modified values: {len(modified_indices)}")
            
            all_bgl.append(df)
        
        # Merge all blood glucose data
        merged_bgl = pd.concat(all_bgl, ignore_index=True)
        merged_bgl = merged_bgl.sort_values('date')
        
        self.data_frames['blood_glucose'] = merged_bgl
        return merged_bgl
    
    def process_daily_readiness(self, pattern='Data/fitness/DailyReadiness/Daily Readiness Score - *.csv'):
        """Process daily readiness scores."""
        print("\nProcessing daily readiness data...")
        all_files = glob.glob(pattern)
        
        if not all_files:
            print("No daily readiness files found")
            return None
            
        dfs = []
        for file in all_files:
            df = pd.read_csv(file)
            df = self._standardize_timestamp(df)
            dfs.append(df)
        
        self.data_frames['daily_readiness'] = pd.concat(dfs, ignore_index=True)
        return self.data_frames['daily_readiness']
    
    def process_sleep(self, profile_path='Data/fitness/SleepScore/Sleep Profile.csv', 
                     score_path='Data/fitness/SleepScore/sleep_score.csv'):
        """Process sleep data."""
        print("\nProcessing sleep data...")
        
        # Process sleep score (this is the main data we care about)
        if os.path.exists(score_path):
            print(f"Reading {score_path}")
            score_df = pd.read_csv(score_path, usecols=['timestamp', 'overall_score', 'deep_sleep_in_minutes', 'resting_heart_rate', 'restlessness'])
            score_df = self._standardize_timestamp(score_df, 'timestamp')
            self.data_frames['sleep_score'] = score_df
        else:
            print(f"Sleep score file not found: {score_path}")
        
        # Process sleep profile (only keep essential columns)
        if os.path.exists(profile_path):
            print(f"Reading {profile_path}")
            profile_df = pd.read_csv(profile_path, usecols=['creation_date', 'sleep_type', 'deep_sleep', 'rem_sleep', 'sleep_duration'])
            profile_df = self._standardize_timestamp(profile_df, 'creation_date')
            self.data_frames['sleep_profile'] = profile_df
        else:
            print(f"Sleep profile file not found: {profile_path}")
        
        return self.data_frames.get('sleep_profile'), self.data_frames.get('sleep_score')
    
    def process_temperature(self):
        """Process temperature data from device and computed temperature files."""
        print("\nProcessing temperature data...")
        
        # Process device temperature files
        device_pattern = os.path.join("Data", "fitness", "Temperature", "Device Temperature - *.csv")
        device_files = glob.glob(device_pattern)
        device_dfs = []
        
        for i, file in enumerate(device_files):
            print(f"Reading {file}")
            df = pd.read_csv(file, usecols=['recorded_time', 'temperature'])
            df['timestamp'] = pd.to_datetime(df['recorded_time'])
            df = df.drop('recorded_time', axis=1)
            device_dfs.append(df)
            
            # Periodically concatenate to save memory
            if (i + 1) % 10 == 0:
                device_dfs = [pd.concat(device_dfs, ignore_index=True)]
        
        if device_dfs:
            device_df = pd.concat(device_dfs, ignore_index=True)
            device_df = device_df.sort_values('timestamp')
            self.device_temperature_df = device_df
        
        # Process computed temperature files
        computed_pattern = os.path.join("Data", "fitness", "Temperature", "Computed Temperature - *.csv")
        computed_files = glob.glob(computed_pattern)
        computed_dfs = []
        
        for i, file in enumerate(computed_files):
            print(f"Reading {file}")
            df = pd.read_csv(file, usecols=['sleep_start', 'sleep_end', 'nightly_temperature'])
            df['start_timestamp'] = pd.to_datetime(df['sleep_start'], format='ISO8601')
            df['end_timestamp'] = pd.to_datetime(df['sleep_end'], format='ISO8601')
            df = df.rename(columns={'nightly_temperature': 'temperature'})
            df = df.drop(['sleep_start', 'sleep_end'], axis=1)
            computed_dfs.append(df)
            
            # Periodically concatenate to save memory
            if (i + 1) % 10 == 0:
                computed_dfs = [pd.concat(computed_dfs, ignore_index=True)]
        
        if computed_dfs:
            computed_df = pd.concat(computed_dfs, ignore_index=True)
            computed_df = computed_df.sort_values('start_timestamp')
            self.computed_temperature_df = computed_df
    
    def create_base_timeline(self):
        """Create minute-by-minute timeline covering all data."""
        print("\nCreating base timeline...")
        
        # Start with blood glucose data range as it's our primary focus
        if 'blood_glucose' in self.data_frames and self.data_frames['blood_glucose'] is not None:
            bgl_df = self.data_frames['blood_glucose']
            start_date = bgl_df['date'].min()
            end_date = bgl_df['date'].max()
            
            # Add a small buffer (1 day) on each end
            start_date -= pd.Timedelta(days=1)
            end_date += pd.Timedelta(days=1)
            
            # Create timeline
            timeline = pd.date_range(
                start=start_date,
                end=end_date,
                freq='min'
            )
            
            base_df = pd.DataFrame({'timestamp': timeline})
            base_df = self._standardize_timestamp(base_df, 'timestamp')
            return base_df
        else:
            raise ValueError("Blood glucose data is required to create the base timeline")
    
    def merge_all_data(self):
        """Merge all processed datasets."""
        print("\nMerging all datasets...")
        
        # Create base timeline
        final_df = self.create_base_timeline()
        print(f"Base timeline created: {len(final_df):,} minutes")
        
        # Merge blood glucose data first (our primary data)
        if 'blood_glucose' in self.data_frames:
            bgl_df = self.data_frames['blood_glucose'].copy()
            
            # Merge on exact timestamp
            final_df = pd.merge(
                final_df,
                bgl_df,
                left_on='timestamp',
                right_on='date',
                how='left'
            )
            
            # Clear memory
            del bgl_df
        
        # Merge other data types (readiness, sleep, etc.)
        for name, df in self.data_frames.items():
            if df is None or df.empty or name == 'blood_glucose':
                continue
            
            print(f"\nProcessing {name} data...")
            merge_df = df.copy()
            
            # Find timestamp columns
            time_cols = [col for col in merge_df.columns if any(
                term in col.lower() for term in ['time', 'date', 'timestamp']
            )]
            
            if time_cols:
                # Use only the first timestamp column
                primary_time_col = time_cols[0]
                
                # Merge on timestamp
                final_df = pd.merge(
                    final_df,
                    merge_df,
                    left_on='timestamp',
                    right_on=primary_time_col,
                    how='left',
                    suffixes=('', f'_{name}')
                )
                
                # Clear memory
                del merge_df
        
        # Remove any duplicate columns
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        
        print(f"\nFinal dataset shape: {final_df.shape}")
        print("\nColumns in final dataset:")
        for col in sorted(final_df.columns):
            print(f"- {col}")
        
        print("\nMissing value percentages:")
        print(final_df.isnull().mean() * 100)
        
        return final_df
    
    def process_spo2(self, pattern='Data/fitness/SPO2/Minute SpO2 - *.csv'):
        """Process SPO2 data."""
        print("\nProcessing SPO2 data...")
        all_files = glob.glob(pattern)
        
        if not all_files:
            print("No SPO2 files found")
            return None
            
        dfs = []
        for file in all_files:
            print(f"Reading {file}")
            # Only read timestamp and value columns
            df = pd.read_csv(file, usecols=['timestamp', 'value'])
            df = self._standardize_timestamp(df)
            dfs.append(df)
            
            # If we have too many DataFrames, concatenate them to save memory
            if len(dfs) >= 10:
                dfs = [pd.concat(dfs, ignore_index=True)]
        
        self.data_frames['spo2'] = pd.concat(dfs, ignore_index=True)
        return self.data_frames['spo2']
    
    def process_stress_score(self, path='Data/fitness/StressScore/Stress Score.csv'):
        """Process stress score data."""
        print("\nProcessing stress score data...")
        
        if not os.path.exists(path):
            print(f"Stress score file not found: {path}")
            return None
            
        print(f"Reading {path}")
        # Only read essential columns
        essential_cols = ['DATE', 'STRESS_SCORE', 'SLEEP_POINTS', 'RESPONSIVENESS_POINTS', 'EXERTION_POINTS', 'STATUS']
        df = pd.read_csv(path, usecols=essential_cols)
        df = self._standardize_timestamp(df, 'DATE')
        
        self.data_frames['stress_score'] = df
        return self.data_frames['stress_score']
    
    def process_hrv(self, summary_pattern='Data/fitness/HeartRateVariability/HRVSummary/Daily Heart Rate Variability Summary - *.csv',
                   details_pattern='Data/fitness/HeartRateVariability/HRVDetails/Heart Rate Variability Details - *.csv'):
        """Process heart rate variability data."""
        print("\nProcessing HRV data...")
        
        # Process HRV summary data
        summary_files = glob.glob(summary_pattern)
        if summary_files:
            print("\nProcessing HRV summary files...")
            summary_dfs = []
            for file in summary_files:
                print(f"Reading {file}")
                # Read all columns since HRV summary files are small
                df = pd.read_csv(file)
                df = self._standardize_timestamp(df)
                summary_dfs.append(df)
                
                # If we have too many DataFrames, concatenate them
                if len(summary_dfs) >= 10:
                    summary_dfs = [pd.concat(summary_dfs, ignore_index=True)]
            
            self.data_frames['hrv_summary'] = pd.concat(summary_dfs, ignore_index=True)
        else:
            print("No HRV summary files found")
        
        # Process HRV details data
        details_files = glob.glob(details_pattern)
        if details_files:
            print("\nProcessing HRV details files...")
            details_dfs = []
            for file in details_files:
                print(f"Reading {file}")
                # Only read essential columns from details files
                df = pd.read_csv(file, usecols=['timestamp', 'rmssd', 'coverage', 'low_frequency', 'high_frequency'])
                df = self._standardize_timestamp(df)
                details_dfs.append(df)
                
                # If we have too many DataFrames, concatenate them
                if len(details_dfs) >= 10:
                    details_dfs = [pd.concat(details_dfs, ignore_index=True)]
            
            self.data_frames['hrv_details'] = pd.concat(details_dfs, ignore_index=True)
        else:
            print("No HRV details files found")
        
        return self.data_frames.get('hrv_summary'), self.data_frames.get('hrv_details')
        
    def process_respiratory_rate(self, pattern='Data/fitness/HeartRateVariability/RespiratoryRateSummary/*.csv'):
        """Process respiratory rate data."""
        print("\nProcessing respiratory rate data...")
        all_files = glob.glob(pattern)
        
        if not all_files:
            print("No respiratory rate files found")
            return None
            
        dfs = []
        for file in all_files:
            print(f"Reading {file}")
            df = pd.read_csv(file)
            df = self._standardize_timestamp(df)
            dfs.append(df)
        
        self.data_frames['respiratory_rate'] = pd.concat(dfs, ignore_index=True)
        return self.data_frames['respiratory_rate']

def main():
    standardizer = HealthDataStandardizer()
    
    # Process all data types
    standardizer.process_blood_glucose()
    standardizer.process_daily_readiness()
    standardizer.process_sleep()
    standardizer.process_temperature()
    standardizer.process_spo2()
    standardizer.process_stress_score()
    standardizer.process_hrv()
    standardizer.process_respiratory_rate()
    
    # Merge all data
    merged_df = standardizer.merge_all_data()
    
    # Save merged dataset
    output_path = "merged_health_data.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged dataset saved to {output_path}")

if __name__ == "__main__":
    main() 