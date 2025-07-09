import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import glob
import os
import sys

class HealthDataStandardizer:
    def __init__(self):
        self.data_frames = {}
        self.DATA_PATHS = {
            'blood_glucose': {
                'pattern': 'data/679372_*.csv'
            },
            'daily_readiness': {
                'pattern': 'data/fitness/DailyReadiness/Daily Readiness Score - *.csv'
            },
            'sleep': {
                'profile': 'data/fitness/SleepScore/Sleep Profile.csv',
                'score': 'data/fitness/SleepScore/sleep_score.csv'
            },
            'temperature': {
                'device': 'data/fitness/Temperature/Device Temperature - *.csv',
                'computed': 'data//fitness/Temperature/Computed Temperature - *.csv'
            },
            'spo2': {
                'pattern': 'data/fitness/SPO2/Minute SpO2 - *.csv',
                'daily_pattern': 'data/fitness/SPO2/Daily SpO2 - *.csv'
            },
            'stress_score': {
                'path': 'data/fitness/StressScore/Stress Score.csv'
            },
            'hrv': {
                'summary': 'data/fitness/HeartRateVariability/HRVSummary/Daily Heart Rate Variability Summary - *.csv',
                'details': 'data/fitness/HeartRateVariability/HRVDetails/Heart Rate Variability Details - *.csv'
            },
            'respiratory_rate': {
                'pattern': 'data/fitness/HeartRateVariability/RespiratoryRateSummary/*.csv'
            }
        }
        
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
    
    def process_blood_glucose(self):
        """Process blood glucose data from multiple files."""
        print("\nProcessing blood glucose data...")
        pattern = self.DATA_PATHS['blood_glucose']['pattern']
        print(f"Looking for files matching pattern: {pattern}")
        
        files = glob.glob(pattern)
        if not files:
            raise Exception(f"No blood glucose files found matching pattern: {pattern}")
        
        print(f"Found {len(files)} files:")
        for file in files:
            print(f"- {file}")
        
        # First pass: determine date ranges for each file
        file_ranges = {}
        for file in files:
            df = pd.read_csv(file, usecols=['date'])
            # Use format='mixed' to handle various datetime formats including milliseconds and timezone offsets
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            file_ranges[file] = {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'n_rows': len(df)
            }
            print(f"\n{file}:")
            print(f"Date range: {file_ranges[file]['start']} to {file_ranges[file]['end']}")
            print(f"Number of rows: {file_ranges[file]['n_rows']}")
        
        # Sort files by start date
        sorted_files = sorted(files, key=lambda x: file_ranges[x]['start'])
        
        # Initialize cutoff_dates dictionary
        cutoff_dates = {}
        
        # Check for overlaps and verify data consistency
        for i in range(len(sorted_files)-1):
            current_file = sorted_files[i]
            next_file = sorted_files[i+1]
            if file_ranges[current_file]['end'] >= file_ranges[next_file]['start']:
                overlap_start = file_ranges[next_file]['start']
                overlap_end = file_ranges[current_file]['end']
                print(f"\nWARNING: Found overlap between files:")
                print(f"- {current_file}: {file_ranges[current_file]['start']} to {file_ranges[current_file]['end']}")
                print(f"- {next_file}: {file_ranges[next_file]['start']} to {file_ranges[next_file]['end']}")
                print(f"Overlap period: {overlap_start} to {overlap_end}")
                
                # Read overlapping portions from both files
                df1 = pd.read_csv(current_file, low_memory=False)
                df2 = pd.read_csv(next_file, low_memory=False)
                df1['date'] = pd.to_datetime(df1['date'], format='mixed')
                df2['date'] = pd.to_datetime(df2['date'], format='mixed')
                
                # Extract overlapping data
                overlap1 = df1[df1['date'] >= overlap_start]
                overlap2 = df2[df2['date'] <= overlap_end]
                
                # Sort both by date for comparison
                overlap1 = overlap1.sort_values('date')
                overlap2 = overlap2.sort_values('date')
                
                # Get all unique dates from both dataframes
                all_dates = pd.concat([overlap1['date'], overlap2['date']]).unique()
                all_dates = np.sort(all_dates)  # Use numpy's sort instead of calling sort directly
                
                # Compare data for each date
                inconsistencies = []
                for date in all_dates:
                    rows1 = overlap1[overlap1['date'] == date]
                    rows2 = overlap2[overlap2['date'] == date]
                    
                    # If one file has data for a date and the other doesn't, that's an inconsistency
                    if len(rows1) == 0 or len(rows2) == 0:
                        if len(rows1) > 0:
                            inconsistencies.append({
                                'date': date,
                                'issue': f"Data only in {current_file}",
                                'rows': rows1
                            })
                        else:
                            inconsistencies.append({
                                'date': date,
                                'issue': f"Data only in {next_file}",
                                'rows': rows2
                            })
                        continue
                    
                    # Compare all relevant columns
                    cols_to_compare = ['bgl', 'trend', 'msg_type', 'text', 'dose_units', 'food_g']
                    for col in cols_to_compare:
                        if col not in rows1.columns or col not in rows2.columns:
                            continue
                            
                        vals1 = set(rows1[col].dropna())
                        vals2 = set(rows2[col].dropna())
                        
                        if vals1 != vals2:
                            inconsistencies.append({
                                'date': date,
                                'issue': f"Different {col} values",
                                'file1_values': vals1,
                                'file2_values': vals2,
                                'rows1': rows1[[col, 'date', 'msg_type', 'text']],
                                'rows2': rows2[[col, 'date', 'msg_type', 'text']]
                            })
                
                if inconsistencies:
                    print("\nWARNING: Found inconsistencies in overlapping data:")
                    for inc in inconsistencies:
                        print(f"\nDate: {inc['date']}")
                        print(f"Issue: {inc['issue']}")
                        if 'file1_values' in inc:
                            print(f"Values in {current_file}: {inc['file1_values']}")
                            print(f"Values in {next_file}: {inc['file2_values']}")
                            print("\nRows from first file:")
                            print(inc['rows1'])
                            print("\nRows from second file:")
                            print(inc['rows2'])
                        else:
                            print("Rows:")
                            print(inc['rows'])
                    
                    raise ValueError("Found inconsistencies in overlapping data between files. Please verify data integrity.")
                else:
                    print("\nVerified: Overlapping data is consistent between files.")
                    # Now we can safely use the cutoff
                    cutoff_dates[current_file] = file_ranges[next_file]['start']
                    print(f"Using data from {current_file} up to {cutoff_dates[current_file]}")
        
        all_bgl = []
        
        # Process each file with cutoffs
        for file in sorted_files:
            print(f"\nReading {file}")
            df = pd.read_csv(file, low_memory=False)
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            
            # Apply cutoff if exists
            if file in cutoff_dates:
                n_before = len(df)
                df = df[df['date'] < cutoff_dates[file]]
                n_filtered = n_before - len(df)
                if n_filtered > 0:
                    print(f"Filtered out {n_filtered} rows after cutoff date {cutoff_dates[file]}")
            
            # Store original values for verification
            original_df = df.copy()
            original_df['original_index'] = range(len(df))
            
            # Standardize timestamp
            df = self._standardize_timestamp(df)
            
            # Remove exact duplicates
            n_before = len(df)
            df = df.drop_duplicates(keep='first')
            n_removed = n_before - len(df)
            if n_removed > 0:
                print(f"Removed {n_removed} exact duplicate rows (all columns matched)")
            
            df = df.sort_values('date')
            df['original_index'] = original_df['original_index']
            
            # Verify no BGL values were modified
            modified_mask = df['bgl'] != original_df.iloc[df['original_index']]['bgl']
            modified_mask = modified_mask & ~pd.isna(df['bgl']) & ~pd.isna(original_df.iloc[df['original_index']]['bgl'])
            if modified_mask.any():
                modified_indices = modified_mask[modified_mask].index
                comparison = pd.DataFrame({
                    'original': original_df.iloc[df.iloc[modified_indices]['original_index']]['bgl'].values,
                    'modified': df.iloc[modified_indices]['bgl'].values,
                    'timestamp': df.iloc[modified_indices]['date']
                })
                error_msg = f"\nBlood glucose values were modified unexpectedly:\n{comparison.head()}\nTotal modified values: {len(modified_indices)}"
                raise Exception(error_msg)
            
            df = df.drop('original_index', axis=1)
            all_bgl.append(df)
        
        # Merge all blood glucose data
        merged_bgl = pd.concat(all_bgl, ignore_index=True)
        merged_bgl = merged_bgl.sort_values('date')
        
        # First remove any remaining exact duplicates after merging files
        n_before = len(merged_bgl)
        merged_bgl = merged_bgl.drop_duplicates(keep='first')
        n_removed = n_before - len(merged_bgl)
        if n_removed > 0:
            print(f"\nRemoved {n_removed} exact duplicate rows after merging all files")
        
        # Now handle rows with same timestamp but different information
        timestamp_dupes = merged_bgl[merged_bgl.duplicated('date', keep=False)]
        if len(timestamp_dupes) > 0:
            n_timestamps = len(timestamp_dupes['date'].unique())
            print(f"\nFound {len(timestamp_dupes)} rows with duplicate timestamps ({n_timestamps} unique timestamps)")
            
            # Check if rows are identical except for msg_type and text
            def check_row_differences(group):
                # Get all columns except msg_type, text, sender_id, __typename, and bgl_date_millis
                cols_to_check = [col for col in group.columns if col not in [
                    'msg_type', 'text', 'sender_id', '__typename', 'bgl_date_millis', 'template', 'Unnamed: 0'
                ]]
                
                # Track if we need to merge any boolean columns
                bool_cols_to_merge = []
                bgl_to_average = None
                trend_to_use = None
                dose_to_use = None
                food_to_use = None
                
                for col in cols_to_check:
                    unique_vals = group[col].dropna().unique()
                    if len(unique_vals) > 1:
                        # For boolean columns, check if we can merge using OR
                        if group[col].dtype == bool or set(unique_vals).issubset({True, False}):
                            bool_cols_to_merge.append(col)
                            continue
                            
                        # For bgl_date_millis, allow different values since they represent exact millisecond timestamps
                        if col == 'bgl_date_millis':
                            continue
                            
                        # For template, allow different templates for third-party integrations
                        if col == 'template':
                            # Keep all unique templates since they represent different third-party integrations
                            # This is expected and not a conflict
                            continue
                            
                        # For BGL, check if values are within 5% of each other
                        if col == 'bgl':
                            # First check if we have a sensor reading (no msg_type)
                            sensor_readings = group[group['msg_type'].isna()]
                            if len(sensor_readings) == 1:
                                # Use the sensor reading
                                bgl_to_average = sensor_readings.iloc[0]['bgl']
                                continue
                            elif len(sensor_readings) > 1:
                                # Multiple sensor readings - prioritize readings with actual trend values
                                readings_with_trend = sensor_readings[sensor_readings['trend'].notna() & (sensor_readings['trend'] != 'NONE')]
                                if len(readings_with_trend) == 1:
                                    # Use the reading with a trend value
                                    bgl_to_average = readings_with_trend.iloc[0]['bgl']
                                    continue
                                elif len(readings_with_trend) > 1:
                                    # Multiple readings with trends - check if they're within 5%
                                    trend_bgls = readings_with_trend['bgl']
                                    mean_val = np.mean(trend_bgls)
                                    max_diff_pct = np.max(np.abs(trend_bgls - mean_val) / mean_val * 100)
                                    if max_diff_pct <= 5:
                                        bgl_to_average = mean_val
                                        continue
                                    else:
                                        # Find columns that have different values
                                        diff_cols = []
                                        for col in readings_with_trend.columns:
                                            if len(readings_with_trend[col].dropna().unique()) > 1:
                                                diff_cols.append(col)
                                        
                                        error_msg = f"\nConflict found at timestamp {group['date'].iloc[0]}:\n"
                                        error_msg += f"Multiple sensor readings with trends and >5% difference in BGL: {trend_bgls.values}\n"
                                        error_msg += "\nColumns with different values:\n"
                                        error_msg += str(readings_with_trend[diff_cols])
                                        raise ValueError(error_msg)
                                else:
                                    # No readings with trends - check if all readings are within 5%
                                    mean_val = np.mean(sensor_readings['bgl'])
                                    max_diff_pct = np.max(np.abs(sensor_readings['bgl'] - mean_val) / mean_val * 100)
                                    if max_diff_pct <= 5:
                                        bgl_to_average = mean_val
                                        continue
                                    else:
                                        # Find columns that have different values
                                        diff_cols = []
                                        for col in sensor_readings.columns:
                                            if len(sensor_readings[col].dropna().unique()) > 1:
                                                diff_cols.append(col)
                                        
                                        error_msg = f"\nConflict found at timestamp {group['date'].iloc[0]}:\n"
                                        error_msg += f"Multiple sensor readings with >5% difference: {sensor_readings['bgl'].values}\n"
                                        error_msg += "\nColumns with different values:\n"
                                        error_msg += str(sensor_readings[diff_cols])
                                        raise ValueError(error_msg)
                            else:
                                # No sensor readings - check if non-sensor readings are within 5%
                                bgl_vals = group['bgl'].dropna().values
                                if len(bgl_vals) > 0:
                                    mean_val = np.mean(bgl_vals)
                                    max_diff_pct = np.max(np.abs(bgl_vals - mean_val) / mean_val * 100)
                                    if max_diff_pct <= 5:
                                        bgl_to_average = mean_val
                                        continue
                                    else:
                                        print(f"\nWarning: Found conflicting non-sensor BGL values at {group['date'].iloc[0]}:")
                                        print(group[['date', 'bgl', 'msg_type', 'text', 'template']])
                                        # Use the first value for now
                                        bgl_to_average = bgl_vals[0]
                                        continue
                                    
                        # For trend, prefer actual trend values over NONE
                        if col == 'trend':
                            trends = [v for v in unique_vals if pd.notna(v) and v != 'NONE']
                            if len(trends) == 1:  # If there's exactly one non-NONE trend
                                trend_to_use = trends[0]
                                continue
                            elif len(trends) > 1:  # If there are multiple actual trends, that's a real conflict
                                error_msg = f"\nConflict found at timestamp {group['date'].iloc[0]}:\n"
                                error_msg += f"Multiple actual trend values found: {trends}\n"
                                error_msg += "\nOriginal rows:\n"
                                error_msg += str(group[['date', col, 'msg_type', 'text']])
                                raise ValueError(error_msg)
                            else:  # All trends are NONE or NaN
                                trend_to_use = 'NONE'
                                continue
                                
                        # For dose_units, check the msg_type and text to differentiate between different insulins
                        if col == 'dose_units':
                            # Group doses by msg_type and insulin type
                            dose_by_type = {}
                            for idx, row in group.iterrows():
                                if pd.notna(row['dose_units']) and row['dose_units'] > 0:
                                    msg_type = row['msg_type'] if pd.notna(row['msg_type']) else 'UNKNOWN'
                                    # Extract insulin type from text if possible
                                    insulin_type = None
                                    if pd.notna(row['text']):
                                        if 'Toujeo' in str(row['text']):
                                            insulin_type = 'Toujeo'
                                        elif 'Humalog' in str(row['text']):
                                            insulin_type = 'Humalog'
                                        # Add more insulin types as needed
                                    
                                    key = f"{msg_type}_{insulin_type}" if insulin_type else msg_type
                                    if key not in dose_by_type:
                                        dose_by_type[key] = []
                                    dose_by_type[key].append((row['dose_units'], row['text']))
                            
                            # Sum doses for each type and warn about multiple doses
                            for key, doses in dose_by_type.items():
                                if len(doses) > 1:
                                    total_dose = sum(dose for dose, _ in doses)
                                    print(f"\nWarning: Found multiple doses for {key} at {group['date'].iloc[0]}:")
                                    print(f"Doses: {[dose for dose, _ in doses]} - Summing to {total_dose}u")
                                    print("Original rows:")
                                    print(group[['date', 'dose_units', 'msg_type', 'text']])
                                    # Store the summed dose
                                    dose_by_type[key] = [(total_dose, doses[0][1])]
                            
                            # If we get here, we've handled all doses
                            continue
                                
                        # For food_g, check if it's the same food type before summing
                        if col == 'food_g':
                            # Group food by type
                            food_by_type = {}
                            for idx, row in group.iterrows():
                                if pd.notna(row['food_g']) and row['food_g'] > 0 and pd.notna(row['text']):
                                    # Remove amount from text to get food type
                                    text = str(row['text'])
                                    # Remove numbers and 'g' to get food type
                                    food_type = ' '.join(word for word in text.split() if not any(c.isdigit() for c in word) and word.lower() != 'g')
                                    food_type = food_type.strip()
                                    
                                    if food_type not in food_by_type:
                                        food_by_type[food_type] = []
                                    food_by_type[food_type].append((row['food_g'], text))
                            
                            # Sum amounts for each food type
                            for food_type, amounts in food_by_type.items():
                                if len(amounts) > 1:
                                    total_amount = sum(amount for amount, _ in amounts)
                                    print(f"\nWarning: Found multiple amounts for '{food_type}' at {group['date'].iloc[0]}:")
                                    print(f"Amounts: {[amount for amount, _ in amounts]} - Summing to {total_amount}g")
                                    print("Original rows:")
                                    print(group[['date', 'food_g', 'msg_type', 'text']])
                                    # Store the summed amount
                                    food_by_type[food_type] = [(total_amount, amounts[0][1])]
                                    continue
                            
                            # If we get here, we've handled all food amounts
                            continue
                        
                        error_msg = f"\nConflict found at timestamp {group['date'].iloc[0]}:\n"
                        error_msg += f"Column '{col}' has multiple values: {unique_vals}\n"
                        error_msg += "\nOriginal rows:\n"
                        error_msg += str(group[['date', col, 'msg_type', 'text']])
                        raise ValueError(error_msg)
                
                # If we have boolean columns to merge, BGL to average, trend to set, or dose to set
                if bool_cols_to_merge or bgl_to_average is not None or trend_to_use is not None or dose_to_use is not None or food_to_use is not None:
                    # Keep one row per unique msg_type
                    unique_msg_types = group['msg_type'].dropna().unique()
                    if len(unique_msg_types) == 0:
                        unique_msg_types = [None]
                    
                    merged_rows = []
                    for msg_type in unique_msg_types:
                        # Start with the first row that matches this msg_type, or first row if msg_type is None
                        if msg_type is None:
                            base_row = group.iloc[0]
                        else:
                            base_row = group[group['msg_type'] == msg_type].iloc[0]
                        
                        merged_row = base_row.copy()
                        
                        # Apply boolean merges
                        for col in bool_cols_to_merge:
                            merged_row[col] = group[col].any()
                            
                        # Apply BGL average if needed
                        if bgl_to_average is not None:
                            merged_row['bgl'] = bgl_to_average
                            
                        # Apply trend if needed
                        if trend_to_use is not None:
                            merged_row['trend'] = trend_to_use
                            
                        # Apply dose if needed
                        if dose_to_use is not None:
                            merged_row['dose_units'] = dose_to_use
                            
                        # Apply food amount if needed
                        if food_to_use is not None:
                            merged_row['food_g'] = food_to_use
                            
                        # Set __typename based on msg_type
                        merged_row['__typename'] = 'Message' if msg_type is not None else 'Reading'
                        
                        merged_rows.append(merged_row)
                    
                    return pd.DataFrame(merged_rows)
                
                # If we get here, all other columns are identical
                return group
            
            # Apply the check to each group of rows with same timestamp
            print("\nChecking for conflicts in duplicate timestamps...")
            try:
                # First verify no conflicts exist and merge boolean columns if needed
                merged_bgl = pd.concat([
                    merged_bgl[~merged_bgl.duplicated('date', keep=False)],  # Rows without duplicates
                    merged_bgl[merged_bgl.duplicated('date', keep=False)].groupby('date').apply(check_row_differences)
                ]).reset_index(drop=True)
                
                # Sort the final dataset
                merged_bgl = merged_bgl.sort_values(['date', 'msg_type'])
                print("All duplicate timestamps either only differ in msg_type/text or were merged using boolean OR")
            except ValueError as e:
                print("\nError: Found conflicting values in duplicate rows!")
                print(str(e))
                raise
            
            print(f"Final dataset: {len(merged_bgl)} rows")
        
        self.data_frames['blood_glucose'] = merged_bgl
        return merged_bgl
    
    def process_daily_readiness(self):
        """Process daily readiness scores."""
        print("\nProcessing daily readiness data...")
        pattern = self.DATA_PATHS['daily_readiness']['pattern']
        all_files = glob.glob(pattern)
        
        if not all_files:
            raise Exception("No daily readiness files found")
            
        dfs = []
        for file in all_files:
            print(f"Reading {file}")
            df = pd.read_csv(file)
            # Standardize timestamp first
            df = self._standardize_timestamp(df, 'date')
            # Rename date column to timestamp to match other dataframes
            df = df.rename(columns={'date': 'timestamp'})
            # Then rename other columns to match expected names in visualization
            df = df.rename(columns={
                'readiness_score_value': 'daily_readiness_readiness_score_value',
                'readiness_state': 'daily_readiness_readiness_state',
                'activity_subcomponent': 'daily_readiness_activity_subcomponent',
                'sleep_subcomponent': 'daily_readiness_sleep_subcomponent',
                'hrv_subcomponent': 'daily_readiness_hrv_subcomponent'
            })
            dfs.append(df)
        
        merged_df = pd.concat(dfs, ignore_index=True)
        # Sort by timestamp
        merged_df = merged_df.sort_values('timestamp')
        print(f"\nDaily readiness data shape: {merged_df.shape}")
        print("\nSample of daily readiness data:")
        print(merged_df.head())
        
        self.data_frames['daily_readiness'] = merged_df
        return self.data_frames['daily_readiness']
    
    def process_sleep(self):
        """Process sleep data."""
        print("\nProcessing sleep data...")
        profile_path = self.DATA_PATHS['sleep']['profile']
        score_path = self.DATA_PATHS['sleep']['score']
        
        # Process sleep score (this is the main data we care about)
        if not os.path.exists(score_path):
            raise Exception(f"Sleep score file not found: {score_path}")
            
        print(f"Reading {score_path}")
        score_df = pd.read_csv(score_path, usecols=['timestamp', 'overall_score', 'composition_score', 'revitalization_score', 
                                                    'duration_score', 'deep_sleep_in_minutes', 'resting_heart_rate', 'restlessness'])
        score_df = self._standardize_timestamp(score_df, 'timestamp')
        self.data_frames['sleep_score'] = score_df
        
        # Process sleep profile (only keep essential columns)
        if not os.path.exists(profile_path):
            raise Exception(f"Sleep profile file not found: {profile_path}")
            
        print(f"Reading {profile_path}")
        profile_df = pd.read_csv(profile_path, usecols=['creation_date', 'sleep_type', 'deep_sleep', 'rem_sleep', 'sleep_duration', 'sleep_start_time', 
                                                        'schedule_variability', 'restorative_sleep', 'time_before_sound_sleep',	'sleep_stability', 
                                                        'nights_with_long_awakenings', 'days_with_naps'])
        profile_df = self._standardize_timestamp(profile_df, 'creation_date')
        self.data_frames['sleep_profile'] = profile_df
        
        return self.data_frames['sleep_profile'], self.data_frames['sleep_score']
    
    def process_temperature(self):
        """Process temperature data from device and computed temperature files."""
        print("\nProcessing temperature data...")
        
        # Process device temperature files
        device_pattern = self.DATA_PATHS['temperature']['device']
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
        computed_pattern = self.DATA_PATHS['temperature']['computed']
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
        """Create timeline covering all data, ensuring no data points are lost."""
        print("\nCreating base timeline...")
        
        # Start with blood glucose data range as it's our primary focus
        if 'blood_glucose' in self.data_frames and self.data_frames['blood_glucose'] is not None:
            bgl_df = self.data_frames['blood_glucose']
            
            # Get all unique timestamps from blood glucose data
            bgl_timestamps = pd.to_datetime(bgl_df['date'].unique())
            
            # Find the overall min and max timestamps
            start_date = bgl_timestamps.min()
            end_date = bgl_timestamps.max()
            
            print(f"Blood glucose data range:")
            print(f"- Start: {start_date}")
            print(f"- End: {end_date}")
            print(f"- Unique timestamps: {len(bgl_timestamps):,}")
            
            # Create 5-minute interval timeline
            timeline = pd.date_range(
                start=start_date.floor('5min'),  # Round down to nearest 5 minutes
                end=end_date.ceil('5min'),    # Round up to nearest 5 minutes
                freq='5min'
            )
            
            # Create base DataFrame with both the 5-minute intervals and any original timestamps
            all_timestamps = np.union1d(timeline, bgl_timestamps)
            base_df = pd.DataFrame({'timestamp': all_timestamps})
            base_df = base_df.sort_values('timestamp')
            
            # Print statistics about the timeline
            print("\nTimeline statistics:")
            print(f"- Total timestamps: {len(base_df):,}")
            print(f"- 5-minute intervals: {len(timeline):,}")
            print(f"- Additional timestamps: {len(base_df) - len(timeline):,}")
            
            return base_df
        else:
            raise ValueError("Blood glucose data is required to create the base timeline")
    
    def merge_all_data(self):
        """Merge all processed datasets."""
        print("\nMerging all datasets...")
        
        # First determine blood glucose time range
        if 'blood_glucose' not in self.data_frames:
            raise ValueError("Blood glucose data is required for merging")
            
        bgl_df = self.data_frames['blood_glucose']
        bgl_start = bgl_df['date'].min()
        bgl_end = bgl_df['date'].max()
        print(f"\nBlood glucose data range: {bgl_start} to {bgl_end}")
        print("Will only keep other health metrics within this time range.")
        
        # Create base timeline
        final_df = self.create_base_timeline()
        print(f"Base timeline created: {len(final_df):,} rows")
        
        # Print data statistics before merging
        print("\nData statistics before merging:")
        for name, df in self.data_frames.items():
            if df is not None and not df.empty:
                print(f"\n{name}:")
                print(f"- Rows: {len(df):,}")
                time_cols = [col for col in df.columns if any(
                    term in col.lower() for term in ['time', 'date', 'timestamp']
                )]
                if time_cols:
                    time_col = time_cols[0]
                    print(f"- Time range: {df[time_col].min()} to {df[time_col].max()}")
                    print(f"- Unique timestamps: {df[time_col].nunique():,}")
        
        # Merge blood glucose data first (our primary data)
        final_df = pd.merge(
            final_df,
            bgl_df,
            left_on='timestamp',
            right_on='date',
            how='outer'  # Keep all timestamps
        )
        print(f"\nAfter merging blood glucose: {len(final_df):,} rows")
        print(f"Unique timestamps: {final_df['timestamp'].nunique():,}")
        print(f"Non-null BGL values: {final_df['bgl'].count():,}")
        
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
                print(f"Merging on {primary_time_col}")
                
                # Filter data to blood glucose time range
                n_before = len(merge_df)
                merge_df = merge_df[
                    (merge_df[primary_time_col] >= bgl_start) & 
                    (merge_df[primary_time_col] <= bgl_end)
                ]
                n_filtered = n_before - len(merge_df)
                if n_filtered > 0:
                    print(f"Filtered out {n_filtered:,} rows outside blood glucose time range")
                if len(merge_df) == 0:
                    print(f"No {name} data within blood glucose time range")
                    continue
                
                # Print data statistics before merge
                print(f"Before merge:")
                print(f"- Rows in incoming data: {len(merge_df):,}")
                print(f"- Unique timestamps in incoming data: {merge_df[primary_time_col].nunique():,}")
                
                # Add prefix to all non-timestamp columns to identify source
                cols_to_rename = [col for col in merge_df.columns if col not in time_cols]
                merge_df = merge_df.rename(columns={col: f"{name}_{col}" for col in cols_to_rename})
                
                # Merge on timestamp
                n_rows_before = len(final_df)
                n_unique_timestamps_before = final_df['timestamp'].nunique()
                
                final_df = pd.merge(
                    final_df,
                    merge_df,
                    left_on='timestamp',
                    right_on=primary_time_col,
                    how='outer'  # Keep all timestamps
                )
                
                # Print merge statistics
                print("\nAfter merge:")
                print(f"- Total rows: {len(final_df):,} (changed by {len(final_df) - n_rows_before:+,})")
                print(f"- Unique timestamps: {final_df['timestamp'].nunique():,} (changed by {final_df['timestamp'].nunique() - n_unique_timestamps_before:+,})")
                
                # Check for any data loss
                added_cols = [col for col in final_df.columns if col.startswith(name)]
                for col in added_cols:
                    orig_count = merge_df[col.replace(f"{name}_", "")].count() if col.replace(f"{name}_", "") in merge_df.columns else 0
                    final_count = final_df[col].count()
                    if final_count != orig_count:
                        print(f"Warning: Possible data loss in column {col}")
                        print(f"- Original non-null values: {orig_count:,}")
                        print(f"- Final non-null values: {final_count:,}")
                
                print(f"Added columns: {[col for col in final_df.columns if col.startswith(name)]}")
                
                # Clear memory
                del merge_df
        
        # Sort by timestamp
        final_df = final_df.sort_values('timestamp')
        
        # Remove any duplicate timestamp columns
        timestamp_cols = [col for col in final_df.columns if any(
            term in col.lower() for term in ['time', 'date', 'timestamp']
        )]
        for col in timestamp_cols:
            if col != 'timestamp':
                final_df = final_df.drop(columns=col)
        
        print(f"\nFinal dataset shape: {final_df.shape}")
        print("\nColumns in final dataset:")
        for col in sorted(final_df.columns):
            non_null = final_df[col].count()
            pct_present = (non_null / len(final_df)) * 100
            print(f"- {col}: {non_null:,} non-null values ({pct_present:.1f}%)")
        
        return final_df
    
    def process_spo2(self):
        """Process SPO2 data."""
        print("\nProcessing SPO2 data...")
        pattern = self.DATA_PATHS['spo2']['pattern']
        daily_pattern = self.DATA_PATHS['spo2']['daily_pattern']
        
        # Process minute-level SPO2 data
        all_files = glob.glob(pattern)
        if not all_files:
            raise Exception("No SPO2 files found")
        dfs = []
        for file in all_files:
            print(f"Reading {file}")
            df = pd.read_csv(file, usecols=['timestamp', 'value'])
            df = self._standardize_timestamp(df)
            dfs.append(df)
            if len(dfs) >= 10:
                dfs = [pd.concat(dfs, ignore_index=True)]
        minute_spo2_df = pd.concat(dfs, ignore_index=True)

        # Process daily-level SPO2 data
        daily_files = glob.glob(daily_pattern)
        if not daily_files:
            raise Exception("No daily SPO2 files found")
        daily_dfs = []
        for file in daily_files:
            print(f"Reading {file}")
            df = pd.read_csv(file, usecols=['timestamp', 'average_value', 'lower_bound', 'upper_bound'])
            df = self._standardize_timestamp(df)
            df = df.rename(columns={
                'average_value': 'spo2_daily_average_value',
                'lower_bound': 'spo2_daily_lower_bound',
                'upper_bound': 'spo2_daily_upper_bound'
            })
            daily_dfs.append(df)
            if len(daily_dfs) >= 10:
                daily_dfs = [pd.concat(daily_dfs, ignore_index=True)]
        daily_spo2_df = pd.concat(daily_dfs, ignore_index=True)

        # Merge minute and daily SPO2 data
        self.data_frames['spo2'] = pd.merge(minute_spo2_df, daily_spo2_df, on='timestamp', how='outer')
        return self.data_frames['spo2']
    
    def process_stress_score(self):
        """Process stress score data."""
        print("\nProcessing stress score data...")
        
        path = self.DATA_PATHS['stress_score']['path']
        if not os.path.exists(path):
            raise Exception(f"Stress score file not found: {path}")
            
        print(f"Reading {path}")
        # Only read essential columns
        essential_cols = ['DATE', 'STRESS_SCORE', 'SLEEP_POINTS', 'RESPONSIVENESS_POINTS', 'EXERTION_POINTS', 'STATUS', 'CALCULATION_FAILED']
        df = pd.read_csv(path, usecols=essential_cols)
        
        # Remove rows where STATUS is 'NO_DATA' or CALCULATION_FAILED is True
        df = df[(df['STATUS'] != 'NO_DATA') & (df['CALCULATION_FAILED'] != True)]

        # Filter out rows where stress score is 0 (represents missing/invalid data)
        n_rows_before = len(df)
        df = df[df['STRESS_SCORE'] != 0]
        n_filtered = n_rows_before - len(df)
        if n_filtered > 0:
            print(f"Filtered out {n_filtered} rows where stress score was 0 (missing/invalid data)")
        
        df = self._standardize_timestamp(df, 'DATE')
        
        self.data_frames['stress_score'] = df
        return self.data_frames['stress_score']
    
    def process_hrv(self):
        """Process heart rate variability data."""
        print("\nProcessing HRV data...")
        
        # Process HRV summary data
        summary_pattern = self.DATA_PATHS['hrv']['summary']
        summary_files = glob.glob(summary_pattern)
        if not summary_files:
            raise Exception("No HRV summary files found")
        
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
        
        # Process HRV details data
        details_pattern = self.DATA_PATHS['hrv']['details']
        details_files = glob.glob(details_pattern)
        if not details_files:
            raise Exception("No HRV details files found")
        
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
        
        details_df = pd.concat(details_dfs, ignore_index=True)
        self.data_frames['hrv_details'] = details_df
        
        return self.data_frames.get('hrv_summary'), self.data_frames.get('hrv_details')
        
    def process_respiratory_rate(self):
        """Process respiratory rate data."""
        print("\nProcessing respiratory rate data...")
        pattern = self.DATA_PATHS['respiratory_rate']['pattern']
        all_files = glob.glob(pattern)
        
        if not all_files:
            raise Exception("No respiratory rate files found")
            
        dfs = []
        for file in all_files:
            print(f"Reading {file}")
            df = pd.read_csv(file)
            df = self._standardize_timestamp(df)
            dfs.append(df)
        
        self.data_frames['respiratory_rate'] = pd.concat(dfs, ignore_index=True)
        return self.data_frames['respiratory_rate']

def main():
    try:
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
        output_path = "output/merged_health_data.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"\nMerged dataset saved to {output_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 