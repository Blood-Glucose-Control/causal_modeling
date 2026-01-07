import pandas as pd
import numpy as np
import argparse
import os

# Map raw column names to our canonical feature names
REQUIRED_MAPPING = {
    'bgl': 'glucose',
    'dose_units': 'insulin',
    'food_g': 'carbs',
    'daily_readiness_daily_readiness_activity_subcomponent': 'exercise',
    'stress_score_STRESS_SCORE': 'stress',
    'hrv_summary_rmssd': 'hrv',
    'sleep_score_overall_score': 'sleep',
    # Temperature Removed (Not found in merged CSV)
    # SPO2 is handled dynamically due to variable naming
}

def load_and_validate(path):
    print(f"Loading raw data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
        
    df = pd.read_csv(path, low_memory=False)
    
    # 1. Validate Timestamp
    time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    if time_col not in df.columns:
        raise ValueError(f"CRITICAL: No timestamp/date column found in {path}")
    
    df['timestamp'] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values('timestamp').set_index('timestamp')
    
    # Create clean dataframe
    clean_df = pd.DataFrame(index=df.index)

    # 2. Dynamic Column Resolution
    
    # SPO2
    if 'spo2_value' in df.columns:
        clean_df['spo2'] = pd.to_numeric(df['spo2_value'], errors='coerce')
    elif 'spo2_spo2_daily_average_value' in df.columns:
        clean_df['spo2'] = pd.to_numeric(df['spo2_spo2_daily_average_value'], errors='coerce')
    else:
        # Warn but don't crash if SPO2 is missing, just fill 0? No, strict mode.
        # But earlier user said "no placeholders".
        # If SPO2 is missing, we must crash or drop the feature from model.
        # The CSV header shows `spo2_value` exists.
        raise ValueError("CRITICAL: No SPO2 column found!")

    # 3. Static Column Mapping & Validation
    for source_col, target_name in REQUIRED_MAPPING.items():
        if source_col in df.columns:
            clean_df[target_name] = pd.to_numeric(df[source_col], errors='coerce')
        else:
            # Try alternate names for Activity/Stress if exact match fails
            if target_name == 'exercise' and 'daily_readiness_activity_subcomponent' in df.columns:
                clean_df['exercise'] = pd.to_numeric(df['daily_readiness_activity_subcomponent'], errors='coerce')
            elif target_name == 'stress' and 'stress_score' in df.columns: # Fallback check
                clean_df['stress'] = pd.to_numeric(df['stress_score'], errors='coerce')
            else:
                raise ValueError(f"CRITICAL: Missing required column '{source_col}' for feature '{target_name}'")

    # 4. Strict Data Cleaning
    print("Validating data integrity...")

    # A. Target (Glucose): Interpolate small gaps, drop large ones.
    # We accept interpolation up to 30 mins (approx 6 steps of 5 mins).
    # Note: limit is int (number of NaNs), not time.
    clean_df['glucose'] = clean_df['glucose'].interpolate(method='time', limit=6)
    
    # Drop rows where glucose is still missing (the gaps were too big)
    original_len = len(clean_df)
    clean_df = clean_df.dropna(subset=['glucose'])
    dropped_rows = original_len - len(clean_df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing glucose data (gaps > 30 mins).")

    # B. Treatments (Insulin/Carbs): NaNs mean "No Action" -> 0
    clean_df['insulin'] = clean_df['insulin'].fillna(0)
    clean_df['carbs'] = clean_df['carbs'].fillna(0)

    # C. Context (Exercise/Stress): NaNs mean "Baseline" -> 0
    clean_df['exercise'] = clean_df['exercise'].fillna(0)
    clean_df['stress'] = clean_df['stress'].fillna(0)

    # D. Vitals (HRV/Sleep/SPO2): Forward Fill (Persist last known state)
    # Physiology doesn't reset to 0 instantly.
    for col in ['hrv', 'sleep', 'spo2']:
        clean_df[col] = clean_df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Final check: If still NaN (e.g. file started with NaN and never got value), crash.
        if clean_df[col].isna().any():
            # If absolutely no data exists for a metric, we can't train.
            raise ValueError(f"CRITICAL: Column '{col}' contains NaNs that could not be filled. Entire history missing?")

    print("Data preparation complete.")
    print(f"Final Shape: {clean_df.shape}")
    print(f"Columns: {clean_df.columns.tolist()}")
    return clean_df

if __name__ == "__main__":
    # Assume running from model-CRN-real/
    INPUT_PATH = "../output/merged_health_data.csv"
    OUTPUT_PATH = "../output/processed_training_data.csv"
    
    try:
        df = load_and_validate(INPUT_PATH)
        df.to_csv(OUTPUT_PATH)
        print(f"Success! Clean data saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"\nFAILURE: {str(e)}")
        exit(1)
