import pandas as pd
import warnings
from datetime import timedelta

def validate_health_data(csv_path='output/merged_health_data.csv'):
    """
    Validate and analyze the merged health dataset.
    
    Args:
        csv_path (str): Path to the merged health data CSV file
    """
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("\n=== Dataset Validation ===")
    
    # 1. Basic dataset properties
    print("\n1. Basic Information:")
    print(f"Total records: {len(df):,}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("\nColumn names:")
    for col in sorted(df.columns):
        print(f"- {col}")
    
    # 2. Time continuity check
    print("\n2. Time Continuity Check:")
    time_diff = df['timestamp'].diff()
    expected_diff = pd.Timedelta(minutes=1)
    gaps = time_diff != expected_diff
    if gaps.any():
        print(f"Found {gaps.sum():,} gaps in the timeline")
        print("\nLargest gaps:")
        gap_sizes = time_diff[gaps].sort_values(ascending=False)
        for i, gap in enumerate(gap_sizes.head()):
            gap_start = df['timestamp'][gap_sizes.index[i] - 1]
            gap_end = df['timestamp'][gap_sizes.index[i]]
            print(f"Gap of {gap}: from {gap_start} to {gap_end}")
    else:
        print("Timeline is continuous with 1-minute intervals")
    
    # 3. Data presence check for key metrics
    print("\n3. Data Presence by Time of Day:")
    df['hour_of_day'] = df['timestamp'].dt.hour
    metrics = {
        'Blood Glucose': 'bgl',
        'SpO2': 'value_minute',
        'Temperature': 'value_device'
    }
    
    for metric, column in metrics.items():
        if column in df.columns:
            presence = df.groupby('hour_of_day')[column].count()
            total_days = (df['timestamp'].max() - df['timestamp'].min()).days
            avg_readings = presence / total_days
            
            print(f"\n{metric} statistics:")
            print(f"Total readings: {presence.sum():,}")
            print(f"Average readings per day: {presence.sum()/total_days:.1f}")
            print("\nAverage readings per hour:")
            for hour, count in avg_readings.items():
                print(f"Hour {hour:02d}: {count:.2f}")
    
    # 4. Value range checks
    print("\n4. Value Range Checks:")
    range_checks = {
        'Blood Glucose (mg/dL)': ('value', 40, 400),
        'SpO2 (%)': ('value_minute', 80, 100),
        'Temperature (°C)': ('value_device', 35, 40)
    }
    
    for metric, (col, min_val, max_val) in range_checks.items():
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"\n{metric}:")
                print(f"Total readings: {len(values):,}")
                print(f"Min: {values.min():.1f}")
                print(f"Max: {values.max():.1f}")
                print(f"Mean: {values.mean():.1f}")
                print(f"Median: {values.median():.1f}")
                print(f"Std Dev: {values.std():.1f}")
                
                # Calculate distribution in 10 bins
                bins = pd.cut(values, bins=10)
                dist = bins.value_counts().sort_index()
                print("\nValue distribution:")
                for bin_range, count in dist.items():
                    print(f"{bin_range}: {count:,}")
                
                out_of_range = ((values < min_val) | (values > max_val)).sum()
                if out_of_range > 0:
                    print(f"\nWarning: {out_of_range:,} values outside expected range [{min_val}-{max_val}]")
    
    # 5. Cross-metric correlation check
    print("\n5. Key Correlations:")
    key_metrics = [col for col in ['value', 'value_minute', 'value_device'] if col in df.columns]
    if len(key_metrics) > 1:
        correlations = df[key_metrics].corr()
        print("\nCorrelation Matrix:")
        print(correlations)
    
    # 6. Missing data analysis
    print("\n6. Missing Data Analysis:")
    missing_pct = df.isnull().mean() * 100
    print("\nMissing value percentages:")
    for col, pct in missing_pct.sort_values(ascending=False).items():
        if pct > 0:
            print(f"{col}: {pct:.1f}%")

def main():
    validate_health_data()
    print("\nValidation complete.")

if __name__ == "__main__":
    main() 