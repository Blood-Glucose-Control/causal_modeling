import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.dataset import GlucoseDataset
from src.data.generator import DiabetesAnalyzer

def get_data_splits(num_days=365, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Generates a continuous dataset and splits it into Train/Val/Test.
    Fits scaler on Train only.
    """
    print(f"Generating {num_days} days of synthetic data...")
    analyzer = DiabetesAnalyzer(seed=seed)
    full_df = analyzer.generate_patient_data(n_days=num_days)
    
    # Calculate split indices based on time
    n = len(full_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = full_df.iloc[:train_end]
    val_df = full_df.iloc[train_end:val_end]
    test_df = full_df.iloc[val_end:]
    
    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Fit Scaler on Train
    print("Fitting Scaler on Training Data...")
    train_features = GlucoseDataset.extract_features(train_df)
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    # Create Dataset Objects
    train_ds = GlucoseDataset(train_df, scaler)
    val_ds = GlucoseDataset(val_df, scaler)
    test_ds = GlucoseDataset(test_df, scaler)
    
    return train_ds, val_ds, test_ds, scaler
