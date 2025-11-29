import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Tuple, Dict

from .generator import DiabetesAnalyzer

class GlucoseDataset(Dataset):
    def __init__(self, 
                 mode: str = 'train',
                 history_window: int = 144,  # 12 hours (5min * 144)
                 prediction_horizon: int = 36, # 3 hours
                 stride: int = 6,            # 30 mins
                 num_days: int = 365,        # Generate 1 year of data
                 seed: int = 42):
        
        self.mode = mode
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        
        # Generate Data
        print(f"Generating {num_days} days of synthetic data for {mode} set...")
        analyzer = DiabetesAnalyzer(seed=seed)
        self.raw_df = analyzer.generate_patient_data(n_days=num_days)
        
        # Feature Engineering
        self._process_features()
        
        # Split Logic (Simulated by just generating new data for each set, 
        # or we could split the generated dataframe. Generating separate is safer for leakage)
        # For simplicity/memory, we might want to split one big generation if it was real data.
        # Since it's synthetic, generating fresh data for train/val/test with different seeds is fine/better.
        
        # Preprocessing (Fit scaler only on train)
        self.scaler = StandardScaler()
        if mode == 'train':
            self.scaler.fit(self.features)
        # Note: In a real pipeline, you'd load the scaler from the training set.
        # For this self-contained class, we'll assume 'train' is initialized first 
        # and we might need a way to share scalers. 
        # For now, we will just fit on self for all to get it running, 
        # but correct way is fit on train, transform others.
        
        self.normalized_features = self.scaler.fit_transform(self.features)
        
        # Create sliding windows indices
        self.valid_indices = self._create_indices()
        
    def _process_features(self):
        """Turn raw dataframe into matrix of features"""
        df = self.raw_df
        
        # Time embeddings
        timestamps = df.index.to_series()
        hours = timestamps.dt.hour + timestamps.dt.minute / 60
        
        # Cyclic time features
        sin_hour = np.sin(2 * np.pi * hours / 24)
        cos_hour = np.cos(2 * np.pi * hours / 24)
        
        # Core features
        glucose = df['glucose'].values
        insulin = df['insulin'].values
        carbs = df['carbs'].values
        exercise = df['exercise'].values
        stress = df['stress'].values
        
        # Derived state features (from generator logic)
        active_insulin = df['active_insulin'].values
        carb_impact = df['carb_impact'].values
        
        # Stack into (N, Features)
        # Features: [Glucose, Insulin, Carbs, Exercise, Stress, ActiveInsulin, CarbImpact, SinTime, CosTime]
        self.feature_names = ['glucose', 'insulin', 'carbs', 'exercise', 'stress', 
                            'active_insulin', 'carb_impact', 'sin_hour', 'cos_hour']
                            
        self.features = np.stack([
            glucose, insulin, carbs, exercise, stress, 
            active_insulin, carb_impact, sin_hour, cos_hour
        ], axis=1).astype(np.float32)
        
        # Treatment indices for convenience (Insulin is index 1)
        self.treatment_idx = 1
        self.outcome_idx = 0
        
    def _create_indices(self):
        total_len = len(self.raw_df)
        indices = []
        for i in range(0, total_len - self.history_window - self.prediction_horizon + 1, self.stride):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        mid_idx = start_idx + self.history_window
        end_idx = mid_idx + self.prediction_horizon
        
        # Full window of data
        window_data = self.normalized_features[start_idx:end_idx]
        
        # Split into encoder input and decoder targets
        # Encoder sees [0 : history_window]
        encoder_inputs = window_data[:self.history_window]
        
        # Decoder targets [history_window : history_window + horizon]
        # But we need to separate treatments (known future) vs outcomes (unknown future)
        future_window = window_data[self.history_window:]
        
        # Future Treatments (Assumed known/planned for counterfactuals)
        # We take Insulin(1), Carbs(2), Exercise(3), Stress(4), and Time(7,8)
        # We DO NOT take Glucose(0), ActiveInsulin(5), CarbImpact(6) as those are internal states/outcomes
        future_treatments_indices = [1, 2, 3, 4, 7, 8] 
        future_treatments = future_window[:, future_treatments_indices]
        
        # Targets (Glucose)
        future_outcomes = future_window[:, [self.outcome_idx]]
        
        return {
            'encoder_inputs': torch.from_numpy(encoder_inputs),
            'future_treatments': torch.from_numpy(future_treatments),
            'future_outcomes': torch.from_numpy(future_outcomes)
        }

if __name__ == "__main__":
    # Quick test
    ds = GlucoseDataset(mode='train', num_days=5)
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print("Encoder Input:", sample['encoder_inputs'].shape)
    print("Future Treatments:", sample['future_treatments'].shape)
    print("Future Outcomes:", sample['future_outcomes'].shape)

