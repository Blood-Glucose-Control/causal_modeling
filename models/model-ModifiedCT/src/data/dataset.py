import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class GlucoseDataset(Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 scaler: StandardScaler,
                 history_window: int = 144,
                 prediction_horizon: int = 36,
                 stride: int = 6):
        
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.scaler = scaler
        
        # Indices
        self.treatment_idx = 1
        self.outcome_idx = 0
        
        # Feature Engineering & Normalization
        # We assume the dataframe is already the correct slice (Train, Val, or Test)
        self.raw_df = dataframe
        self.features = self.extract_features(self.raw_df)
        self.normalized_features = self.scaler.transform(self.features)
        
        # Create sliding windows indices
        self.valid_indices = self._create_indices(len(self.raw_df))
        
    @staticmethod
    def extract_features(df):
        """Turn raw dataframe into matrix of features"""
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
        return np.stack([
            glucose, insulin, carbs, exercise, stress, 
            active_insulin, carb_impact, sin_hour, cos_hour
        ], axis=1).astype(np.float32)

    def transform_new_data(self, df):
        """Transform a new dataframe using the dataset's scaler."""
        features = self.extract_features(df)
        normalized = self.scaler.transform(features)
        return normalized

    def _create_indices(self, total_len):
        indices = []
        # Ensure we don't go out of bounds
        for i in range(0, total_len - self.history_window - self.prediction_horizon + 1, self.stride):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.get_window(self.normalized_features, self.valid_indices[idx])
        
    def get_window(self, data_matrix, start_idx):
        mid_idx = start_idx + self.history_window
        end_idx = mid_idx + self.prediction_horizon
        
        window_data = data_matrix[start_idx:end_idx]
        
        encoder_inputs = window_data[:self.history_window]
        future_window = window_data[self.history_window:]
        
        # Future Treatments: Insulin(1), Carbs(2), Exercise(3), Stress(4), Time(7,8)
        future_treatments_indices = [1, 2, 3, 4, 7, 8] 
        future_treatments = future_window[:, future_treatments_indices]
        
        future_outcomes = future_window[:, [self.outcome_idx]]
        
        return {
            'encoder_inputs': torch.from_numpy(encoder_inputs),
            'future_treatments': torch.from_numpy(future_treatments),
            'future_outcomes': torch.from_numpy(future_outcomes)
        }
