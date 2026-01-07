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
        
        # Indices for Decoder
        self.treatment_idx = 1
        self.outcome_idx = 0
        
        # 1. Validate Input
        self.raw_df = dataframe.copy()
        
        if not isinstance(self.raw_df.index, pd.DatetimeIndex):
            if 'timestamp' in self.raw_df.columns:
                self.raw_df.set_index('timestamp', inplace=True)
                self.raw_df.index = pd.to_datetime(self.raw_df.index)
            else:
                raise ValueError("Dataset requires DatetimeIndex. Please run prep_real_data.py first.")
        
        # 2. Extract Features (Strict Mode - assumes prep_real_data.py was run)
        self.features = self.extract_features(self.raw_df)
        self.normalized_features = self.scaler.transform(self.features)
        
        # 3. Create sliding windows
        self.valid_indices = self._create_indices(len(self.raw_df))
        
    @staticmethod
    def calculate_decay(series, decay_factor=0.95):
        """Simple recursive decay to estimate 'On Board' values."""
        values = series.values
        decayed = np.zeros_like(values, dtype=np.float32)
        current_val = 0.0
        for i in range(len(values)):
            current_val = current_val * decay_factor + values[i]
            decayed[i] = current_val
        return decayed

    @staticmethod
    def extract_features(df):
        """
        Turn pre-processed dataframe into matrix of features.
        Expects canonical column names: ['glucose', 'insulin', 'carbs', 'exercise', 'stress', 'hrv', 'sleep', 'spo2']
        """
        try:
            glucose = df['glucose'].values
            insulin = df['insulin'].values
            carbs = df['carbs'].values
            exercise = df['exercise'].values
            stress = df['stress'].values
            hrv = df['hrv'].values
            sleep = df['sleep'].values
            spo2 = df['spo2'].values
            # Temperature removed as it was missing in source
        except KeyError as e:
            raise ValueError(f"Missing required column: {e}. Did you run prep_real_data.py?")

        # Derived State Features (IOB / COB) - Engineering
        active_insulin = GlucoseDataset.calculate_decay(pd.Series(insulin), decay_factor=0.95)
        carb_impact = GlucoseDataset.calculate_decay(pd.Series(carbs), decay_factor=0.92)

        # Time Features
        timestamps = df.index.to_series()
        hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
        sin_hour = np.sin(2 * np.pi * hours / 24.0).values
        cos_hour = np.cos(2 * np.pi * hours / 24.0).values
        
        # Stack into (N, Features)
        # 0: Glucose, 1: Insulin, 2: Carbs, 3: Exercise, 4: Stress, 
        # 5: ActIns, 6: CarbImp, 7: Sin, 8: Cos,
        # 9: HRV, 10: Sleep, 11: SPO2
        # TOTAL DIM = 12
        return np.stack([
            glucose, insulin, carbs, exercise, stress, 
            active_insulin, carb_impact, sin_hour, cos_hour,
            hrv, sleep, spo2
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
        
        # Future Inputs: Treatments + Time
        # 1: Insulin, 2: Carbs, 7: Sin, 8: Cos
        future_treatments_indices = [1, 2, 7, 8] 
        future_treatments = future_window[:, future_treatments_indices]
        
        future_outcomes = future_window[:, [self.outcome_idx]]
        
        return {
            'encoder_inputs': torch.from_numpy(encoder_inputs),
            'future_treatments': torch.from_numpy(future_treatments),
            'future_outcomes': torch.from_numpy(future_outcomes)
        }
