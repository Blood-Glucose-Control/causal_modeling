#!/usr/bin/env python

import sys
sys.path.append('.')

from utils.glucose_simulation import get_glucose_sim_data
from utils.glucose_evaluation_utils import get_processed_data

# Load and process data
pickle_map = get_glucose_sim_data('../Data/ml_dataset.csv', sequence_length=10, prediction_horizon=3)

print("Raw data shapes:")
for key, data in pickle_map.items():
    if isinstance(data, dict):
        print(f"{key}:")
        for subkey, subdata in data.items():
            if hasattr(subdata, 'shape'):
                print(f"  {subkey}: {subdata.shape}")
    else:
        print(f"{key}: {data}")

print("\nProcessed data shapes:")
training_processed = get_processed_data(pickle_map['training_data'], pickle_map['scaling_data'])
for key, data in training_processed.items():
    if hasattr(data, 'shape'):
        print(f"{key}: {data.shape}")
    else:
        print(f"{key}: {data}")