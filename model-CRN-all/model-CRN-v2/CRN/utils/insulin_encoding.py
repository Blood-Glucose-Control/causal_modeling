# Copyright (c) 2024, Diabetes CRN Adaptation
# Insulin dosage encoding for ordinal treatment levels in diabetes management

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1


class InsulinEncoder:
    """
    Insulin dosage encoding system for diabetes CRN modeling.
    
    Uses integer encoding to map ordinal dose levels to integer values [1,2,3,4,5].
    This approach respects the ordinal nature of insulin dosing decisions where
    higher numbers represent higher doses.
    """
    
    def __init__(self, num_dose_levels=5):
        """
        Initialize insulin encoder with integer encoding.
        
        Args:
            num_dose_levels: Number of discrete dose categories (default: 5)
        """
        self.num_dose_levels = num_dose_levels
        
        # Define dose thresholds for discretization
        # These can be adjusted based on clinical guidelines
        self.dose_thresholds = self._create_dose_thresholds()
    
    def _create_dose_thresholds(self):
        """Create thresholds for converting continuous doses to discrete levels."""
        # Conservative insulin dose ranges (units):
        # Level 1: 0.5-2.0 units (very low)
        # Level 2: 2.0-4.0 units (low) 
        # Level 3: 4.0-6.0 units (moderate)
        # Level 4: 6.0-8.0 units (high)
        # Level 5: 8.0+ units (very high)
        return [0.0, 2.0, 4.0, 6.0, 8.0, float('inf')]
    
    def continuous_to_discrete(self, insulin_doses):
        """
        Convert continuous insulin doses to discrete dose levels.
        
        Args:
            insulin_doses: Array of continuous insulin doses (units)
            
        Returns:
            Array of discrete dose levels [1, 2, 3, 4, 5]
        """
        insulin_doses = np.asarray(insulin_doses)
        discrete_levels = np.zeros_like(insulin_doses, dtype=int)
        
        for i in range(self.num_dose_levels):
            mask = (insulin_doses >= self.dose_thresholds[i]) & \
                   (insulin_doses < self.dose_thresholds[i + 1])
            discrete_levels[mask] = i + 1
        
        # Handle zero doses separately (no insulin = level 0, but we map to level 1 for simplicity)
        zero_mask = insulin_doses == 0
        discrete_levels[zero_mask] = 1
        
        return discrete_levels
    
    def discrete_to_continuous(self, dose_levels):
        """
        Convert discrete dose levels back to representative continuous doses.
        
        Args:
            dose_levels: Array of discrete dose levels [1, 2, 3, 4, 5]
            
        Returns:
            Array of representative continuous doses
        """
        dose_levels = np.asarray(dose_levels, dtype=int)
        continuous_doses = np.zeros_like(dose_levels, dtype=float)
        
        # Map each level to the midpoint of its range
        level_midpoints = [
            1.25,   # Level 1: midpoint of 0.5-2.0
            3.0,    # Level 2: midpoint of 2.0-4.0
            5.0,    # Level 3: midpoint of 4.0-6.0
            7.0,    # Level 4: midpoint of 6.0-8.0
            9.0     # Level 5: representative of 8.0+
        ]
        
        for i, midpoint in enumerate(level_midpoints):
            mask = dose_levels == (i + 1)
            continuous_doses[mask] = midpoint
        
        return continuous_doses
    
    def encode_for_model(self, dose_levels):
        """
        Encode discrete dose levels for CRN model input using integer encoding.
        
        Args:
            dose_levels: Array of discrete dose levels [1, 2, 3, 4, 5]
            
        Returns:
            Integer encoded representation as float32 array
        """
        dose_levels = np.asarray(dose_levels, dtype=int)
        # Simple integer encoding: [1, 2, 3, 4, 5]
        return dose_levels.astype(np.float32)
    
    def get_treatment_input_dim(self):
        """Get the input dimension for the treatment encoding (always 1 for integer encoding)."""
        return 1


class DiabetesDataProcessor:
    """
    Data processing utilities for converting diabetes-data-api output to CRN format.
    
    Handles the transformation of continuous glucose/insulin time series data 
    into the discrete sequence format expected by the CRN model.
    """
    
    def __init__(self, insulin_encoder, max_sequence_length=60):
        """
        Initialize data processor.
        
        Args:
            insulin_encoder: InsulinEncoder instance
            max_sequence_length: Maximum sequence length for CRN model
        """
        self.insulin_encoder = insulin_encoder
        self.max_sequence_length = max_sequence_length
    
    def process_patient_data(self, patient_df, outcome_window_hours=3):
        """
        Convert diabetes-data-api DataFrame to CRN training format.
        
        Args:
            patient_df: DataFrame from DiabetesAnalyzer.generate_patient_data()
            outcome_window_hours: Hours after each insulin dose to predict outcomes
            
        Returns:
            Dictionary with CRN-compatible training data
        """
        # Extract insulin intervention timepoints
        insulin_times = patient_df[patient_df['insulin'] > 0].index
        
        if len(insulin_times) == 0:
            raise ValueError("No insulin interventions found in patient data")
        
        sequences = []
        for insulin_time in insulin_times:
            # Create sequence window around insulin intervention
            seq_start = max(patient_df.index[0], insulin_time - pd.Timedelta(hours=4))  # 4 hours history
            seq_end = insulin_time + pd.Timedelta(hours=outcome_window_hours)
            
            # Extract sequence data
            seq_data = patient_df.loc[seq_start:seq_end].copy()
            
            if len(seq_data) < self.max_sequence_length:
                # Pad sequence if too short
                seq_data = self._pad_sequence(seq_data, self.max_sequence_length)
            elif len(seq_data) > self.max_sequence_length:
                # Truncate sequence if too long
                seq_data = seq_data.iloc[:self.max_sequence_length]
            
            sequences.append(seq_data)
        
        return self._format_for_crn(sequences)
    
    def _pad_sequence(self, seq_data, target_length):
        """Pad sequence to target length with appropriate values."""
        current_length = len(seq_data)
        if current_length >= target_length:
            return seq_data
        
        # Create padding with baseline values
        padding_length = target_length - current_length
        padding_index = pd.date_range(
            start=seq_data.index[0] - pd.Timedelta(minutes=5 * padding_length),
            periods=padding_length,
            freq='5min'
        )
        
        padding_df = pd.DataFrame(index=padding_index)
        padding_df['glucose'] = 100  # Baseline glucose
        padding_df['insulin'] = 0.0
        padding_df['carbs'] = 0
        padding_df['exercise'] = 0
        padding_df['stress'] = 0.0
        padding_df['active_insulin'] = 0.0
        padding_df['carb_impact'] = 0.0
        padding_df['meal_insulin_delay'] = 0
        padding_df['intervention_id'] = ''
        
        return pd.concat([padding_df, seq_data])
    
    def _format_for_crn(self, sequences):
        """Format sequences into CRN-compatible tensors."""
        num_sequences = len(sequences)
        num_covariates = 4  # glucose, carbs, exercise, stress
        num_treatments = self.insulin_encoder.get_treatment_input_dim()
        num_outputs = 1  # glucose prediction
        
        # Initialize arrays
        current_covariates = np.zeros((num_sequences, self.max_sequence_length, num_covariates))
        previous_treatments = np.zeros((num_sequences, self.max_sequence_length, num_treatments))
        current_treatments = np.zeros((num_sequences, self.max_sequence_length, num_treatments))
        outputs = np.zeros((num_sequences, self.max_sequence_length, num_outputs))
        active_entries = np.zeros((num_sequences, self.max_sequence_length, num_outputs))
        
        for i, seq_data in enumerate(sequences):
            # Ensure exact length match by truncating if necessary
            seq_len = min(len(seq_data), self.max_sequence_length)
            
            # Extract covariates (glucose, carbs, exercise, stress)
            current_covariates[i, :seq_len, 0] = seq_data['glucose'].values[:seq_len]
            current_covariates[i, :seq_len, 1] = seq_data['carbs'].values[:seq_len]
            current_covariates[i, :seq_len, 2] = seq_data['exercise'].values[:seq_len]
            current_covariates[i, :seq_len, 3] = seq_data['stress'].values[:seq_len]
            
            # Convert insulin doses to discrete levels and encode
            insulin_doses = seq_data['insulin'].values[:seq_len]
            discrete_levels = self.insulin_encoder.continuous_to_discrete(insulin_doses)
            encoded_treatments = self.insulin_encoder.encode_for_model(discrete_levels)
            
            # Integer encoding: single dimension
            current_treatments[i, :seq_len, 0] = encoded_treatments
            # Previous treatments are shifted by one timestep
            if seq_len > 1:
                previous_treatments[i, 1:seq_len, 0] = encoded_treatments[:-1]
            
            # Outputs are glucose values (for next-step prediction)
            if seq_len > 1:
                outputs[i, :seq_len-1, 0] = seq_data['glucose'].values[1:seq_len]  # Shift by one for prediction
                outputs[i, seq_len-1, 0] = seq_data['glucose'].values[seq_len-1]  # Last value unchanged
            
            # Active entries mask (mark valid entries)
            active_entries[i, :seq_len, 0] = 1.0
        
        # Normalize outputs (glucose values)
        output_means = np.mean(outputs[active_entries == 1])
        output_stds = np.std(outputs[active_entries == 1])
        normalized_outputs = (outputs - output_means) / output_stds
        
        return {
            'current_covariates': current_covariates,
            'previous_treatments': previous_treatments,
            'current_treatments': current_treatments,
            'outputs': normalized_outputs,
            'active_entries': active_entries,
            'unscaled_outputs': outputs,
            'output_means': output_means,
            'output_stds': output_stds,
            'sequence_lengths': np.full(num_sequences, self.max_sequence_length)
        }


# Utility functions
def process_diabetes_data_for_crn(patient_data, max_sequence_length=60, num_dose_levels=5):
    """
    One-stop function to process diabetes data for CRN training.
    
    Args:
        patient_data: DataFrame from diabetes-data-api
        max_sequence_length: Maximum sequence length for CRN
        num_dose_levels: Number of discrete dose categories
        
    Returns:
        Dictionary with CRN-compatible training data
    """
    encoder = InsulinEncoder(num_dose_levels=num_dose_levels)
    processor = DiabetesDataProcessor(encoder, max_sequence_length=max_sequence_length)
    return processor.process_patient_data(patient_data)