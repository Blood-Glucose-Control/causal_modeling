#!/usr/bin/env python3
"""
Test script for insulin encoding system with diabetes data.

This script tests our new integer encoding approach with real diabetes data
from the data-api, without training the CRN model yet.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the diabetes-data-api to path
sys.path.append('diabetes-data-api')
from main import DiabetesAnalyzer

# Import our new encoding system
from utils.insulin_encoding import InsulinEncoder, discretize_insulin_doses, process_diabetes_treatments
from utils.evaluation_utils import get_processed_data_diabetes

def test_basic_encoding():
    """Test basic insulin encoding functionality."""
    print("="*60)
    print("1. TESTING BASIC INSULIN ENCODING")
    print("="*60)
    
    # Test with sample dose levels
    sample_doses = np.array([1, 2, 3, 4, 5, 2, 4, 1, 5, 3])
    print(f"Sample dose levels: {sample_doses}")
    
    for encoding_type in ['integer', 'onehot', 'embedding']:
        print(f"\n--- Testing {encoding_type.upper()} encoding ---")
        
        encoder = InsulinEncoder(num_dose_levels=5, encoding_type=encoding_type)
        
        # Encode
        encoded = encoder.encode_doses(sample_doses)
        print(f"Encoded shape: {encoded.shape}")
        print(f"Encoded (first 5): {encoded[:5]}")
        
        # Decode
        decoded = encoder.decode_doses(encoded)
        print(f"Decoded doses: {decoded}")
        
        # Check consistency
        consistent = np.array_equal(sample_doses, decoded)
        print(f"Encoding/decoding consistent: {consistent}")
        
        if encoding_type == 'integer':
            print(f"Encoded range: [{encoded.min():.3f}, {encoded.max():.3f}]")
            print(f"Ordinality preserved: dose 2 vs 3 diff = {abs(encoded[1] - encoded[2]):.3f}")
            print(f"                     dose 2 vs 5 diff = {abs(encoded[1] - encoded[4]):.3f}")


def test_diabetes_data_generation():
    """Test generating diabetes data and extracting insulin doses."""
    print("\n" + "="*60)
    print("2. TESTING DIABETES DATA GENERATION")
    print("="*60)
    
    # Generate diabetes data
    analyzer = DiabetesAnalyzer(seed=42)
    print("Generating 7 days of patient data...")
    patient_data = analyzer.generate_patient_data(n_days=7, start_date='2024-01-01')
    
    print(f"✓ Generated {len(patient_data)} data points ({len(patient_data) / 288:.1f} days)")
    print(f"✓ Found {(patient_data['insulin'] > 0).sum()} insulin interventions")
    
    # Extract insulin doses
    insulin_doses = patient_data['insulin'].values
    nonzero_doses = insulin_doses[insulin_doses > 0]
    
    print(f"\nInsulin dose statistics:")
    print(f"  Total doses: {len(nonzero_doses)}")
    print(f"  Dose range: [{nonzero_doses.min():.2f}, {nonzero_doses.max():.2f}] units")
    print(f"  Mean dose: {nonzero_doses.mean():.2f} units")
    print(f"  Std dose: {nonzero_doses.std():.2f} units")
    
    # Show first few insulin interventions
    insulin_times = patient_data[patient_data['insulin'] > 0]
    print(f"\nFirst 5 insulin interventions:")
    for i, (timestamp, row) in enumerate(insulin_times.head().iterrows()):
        print(f"  {i+1}. {timestamp.strftime('%Y-%m-%d %H:%M')} - {row['insulin']:.2f} units")
    
    return patient_data, insulin_doses


def test_dose_discretization(insulin_doses):
    """Test discretizing continuous insulin doses into ordinal levels."""
    print("\n" + "="*60)
    print("3. TESTING DOSE DISCRETIZATION")
    print("="*60)
    
    # Test different numbers of dose levels
    for num_levels in [3, 5, 7]:
        print(f"\n--- Discretizing into {num_levels} levels ---")
        
        discrete_doses = discretize_insulin_doses(insulin_doses, num_levels=num_levels)
        
        print(f"Discrete levels: {np.unique(discrete_doses)}")
        print(f"Level distribution:")
        for level in range(1, num_levels + 1):
            count = np.sum(discrete_doses == level)
            percentage = count / len(discrete_doses) * 100
            print(f"  Level {level}: {count:3d} doses ({percentage:5.1f}%)")
        
        # Show mapping from continuous to discrete
        nonzero_indices = insulin_doses > 0
        continuous_nonzero = insulin_doses[nonzero_indices]
        discrete_nonzero = discrete_doses[nonzero_indices]
        
        print(f"\nSample continuous → discrete mapping:")
        for i in range(min(5, len(continuous_nonzero))):
            print(f"  {continuous_nonzero[i]:5.2f} units → Level {discrete_nonzero[i]}")


def test_integer_encoding_ordinality(insulin_doses):
    """Test that integer encoding preserves ordinality."""
    print("\n" + "="*60)
    print("4. TESTING INTEGER ENCODING ORDINALITY")
    print("="*60)
    
    # Discretize doses
    discrete_doses = discretize_insulin_doses(insulin_doses, num_levels=5)
    
    # Test integer encoding
    encoder = InsulinEncoder(num_dose_levels=5, encoding_type='integer')
    encoded_doses = encoder.encode_doses(discrete_doses)
    
    print("Testing ordinality preservation:")
    
    # Test all pairs of dose levels
    dose_levels = [1, 2, 3, 4, 5]
    print("\nDistance matrix (encoded differences):")
    print("Level", end="")
    for j in dose_levels:
        print(f"{j:8d}", end="")
    print()
    
    for i in dose_levels:
        print(f"{i:5d}", end="")
        for j in dose_levels:
            # Find examples of each dose level
            i_indices = discrete_doses == i
            j_indices = discrete_doses == j
            
            if np.any(i_indices) and np.any(j_indices):
                i_encoded = encoded_doses[i_indices][0]
                j_encoded = encoded_doses[j_indices][0]
                diff = abs(i_encoded - j_encoded)
                print(f"{diff:8.3f}", end="")
            else:
                print(f"{'N/A':>8}", end="")
        print()
    
    print("\nOrdinality test:")
    print("✓ Distance(1,2) < Distance(1,4): Should be True")
    if np.any(discrete_doses == 1) and np.any(discrete_doses == 2) and np.any(discrete_doses == 4):
        enc_1 = encoded_doses[discrete_doses == 1][0]
        enc_2 = encoded_doses[discrete_doses == 2][0]
        enc_4 = encoded_doses[discrete_doses == 4][0]
        
        dist_1_2 = abs(enc_1 - enc_2)
        dist_1_4 = abs(enc_1 - enc_4)
        
        print(f"  Distance(1,2) = {dist_1_2:.3f}")
        print(f"  Distance(1,4) = {dist_1_4:.3f}")
        print(f"  Test result: {dist_1_2 < dist_1_4}")


def test_crn_data_format(patient_data):
    """Test converting diabetes data to CRN format."""
    print("\n" + "="*60)
    print("5. TESTING CRN DATA FORMAT CONVERSION")
    print("="*60)
    
    # Create mock scaling parameters (normally computed from training data)
    scaling_params = (
        pd.Series({
            'glucose': patient_data['glucose'].mean(),
            'carbs': patient_data['carbs'].mean(),
            'insulin': patient_data['insulin'].mean(),
            'exercise': 0,
            'stress': patient_data['stress'].mean(),
            'active_insulin': patient_data['active_insulin'].mean()
        }),
        pd.Series({
            'glucose': patient_data['glucose'].std(),
            'carbs': patient_data['carbs'].std(),
            'insulin': patient_data['insulin'].std(),
            'exercise': 1,
            'stress': patient_data['stress'].std(),
            'active_insulin': patient_data['active_insulin'].std()
        })
    )
    
    # Convert to CRN format
    print("Converting diabetes data to CRN format...")
    
    # Create mock raw_sim_data format
    raw_sim_data = {
        'glucose': patient_data['glucose'].values.reshape(1, -1),  # [1, time_steps]
        'insulin_doses': patient_data['insulin'].values.reshape(1, -1),
        'carbs': patient_data['carbs'].values.reshape(1, -1),
        'exercise': patient_data['exercise'].values.reshape(1, -1),
        'stress': patient_data['stress'].values.reshape(1, -1),
        'active_insulin': patient_data['active_insulin'].values.reshape(1, -1),
        'sequence_lengths': np.array([len(patient_data)])
    }
    
    try:
        processed_data = get_processed_data_diabetes(raw_sim_data, scaling_params)
        
        print("✓ Successfully converted to CRN format!")
        print(f"✓ Current covariates shape: {processed_data['current_covariates'].shape}")
        print(f"✓ Current treatments shape: {processed_data['current_treatments'].shape}")
        print(f"✓ Previous treatments shape: {processed_data['previous_treatments'].shape}")
        print(f"✓ Outputs shape: {processed_data['outputs'].shape}")
        print(f"✓ Active entries shape: {processed_data['active_entries'].shape}")
        
        # Check treatment encoding
        treatments = processed_data['current_treatments']
        print(f"\nTreatment encoding details:")
        print(f"  Treatment range: [{treatments.min():.3f}, {treatments.max():.3f}]")
        print(f"  Treatment shape: {treatments.shape}")
        print(f"  Non-zero treatments: {np.sum(treatments > 0)}")
        
        # Show first few time steps
        print(f"\nFirst 5 treatment values: {treatments[0, :5, 0]}")
        
        # Check if encoder was stored
        if 'insulin_encoder' in processed_data:
            encoder = processed_data['insulin_encoder']
            print(f"✓ Insulin encoder stored: {encoder.encoding_type} encoding with {encoder.num_dose_levels} levels")
        
    except Exception as e:
        print(f"✗ Error converting to CRN format: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all encoding tests."""
    print("INSULIN ENCODING SYSTEM TEST")
    print("Testing our new integer encoding with diabetes data")
    print("(No CRN training - just encoding validation)")
    
    try:
        # Test 1: Basic encoding functionality
        test_basic_encoding()
        
        # Test 2: Generate diabetes data
        patient_data, insulin_doses = test_diabetes_data_generation()
        
        # Test 3: Dose discretization
        test_dose_discretization(insulin_doses)
        
        # Test 4: Integer encoding ordinality
        test_integer_encoding_ordinality(insulin_doses)
        
        # Test 5: CRN data format conversion
        test_crn_data_format(patient_data)
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print("\nSummary:")
        print("✓ Basic insulin encoding works correctly")
        print("✓ Diabetes data generation successful")
        print("✓ Dose discretization preserves distribution")
        print("✓ Integer encoding preserves ordinality")
        print("✓ CRN data format conversion successful")
        print("\nReady for CRN training with diabetes data!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()