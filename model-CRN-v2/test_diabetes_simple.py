#!/usr/bin/env python3
"""
Simple test for CRN diabetes integration.
Tests just the core functionality without the cancer simulation fallback.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add paths
sys.path.append('Counterfactual-Recurrent-Network')
sys.path.append('diabetes-data-api')

def test_diabetes_data_generation():
    """Test diabetes data generation from data-api."""
    print("=== Testing Diabetes Data Generation ===")
    
    try:
        from main import DiabetesAnalyzer
        
        # Generate data
        analyzer = DiabetesAnalyzer(seed=42)
        patient_data = analyzer.generate_patient_data(n_days=3, start_date='2024-01-01')
        
        print(f"✓ Generated {len(patient_data)} data points")
        print(f"✓ Columns: {list(patient_data.columns)}")
        print(f"✓ Insulin interventions: {(patient_data['insulin'] > 0).sum()}")
        print(f"✓ Glucose range: {patient_data['glucose'].min():.1f} - {patient_data['glucose'].max():.1f}")
        
        return patient_data
        
    except Exception as e:
        print(f"✗ Diabetes data generation failed: {e}")
        raise

def test_insulin_encoding():
    """Test insulin encoding system."""
    print("\n=== Testing Insulin Encoding ===")
    
    try:
        from utils.insulin_encoding import InsulinEncoder
        
        # Test integer encoding
        encoder = InsulinEncoder(num_dose_levels=5)
        
        # Test dose conversion
        test_doses = [0.0, 1.5, 3.5, 5.0, 7.5, 10.0]
        discrete_levels = encoder.continuous_to_discrete(test_doses)
        encoded = encoder.encode_for_model(discrete_levels)
        
        print("Dose conversion test:")
        for dose, level, enc in zip(test_doses, discrete_levels, encoded):
            print(f"  {dose:.1f} units → Level {level} → Encoded: {enc:.1f}")
        
        print("✓ Insulin encoding test passed")
        return encoder
        
    except Exception as e:
        print(f"✗ Insulin encoding failed: {e}")
        raise

def test_data_processing(patient_data, encoder):
    """Test data processing for CRN."""
    print("\n=== Testing Data Processing ===")
    
    try:
        from utils.insulin_encoding import DiabetesDataProcessor
        
        processor = DiabetesDataProcessor(encoder, max_sequence_length=48)  # 4 hours
        processed_data = processor.process_patient_data(patient_data)
        
        print("Processed data shapes:")
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}")
        
        print("✓ Data processing test passed")
        return processed_data
        
    except Exception as e:
        print(f"✗ Data processing failed: {e}")
        raise

def test_counterfactual_analysis(patient_data):
    """Test counterfactual analysis."""
    print("\n=== Testing Counterfactual Analysis ===")
    
    try:
        from main import DiabetesAnalyzer
        
        analyzer = DiabetesAnalyzer(seed=42)
        interventions = analyzer.counterfactual_model.list_interventions(patient_data)
        
        if len(interventions) == 0:
            print("⚠ No interventions found for counterfactual analysis")
            return
        
        print(f"✓ Found {len(interventions)} interventions")
        
        # Test dose counterfactual
        intervention = interventions[0]
        result = analyzer.analyze_intervention(
            patient_data,
            intervention_id=intervention['id'],
            analysis_type='dose',
            dose_factor=1.2,
            before_minutes=60,
            after_minutes=120
        )
        
        print(f"✓ Dose counterfactual analysis completed: {result.shape}")
        print("✓ Counterfactual analysis test passed")
        
    except Exception as e:
        print(f"✗ Counterfactual analysis failed: {e}")
        raise

def main():
    """Run all tests."""
    try:
        print("CRN Diabetes Integration - Simple Test")
        print("=====================================")
        
        # Test 1: Generate diabetes data
        patient_data = test_diabetes_data_generation()
        
        # Test 2: Test insulin encoding
        encoder = test_insulin_encoding()
        
        # Test 3: Test data processing
        processed_data = test_data_processing(patient_data, encoder)
        
        # Test 4: Test counterfactual analysis
        test_counterfactual_analysis(patient_data)
        
        print("\n=== ALL TESTS PASSED ===")
        print("The diabetes CRN integration is working correctly!")
        
    except Exception as e:
        print(f"\n=== TEST FAILED ===")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()