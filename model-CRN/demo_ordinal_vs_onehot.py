#!/usr/bin/env python3

"""
Demonstration script comparing ordinal vs one-hot treatment encoding in CRN.
Shows the key differences in architecture and capabilities.
"""

import numpy as np
import logging
from utils.glucose_evaluation_utils import get_processed_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_glucose_data():
    """Create sample glucose data for demonstration."""
    np.random.seed(42)
    
    num_patients = 10
    sequence_length = 20
    
    # Generate realistic patterns
    raw_data = {
        'current_covariates': np.random.normal(120, 20, (num_patients, sequence_length, 2)),
        'current_treatments': np.random.exponential(2, (num_patients, sequence_length, 1)),  # Continuous insulin doses
        'outputs': np.random.normal(120, 30, (num_patients, sequence_length, 1))
    }
    
    return raw_data

def demo_encoding_comparison():
    """Demonstrate the difference between ordinal and one-hot encoding."""
    logger.info("=== CRN Ordinal vs One-Hot Treatment Encoding Demo ===")
    
    # Create sample data
    raw_data = create_sample_glucose_data()
    
    # Scaling parameters
    scaling_params = (
        {'glucose': 120.0, 'glucose_history': 120.0, 'insulin': 2.0},
        {'glucose': 30.0, 'glucose_history': 20.0, 'insulin': 1.5}
    )
    
    logger.info("\\n1. ORIGINAL ONE-HOT ENCODING:")
    logger.info("   • Treatment representation: Binary categories")
    logger.info("   • Domain adversary: Softmax classifier")
    logger.info("   • Loss function: Cross-entropy")
    logger.info("   • Capability: When to give insulin (timing)")
    
    onehot_data = get_processed_data(raw_data, scaling_params, ordinal_treatments=False)
    logger.info(f"   • Treatment tensor shape: {onehot_data['current_treatments'].shape}")
    logger.info(f"   • Number of treatment categories: {onehot_data['num_treatments']}")
    logger.info(f"   • Treatment example: {onehot_data['current_treatments'][0, 0]}")
    
    logger.info("\\n2. NEW ORDINAL ENCODING:")
    logger.info("   • Treatment representation: Continuous dosage values")
    logger.info("   • Domain adversary: MSE regressor")
    logger.info("   • Loss function: Mean squared error")
    logger.info("   • Capability: When AND how much insulin (timing + intensity)")
    
    ordinal_data = get_processed_data(raw_data, scaling_params, ordinal_treatments=True)
    logger.info(f"   • Treatment tensor shape: {ordinal_data['current_treatments'].shape}")
    logger.info(f"   • Number of treatment dimensions: {ordinal_data['num_treatments']}")
    logger.info(f"   • Treatment example: {ordinal_data['current_treatments'][0, 0]}")
    
    logger.info("\\n3. KEY ARCHITECTURAL DIFFERENCES:")
    logger.info("   One-Hot → Ordinal Changes:")
    logger.info("   • build_treatment_assignments_one_hot() → build_treatment_assignments_ordinal()")
    logger.info("   • tf.nn.softmax() → Linear output (no activation)")
    logger.info("   • Cross-entropy loss → MSE loss")
    logger.info("   • [1,0] or [0,1] → Single continuous value")
    
    logger.info("\\n4. BENEFITS OF ORDINAL ENCODING:")
    logger.info("   ✓ Models insulin dosage amounts (not just presence/absence)")
    logger.info("   ✓ Domain adversary tracks 'how far off' predictions are")
    logger.info("   ✓ Suitable for continuous treatment optimization")
    logger.info("   ✓ Better for personalized insulin dosing recommendations")
    
    logger.info("\\n5. TRAINING COMMAND:")
    logger.info("   # Ordinal (NEW):")
    logger.info("   python test_crn_glucose.py --ordinal_treatments=True")
    logger.info("   # One-hot (ORIGINAL):")
    logger.info("   python test_crn_glucose.py --ordinal_treatments=False")
    
    return True

if __name__ == '__main__':
    demo_encoding_comparison()
    logger.info("\\n✓ Demo completed successfully!")