# Copyright (c) 2024, Diabetes CRN Adaptation
"""
Modular insulin dose encoding system for CRN diabetes adaptation.

This module provides different encoding strategies for ordinal insulin dosage levels,
enabling easy experimentation and comparison of encoding approaches.
"""

import numpy as np
import tensorflow as tf


class InsulinEncoder:
    """
    Base class for insulin dose encoding strategies.
    
    Supports different encoding approaches for ordinal insulin doses (1-5 levels)
    to work with the CRN adversarial training framework.
    """
    
    def __init__(self, num_dose_levels=5, encoding_type='integer'):
        """
        Initialize insulin encoder.
        
        Args:
            num_dose_levels (int): Number of ordinal dose levels (default: 5)
            encoding_type (str): Encoding strategy - 'integer', 'onehot', 'embedding'
        """
        self.num_dose_levels = num_dose_levels
        self.encoding_type = encoding_type
        self.dose_values = np.arange(1, num_dose_levels + 1)  # [1, 2, 3, 4, 5]
        
    def encode_doses(self, dose_labels):
        """
        Encode dose labels according to specified strategy.
        
        Args:
            dose_labels (np.array): Integer dose levels [1, 2, 3, 4, 5]
            
        Returns:
            np.array: Encoded dose representations
        """
        if self.encoding_type == 'integer':
            return self._integer_encode(dose_labels)
        elif self.encoding_type == 'onehot':
            return self._onehot_encode(dose_labels)
        elif self.encoding_type == 'embedding':
            return self._embedding_encode(dose_labels)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
            
    def _integer_encode(self, dose_labels):
        """
        Integer encoding: map doses 1-5 to integer values.
        
        This is the primary approach for ordinal insulin doses.
        Preserves ordinality for regression adversary.
        """
        # Normalize to [0, 1] range for better neural network training
        normalized_doses = (dose_labels - 1) / (self.num_dose_levels - 1)
        return normalized_doses.astype(np.float32)
        
    def _onehot_encode(self, dose_labels):
        """
        One-hot encoding: traditional categorical approach.
        
        Included for comparison with cancer model approach.
        Loses ordinality information.
        """
        one_hot = np.zeros((len(dose_labels), self.num_dose_levels))
        for i, dose in enumerate(dose_labels):
            if 1 <= dose <= self.num_dose_levels:
                one_hot[i, dose - 1] = 1
        return one_hot.astype(np.float32)
        
    def _embedding_encode(self, dose_labels):
        """
        Embedding encoding: learnable dense representations.
        
        Maps dose levels to learnable embeddings.
        Can capture ordinality through training.
        """
        # For embedding, return integer indices for lookup
        return (dose_labels - 1).astype(np.int32)  # Convert to 0-indexed
        
    def decode_doses(self, encoded_doses):
        """
        Decode encoded representations back to dose levels.
        
        Args:
            encoded_doses (np.array): Encoded dose representations
            
        Returns:
            np.array: Original dose levels [1, 2, 3, 4, 5]
        """
        if self.encoding_type == 'integer':
            return self._integer_decode(encoded_doses)
        elif self.encoding_type == 'onehot':
            return self._onehot_decode(encoded_doses)
        elif self.encoding_type == 'embedding':
            return self._embedding_decode(encoded_doses)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
            
    def _integer_decode(self, encoded_doses):
        """Decode integer-encoded doses back to levels 1-5."""
        # Denormalize and round to nearest integer
        denormalized = encoded_doses * (self.num_dose_levels - 1) + 1
        return np.round(np.clip(denormalized, 1, self.num_dose_levels)).astype(np.int32)
        
    def _onehot_decode(self, encoded_doses):
        """Decode one-hot encoded doses back to levels 1-5."""
        return np.argmax(encoded_doses, axis=-1) + 1
        
    def _embedding_decode(self, encoded_doses):
        """Decode embedding indices back to levels 1-5."""
        return encoded_doses + 1
        
    def get_output_dim(self):
        """
        Get output dimension for encoded doses.
        
        Returns:
            int: Dimension of encoded representation
        """
        if self.encoding_type == 'integer':
            return 1
        elif self.encoding_type == 'onehot':
            return self.num_dose_levels
        elif self.encoding_type == 'embedding':
            return 1  # Just the index
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")


def process_diabetes_treatments(raw_insulin_doses, encoding_type='integer', num_dose_levels=5):
    """
    Process raw insulin dose data for CRN training.
    
    Converts continuous insulin doses to ordinal categories and encodes them
    according to the specified strategy.
    
    Args:
        raw_insulin_doses (np.array): Raw insulin doses (continuous values)
        encoding_type (str): Encoding strategy - 'integer', 'onehot', 'embedding'
        num_dose_levels (int): Number of dose levels to discretize into
        
    Returns:
        tuple: (encoded_treatments, encoder_instance)
    """
    # Discretize continuous doses into ordinal levels
    dose_levels = discretize_insulin_doses(raw_insulin_doses, num_dose_levels)
    
    # Initialize encoder
    encoder = InsulinEncoder(num_dose_levels=num_dose_levels, encoding_type=encoding_type)
    
    # Encode dose levels
    encoded_treatments = encoder.encode_doses(dose_levels)
    
    return encoded_treatments, encoder


def discretize_insulin_doses(insulin_doses, num_levels=5):
    """
    Discretize continuous insulin doses into ordinal levels.
    
    Uses quantile-based binning to ensure balanced distribution across levels.
    
    Args:
        insulin_doses (np.array): Continuous insulin dose values
        num_levels (int): Number of ordinal levels (default: 5)
        
    Returns:
        np.array: Discretized dose levels [1, 2, 3, 4, 5]
    """
    # Remove zero doses (no insulin) - handle separately if needed
    nonzero_doses = insulin_doses[insulin_doses > 0]
    
    if len(nonzero_doses) == 0:
        return np.ones_like(insulin_doses, dtype=np.int32)  # All level 1
        
    # Create quantile-based bins
    quantiles = np.linspace(0, 1, num_levels + 1)
    bin_edges = np.quantile(nonzero_doses, quantiles)
    bin_edges[0] = 0  # Include zero doses in first bin
    bin_edges[-1] = np.inf  # Handle any outliers
    
    # Discretize doses
    dose_levels = np.digitize(insulin_doses, bin_edges)
    dose_levels = np.clip(dose_levels, 1, num_levels)  # Ensure range [1, num_levels]
    
    return dose_levels.astype(np.int32)


def create_embedding_layer(num_dose_levels, embedding_dim=8, name_scope="dose_embedding"):
    """
    Create TensorFlow embedding layer for dose encoding.
    
    Args:
        num_dose_levels (int): Number of dose levels
        embedding_dim (int): Dimension of embedding vectors
        name_scope (str): TensorFlow variable scope name
        
    Returns:
        function: Embedding function that takes dose indices and returns embeddings
    """
    def embedding_lookup(dose_indices):
        with tf.compat.v1.variable_scope(name_scope, reuse=tf.compat.v1.AUTO_REUSE):
            embedding_matrix = tf.compat.v1.get_variable(
                "embedding_matrix",
                shape=[num_dose_levels, embedding_dim],
                initializer=tf.compat.v1.initializers.random_normal(stddev=0.1)
            )
            embeddings = tf.nn.embedding_lookup(embedding_matrix, dose_indices)
            return embeddings
    
    return embedding_lookup


# Example usage and testing functions
def test_insulin_encoding():
    """Test function to verify encoding/decoding consistency."""
    
    # Test data
    sample_doses = np.array([1, 2, 3, 4, 5, 2, 4, 1, 5, 3])
    
    print("Testing Insulin Encoding System")
    print("=" * 40)
    
    for encoding_type in ['integer', 'onehot', 'embedding']:
        print(f"\nTesting {encoding_type} encoding:")
        
        encoder = InsulinEncoder(num_dose_levels=5, encoding_type=encoding_type)
        
        # Encode
        encoded = encoder.encode_doses(sample_doses)
        print(f"Original doses: {sample_doses}")
        print(f"Encoded shape: {encoded.shape}")
        print(f"Encoded (first 5): {encoded[:5]}")
        
        # Decode
        decoded = encoder.decode_doses(encoded)
        print(f"Decoded doses: {decoded}")
        
        # Check consistency
        consistent = np.array_equal(sample_doses, decoded)
        print(f"Encoding/decoding consistent: {consistent}")


if __name__ == "__main__":
    test_insulin_encoding()