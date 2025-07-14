# TODO.md: Modifying CRN Architecture for Ordinal Insulin Dosage Encoding

## Current Problem Analysis

The current CRN model uses **discrete one-hot encoding** for treatments:
- `CRN_model.py:82-97`: `build_treatment_assignments_one_hot()` method
- `utils/evaluation_utils.py:73-88`: Cancer treatment encoding (4 discrete categories)
- Current adversarial head: K-way softmax classifier with cross-entropy loss

For diabetes, insulin dosage is **ordinal** (1-5 categories representing increasing dose levels), requiring an approach that respects dose ordering.

## Solution: Integer Encoding + Regression Adversary

### Step 1: Create Modular Encoding System ✅
**New file**: `utils/insulin_encoding.py`
- Encapsulates all insulin dose encoding/decoding logic
- Provides clean interface for different encoding strategies
- Enables easy testing of alternative approaches

### Step 2: Implement Integer Encoding ✅
**Core components**:
- Map dose categories 1-5 to integer values [1, 2, 3, 4, 5]
- Optional: Add small embedding layer for dense representation
- Maintain compatibility with existing data pipeline

### Step 3: Replace Adversarial Head ✅
**Modify**: `CRN_model.py`
- Replace `build_treatment_assignments_one_hot()` with `build_treatment_assignments_regression()`
- Change from K-way softmax to single-output regression
- Switch loss from cross-entropy to MSE: respects dose ordering (confusing dose 2 vs 3 less penalized than 2 vs 5)

### Step 4: Update Data Processing ✅
**Modify**: `utils/evaluation_utils.py`
- Replace cancer-specific one-hot encoding with integer dose encoding
- Update data preprocessing pipeline for diabetes data integration
- Maintain backward compatibility for testing

### Step 5: Integration Testing 🔄
**Test scenarios**:
- Verify integer encoding preserves ordinality
- Compare MSE vs cross-entropy adversarial performance
- Validate with diabetes data from data-api module

## Implementation Details

### Minimal Surgery Approach
1. **Add new methods** alongside existing ones (don't remove cancer functionality)
2. **Parameterize encoding type** in model initialization
3. **Localize changes** to adversarial head ($G_a$) - no need to re-derive link functions
4. **Preserve existing interfaces** for cancer model compatibility

### Key Files Modified
- `utils/insulin_encoding.py` (new) ✅
- `CRN_model.py` (add regression adversary method) ✅
- `utils/evaluation_utils.py` (add integer encoding function) ✅
- `test_crn.py` (add encoding type parameter) ✅

## Architecture Changes

### Current Cancer Model Flow
```
Treatments (binary) → One-hot encoding [4 classes] → Softmax adversary → Cross-entropy loss
```

### New Diabetes Model Flow
```
Insulin doses (1-5) → Integer encoding → Optional embedding → Regression adversary → MSE loss
```

## Usage Instructions

### For Diabetes Training (Integer Encoding)
```bash
uv run python test_crn.py --model_name=diabetes_test --treatment_encoding=integer --num_dose_levels=5
```

### For Cancer Training (Original One-hot)
```bash
uv run python test_crn.py --model_name=cancer_test --treatment_encoding=onehot --num_treatments=4
```

## Benefits of This Approach

1. **Ordinal Awareness**: MSE loss naturally encodes that dose differences matter (2 vs 3 is less bad than 2 vs 5)
2. **Minimal Surgery**: Changes localized to adversarial head, no need to re-derive link functions
3. **Modular Design**: Easy to swap encoding strategies for experimentation
4. **Backward Compatibility**: Cancer model functionality preserved
5. **Extensible**: Framework supports other ordinal treatment scenarios

## Next Steps

1. ✅ Implement core integer encoding system
2. ✅ Add regression adversarial training
3. ✅ Update data processing pipeline
4. 🔄 Test with synthetic diabetes data
5. 📋 Integrate with data-api counterfactual analysis
6. 📋 Compare performance: integer vs one-hot encoding
7. 📋 Validate clinical applicability with real patient data

## Notes

- Integer encoding approach chosen for its simplicity and theoretical soundness
- Alternative approaches (ordinal regression, ranking losses) available in `insulin_encoding.py` for future exploration
- This framework supports any ordinal treatment domain, not just diabetes