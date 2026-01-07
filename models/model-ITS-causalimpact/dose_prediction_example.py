import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from its_package.prediction.model_predictor import InterventionPredictor
from its_package.data_handling.data_loader import load_csv_data

# 1. Load pre-trained models
predictor = InterventionPredictor()

# 2. Define scenario parameters
intervention_time = pd.Timestamp("2023-01-01 12:00:00")  # Example intervention time
actual_dose = 5.0  # The dose used in the actual data/model
counterfactual_doses = [0.0, 2.5, 7.5, 10.0]  # Doses to simulate

# 3. Load some historical data for reference
data = load_csv_data("../synthetic_data/data/ml_dataset.csv")
pre_data = data.loc[intervention_time - pd.Timedelta(hours=1):intervention_time]

# 4. Generate predictions with CausalImpact model
print("Generating predictions with CausalImpact model...")
predictor.load_model("trained_causalimpact_model.pkl")
ci_results = predictor.predict_dose_counterfactuals(
    pre_period_data=pre_data,
    intervention_time=intervention_time,
    post_period_length="2h",
    actual_dose=actual_dose,
    counterfactual_doses=counterfactual_doses,
    time_frequency="5min"
)

# 5. Visualize CausalImpact predictions
predictor.plot_dose_counterfactuals(
    pre_data,
    ci_results,
    intervention_time,
    target_col='glucose',
    output_path="causalimpact_dose_counterfactuals.png"
)

# 6. Generate predictions with Statsmodels ITS
print("Generating predictions with Statsmodels ITS model...")
predictor.load_model("trained_statsmodels_model.pkl")
sm_results = predictor.predict_dose_counterfactuals(
    pre_period_data=pre_data,
    intervention_time=intervention_time,
    post_period_length="2h",
    actual_dose=actual_dose,
    counterfactual_doses=counterfactual_doses,
    time_frequency="5min"
)

# 7. Visualize Statsmodels predictions
predictor.plot_dose_counterfactuals(
    pre_data,
    sm_results,
    intervention_time,
    target_col='glucose',
    output_path="statsmodels_dose_counterfactuals.png"
)

# 8. Compare minimum glucose values across doses
print("\nMinimum Glucose Values Across Different Doses:")
print("=" * 50)
print(f"{'Dose':<10} {'CausalImpact Min':<20} {'StatsModels Min':<20}")
print("-" * 50)

for dose in [actual_dose] + counterfactual_doses:
    key = 'actual' if dose == actual_dose else f'dose_{dose}'
    ci_min = ci_results[key]['predicted'].min() if key in ci_results else float('nan')
    sm_min = sm_results[key]['predicted'].min() if key in sm_results else float('nan')
    print(f"{dose:<10.1f} {ci_min:<20.1f} {sm_min:<20.1f}")

# 9. Example: Find optimal dose for target glucose of 100 mg/dL
target_glucose = 100
print("\nFinding dose to achieve target glucose of 100 mg/dL...")

# This would typically involve more sophisticated optimization
# but we'll use a simplified approach for illustration
