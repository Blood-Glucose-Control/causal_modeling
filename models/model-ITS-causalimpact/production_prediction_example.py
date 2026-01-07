import pandas as pd
from its_package.prediction.model_predictor import InterventionPredictor
from its_package.its_models.causalimpact_model import CausalImpactModel
from its_package.its_models.statsmodels_its import StatsmodelsITSModel
from its_package.data_handling.data_loader import load_csv_data, clean_data_for_causalimpact

# 1. Load sample data (historical data only)
data = load_csv_data("../synthetic_data/data/ml_dataset.csv")
cutoff_time = "2023-01-01 12:00:00"  # Some time point
historical_data = data.loc[:cutoff_time].copy()

# 2. Define pre-period and post-period for training
event_time = pd.Timestamp(cutoff_time) 
pre_start = event_time - pd.Timedelta(hours=1)
pre_end = event_time
pre_period = [pre_start, pre_end]

# 3. Train models (this would be done beforehand)
# CausalImpact model
ci_data = clean_data_for_causalimpact(historical_data.loc[pre_start:pre_end])
ci_model = CausalImpactModel()
ci_model.fit(ci_data, pre_period, [pre_end, pre_end + pd.Timedelta(minutes=1)])  # Dummy post-period
ci_model_path = ci_model.save_model("trained_causalimpact_model.pkl")

# Statsmodels ITS model
sm_model = StatsmodelsITSModel()
sm_model.fit(historical_data.loc[pre_start:pre_end], pre_period, [pre_end, pre_end + pd.Timedelta(minutes=1)])  # Dummy post-period
sm_model_path = sm_model.save_model("trained_statsmodels_model.pkl")

# 4. In production: Load model and make predictions
# This is what would happen in your production environment
predictor = InterventionPredictor()

# Load CausalImpact model and predict
predictor.load_model(ci_model_path)
ci_predictions = predictor.predict(
    pre_period_data=ci_data,
    intervention_time=event_time,
    post_period_length="30min",
    time_frequency="5min"
)

# Load Statsmodels model and predict
predictor.load_model(sm_model_path)
sm_predictions = predictor.predict(
    pre_period_data=historical_data.loc[pre_start:pre_end],
    intervention_time=event_time,
    post_period_length="30min",
    intervention_value=5,  # Value of intervention if needed
    time_frequency="5min"
)

# 5. Plot predictions
predictor.plot_prediction(
    historical_data.loc[pre_start:pre_end], 
    ci_predictions, 
    event_time, 
    target_col='glucose', 
    output_path="causalimpact_prediction.png"
)

predictor.plot_prediction(
    historical_data.loc[pre_start:pre_end], 
    sm_predictions, 
    event_time, 
    target_col='glucose', 
    output_path="statsmodels_prediction.png"
)

# 6. Print predicted values
print("\nCausalImpact Predictions:")
print(ci_predictions)

print("\nStatsmodels Predictions:")
print(sm_predictions)
