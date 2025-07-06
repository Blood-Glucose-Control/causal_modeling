#!/bin/bash

# Launch CRN Interactive Counterfactual Analysis App
echo "🩸 Launching CRN Interactive Counterfactual Analysis..."
echo "========================================================="
echo ""
echo "Features:"
echo "  📅 Date picker for analysis"
echo "  🩸 Glucose timeline visualization"
echo "  💉 Interactive insulin dose modification"
echo "  📊 Real-time counterfactual predictions"
echo "  📈 N-step ahead glucose forecasting"
echo ""
echo "The app will open in your default web browser."
echo "Use Ctrl+C to stop the server."
echo ""

# Activate virtual environment
source crn_env/bin/activate

# Launch Streamlit app
streamlit run interactive_counterfactual_app.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false