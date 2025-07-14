#!/usr/bin/env python3
"""
Minimal Streamlit Glucose Visualizer

Interactive app to visualize blood glucose data and explore counterfactual scenarios.
Run with: streamlit run glucose_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from datetime import datetime, timedelta

# Add the diabetes-data-api to path
sys.path.append('diabetes-data-api')
from main import DiabetesAnalyzer

# Page config
st.set_page_config(
    page_title="Glucose Counterfactual Visualizer",
    page_icon="🩸",
    layout="wide"
)

@st.cache_data
def generate_patient_data(days=7, seed=42):
    """Generate patient data with caching."""
    analyzer = DiabetesAnalyzer(seed=seed)
    data = analyzer.generate_patient_data(n_days=days, start_date='2024-01-01')
    interventions = analyzer.counterfactual_model.list_interventions(data)
    return data, interventions, analyzer

def create_glucose_plot(patient_data, interventions, selected_intervention=None, counterfactual_data=None, dose_factor=1.0):
    """Create interactive glucose plot with Plotly."""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Blood Glucose', 'Insulin & Meals'),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Plot baseline glucose
    fig.add_trace(
        go.Scatter(
            x=patient_data.index,
            y=patient_data['glucose'],
            mode='lines',
            name='Blood Glucose',
            line=dict(color='blue', width=2),
            hovertemplate='%{x}<br>Glucose: %{y} mg/dL<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot intervention points
    intervention_times = [intervention['timestamp'] for intervention in interventions]
    intervention_doses = [intervention['dose'] for intervention in interventions]
    intervention_glucose = []
    
    for intervention in interventions:
        time_idx = abs(patient_data.index - intervention['timestamp']).argmin()
        glucose_val = patient_data['glucose'].iloc[time_idx]
        intervention_glucose.append(glucose_val)
    
    fig.add_trace(
        go.Scatter(
            x=intervention_times,
            y=intervention_glucose,
            mode='markers',
            name='Insulin Interventions',
            marker=dict(color='red', size=10, symbol='circle'),
            text=[f"{dose:.1f}u" for dose in intervention_doses],
            textposition="top center",
            hovertemplate='%{x}<br>Dose: %{text}<br>Glucose: %{y} mg/dL<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Highlight selected intervention
    if selected_intervention:
        fig.add_shape(
            type="line",
            x0=selected_intervention['timestamp'],
            x1=selected_intervention['timestamp'],
            y0=0, y1=1,
            yref="y domain",
            line=dict(color="red", width=3, dash="dash"),
            row=1, col=1
        )
        fig.add_annotation(
            x=selected_intervention['timestamp'],
            y=0.95,
            yref="y domain",
            text=f"Selected: {selected_intervention['dose']:.1f}u",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            row=1, col=1
        )
    
    # Plot counterfactual if available
    if counterfactual_data is not None:
        cf_meta = list(counterfactual_data.attrs.values())[-1]
        cf_col = f"cf{cf_meta['cf_number']}_glucose"
        
        # Only show counterfactual from intervention time onwards
        intervention_time = selected_intervention['timestamp']
        cf_post_intervention = counterfactual_data[counterfactual_data.index >= intervention_time]
        
        fig.add_trace(
            go.Scatter(
                x=cf_post_intervention.index,
                y=cf_post_intervention[cf_col],
                mode='lines',
                name=f'Counterfactual ({dose_factor:.1f}x dose)',
                line=dict(color='green', width=3, dash='dash'),
                hovertemplate='%{x}<br>CF Glucose: %{y} mg/dL<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add analysis window
        window_start = cf_meta['window_start']
        window_end = cf_meta['window_end']
        
        fig.add_shape(
            type="rect",
            x0=window_start, x1=window_end,
            y0=0, y1=1,
            yref="y domain",
            fillcolor="lightgray",
            opacity=0.2,
            layer="below",
            row=1, col=1
        )
    
    # Plot insulin doses
    insulin_mask = patient_data['insulin'] > 0
    insulin_times = patient_data.index[insulin_mask]
    insulin_doses = patient_data['insulin'][insulin_mask]
    
    fig.add_trace(
        go.Scatter(
            x=insulin_times,
            y=insulin_doses,
            mode='markers',
            name='Insulin Doses',
            marker=dict(color='red', size=8, symbol='circle'),
            hovertemplate='%{x}<br>Insulin: %{y:.1f} units<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Plot meals
    meal_mask = patient_data['carbs'] > 0
    meal_times = patient_data.index[meal_mask]
    meal_carbs = patient_data['carbs'][meal_mask]
    
    fig.add_trace(
        go.Bar(
            x=meal_times,
            y=meal_carbs,
            name='Meals (carbs)',
            marker_color='brown',
            opacity=0.6,
            hovertemplate='%{x}<br>Carbs: %{y}g<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Layout with constant uirevision to preserve all zoom/pan states
    fig.update_layout(
        height=600,
        title_text="Interactive Glucose & Counterfactual Visualizer",
        showlegend=True,
        hovermode='x unified',
        uirevision='preserve_zoom'  # Always preserve zoom/pan state
    )
    
    # Also set uirevision for axes to ensure zoom is preserved
    fig.update_xaxes(uirevision='preserve_zoom')
    fig.update_yaxes(uirevision='preserve_zoom')
    
    fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1)
    fig.update_yaxes(title_text="Insulin/Carbs", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    # No automatic zoom/pan - completely manual control
    
    return fig

def main():
    st.title("🩸 Glucose Counterfactual Visualizer")
    st.markdown("Explore insulin interventions and their counterfactual effects on blood glucose.")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Data generation
    days = st.sidebar.slider("Days of data", 3, 14, 7)
    seed = st.sidebar.number_input("Random seed", 1, 100, 42)
    
    if st.sidebar.button("Generate New Data"):
        st.cache_data.clear()
    
    # Generate data
    with st.spinner("Generating patient data..."):
        patient_data, interventions, analyzer = generate_patient_data(days, seed)
    
    st.sidebar.success(f"✓ {len(patient_data)} data points")
    st.sidebar.success(f"✓ {len(interventions)} interventions")
    
    # Intervention selection
    st.sidebar.subheader("Select Intervention")
    
    # Minimal session state (no zoom tracking needed)
    if 'last_selected_intervention' not in st.session_state:
        st.session_state.last_selected_intervention = None
    
    if interventions:
        intervention_options = [
            f"{i+1}: {intervention['timestamp'].strftime('%m/%d %H:%M')} - {intervention['dose']:.1f}u"
            for i, intervention in enumerate(interventions)
        ]
        
        selected_idx = st.sidebar.selectbox(
            "Choose intervention:",
            range(len(intervention_options)),
            format_func=lambda x: intervention_options[x]
        )
        
        selected_intervention = interventions[selected_idx]
        
        # Update session state with current intervention (no zoom behavior)
        st.session_state.last_selected_intervention = selected_intervention
        
        # Counterfactual controls
        st.sidebar.subheader("Counterfactual Scenario")
        dose_factor = st.sidebar.slider(
            "Dose multiplier",
            0.5, 2.0, 1.0, 0.1,
            help="1.0 = original dose, >1.0 = more insulin, <1.0 = less insulin"
        )
        
        # Generate counterfactual
        counterfactual_data = None
        if dose_factor != 1.0:
            with st.spinner("Generating counterfactual..."):
                try:
                    counterfactual_data = analyzer.analyze_intervention(
                        patient_data,
                        intervention_id=selected_intervention['id'],
                        analysis_type='dose',
                        dose_factor=dose_factor,
                        before_minutes=120,
                        after_minutes=360
                    )
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
        
        # Display intervention details
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Selected Intervention:**")
        st.sidebar.markdown(f"Time: {selected_intervention['timestamp']}")
        st.sidebar.markdown(f"Original: {selected_intervention['dose']:.1f} units")
        st.sidebar.markdown(f"Modified: {selected_intervention['dose'] * dose_factor:.1f} units")
        
    else:
        selected_intervention = None
        counterfactual_data = None
        st.sidebar.warning("No interventions found")
    
    # Main plot
    fig = create_glucose_plot(
        patient_data, 
        interventions, 
        selected_intervention, 
        counterfactual_data,
        dose_factor
    )
    
    # Use minimal key that only changes when data changes, not when controls change
    st.plotly_chart(fig, use_container_width=True, key=f"plot_{seed}_{days}")
    
    # Show counterfactual metrics
    if counterfactual_data is not None and selected_intervention:
        st.subheader("Counterfactual Analysis")
        
        cf_meta = list(counterfactual_data.attrs.values())[-1]
        cf_col = f"cf{cf_meta['cf_number']}_glucose"
        
        # Calculate differences at key time points
        intervention_time = selected_intervention['timestamp']
        
        cols = st.columns(4)
        
        for i, minutes_after in enumerate([30, 60, 120, 180]):
            target_time = intervention_time + timedelta(minutes=minutes_after)
            target_idx = abs(counterfactual_data.index - target_time).argmin()
            
            if target_idx < len(counterfactual_data):
                baseline_val = counterfactual_data['glucose'].iloc[target_idx]
                cf_val = counterfactual_data[cf_col].iloc[target_idx]
                diff = cf_val - baseline_val
                
                with cols[i]:
                    st.metric(
                        f"+{minutes_after}min",
                        f"{cf_val:.0f} mg/dL",
                        f"{diff:+.0f} mg/dL"
                    )
    
    # Instructions
    with st.expander("Instructions"):
        st.markdown("""
        1. **Select intervention** from the sidebar dropdown
        2. **Adjust dose multiplier** to explore "what if" scenarios
        3. **Zoom and pan** the plot for detailed time windows
        4. **Hover** over data points for exact values
        5. **Green dashed line** shows counterfactual glucose response
        6. **Metrics below** show glucose differences at key time points
        """)

if __name__ == "__main__":
    main()