#!/usr/bin/env python3

"""
Interactive Counterfactual Analysis Tool for CRN Ordinal Model

Features:
- Date picker to select analysis day
- Timeline view of glucose and insulin interventions  
- Interactive sliders to modify insulin doses
- Real-time prediction updates for N timesteps ahead
- Comparison between original and modified scenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add local imports
sys.path.append('.')
from simple_test_generator import SimpleGlucoseGenerator

class InteractiveCounterfactualAnalyzer:
    def __init__(self):
        self.generator = SimpleGlucoseGenerator(seed=42)
        self.data_cache = {}
        
    @st.cache_data
    def generate_glucose_data(_self, days=7, start_date='2024-01-01'):
        """Generate glucose data with caching"""
        return _self.generator.generate_data(days=days, start_date=start_date)
    
    def simulate_glucose_response(self, base_data, modified_insulin, start_idx, n_steps=48):
        """
        Simulate glucose response with modified insulin starting from start_idx.
        
        Args:
            base_data: Original glucose data
            modified_insulin: New insulin dosing schedule
            start_idx: Index to start modifications from
            n_steps: Number of steps to predict ahead
        
        Returns:
            Modified glucose predictions
        """
        # Create modified scenario
        df = base_data.copy()
        df.loc[df.index[start_idx:start_idx+n_steps], 'insulin'] = modified_insulin[:n_steps]
        
        # Re-simulate glucose starting from modification point
        glucose = np.array(df['predicted_glucose'] if 'predicted_glucose' in df.columns else df['glucose'])
        
        # Only re-calculate from the modification point onwards
        for t in range(start_idx, min(len(df), start_idx + n_steps)):
            insulin_activity = 0
            carb_impact = 0
            
            # Calculate lagged effects
            for past_t in range(max(0, t - self.generator.params['insulin_duration']//5), t):
                if df['insulin'].iloc[past_t] > 0:
                    time_diff = (t - past_t) * 5
                    insulin_activity += self.generator._insulin_curve(time_diff, df['insulin'].iloc[past_t])
            
            for past_t in range(max(0, t - self.generator.params['carb_duration']//5), t):
                if df['carbs'].iloc[past_t] > 0:
                    time_diff = (t - past_t) * 5
                    carb_impact += self.generator._carb_curve(time_diff, df['carbs'].iloc[past_t])
            
            # Calculate effects
            current_time = df.index[t]
            exercise_effect = 1 - (df['exercise'].iloc[t] * self.generator.params['exercise_sensitivity'] / 100)
            stress_effect = df['stress'].iloc[t] * self.generator.params['stress_effect']
            dawn_effect = self.generator._dawn_effect(current_time.hour + current_time.minute/60)
            
            # Calculate target glucose
            target_glucose = (
                self.generator.params['basal_glucose']
                + carb_impact * self.generator.params['carb_impact']
                - insulin_activity * self.generator.params['insulin_sensitivity'] * exercise_effect
                + stress_effect
                + dawn_effect
            )
            
            # Add momentum (glucose doesn't change instantly)
            if t > 0:
                glucose[t] = 0.9 * glucose[t-1] + 0.1 * target_glucose
            
        df['predicted_glucose'] = np.clip(glucose, 40, 400)
        return df

def main():
    st.set_page_config(
        page_title="CRN Interactive Counterfactual Analysis",
        page_icon="🩸",
        layout="wide"
    )
    
    st.title("🩸 CRN Interactive Counterfactual Analysis")
    st.markdown("**Explore how different insulin doses affect glucose predictions**")
    
    # Initialize analyzer
    analyzer = InteractiveCounterfactualAnalyzer()
    
    # Sidebar controls
    st.sidebar.header("📅 Analysis Controls")
    
    # Date selection
    start_date = st.sidebar.date_input(
        "Select Analysis Date",
        value=datetime(2024, 1, 15),
        min_value=datetime(2024, 1, 1),
        max_value=datetime(2024, 3, 31)
    )
    
    # Prediction horizon
    prediction_hours = st.sidebar.slider(
        "Prediction Horizon (hours)",
        min_value=1,
        max_value=24,
        value=8,
        help="How many hours ahead to predict"
    )
    prediction_steps = prediction_hours * 12  # 5-minute intervals
    
    # Generate data for selected date range
    with st.spinner("Generating glucose data..."):
        full_data = analyzer.generate_glucose_data(
            days=7, 
            start_date=start_date.strftime('%Y-%m-%d')
        )
    
    # Filter data for selected day
    day_start = pd.Timestamp(start_date)
    day_end = day_start + timedelta(days=1)
    day_data = full_data[(full_data.index >= day_start) & (full_data.index < day_end)].copy()
    
    if len(day_data) == 0:
        st.error("No data available for selected date. Try a different date.")
        return
    
    # Find insulin interventions for the day
    insulin_events = day_data[day_data['insulin'] > 0].copy()
    
    if len(insulin_events) == 0:
        st.warning("No insulin interventions found for this day. Generating synthetic interventions...")
        # Add some synthetic insulin events
        meal_times = day_data[day_data['carbs'] > 0].index
        for meal_time in meal_times:
            day_data.loc[meal_time, 'insulin'] = np.random.uniform(3, 8)
        insulin_events = day_data[day_data['insulin'] > 0].copy()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Intervention Controls")
    st.sidebar.markdown(f"Found **{len(insulin_events)}** insulin interventions")
    
    # Select intervention to modify
    if len(insulin_events) > 0:
        intervention_options = [
            f"{i+1}: {time.strftime('%H:%M')} - {dose:.1f}u" 
            for i, (time, dose) in enumerate(zip(insulin_events.index, insulin_events['insulin']))
        ]
        
        selected_intervention_idx = st.sidebar.selectbox(
            "Select Intervention to Modify",
            range(len(intervention_options)),
            format_func=lambda x: intervention_options[x]
        )
        
        selected_time = insulin_events.index[selected_intervention_idx]
        original_dose = insulin_events.iloc[selected_intervention_idx]['insulin']
        
        # Dose modification slider
        new_dose = st.sidebar.slider(
            f"New Insulin Dose (original: {original_dose:.1f}u)",
            min_value=0.0,
            max_value=20.0,
            value=float(original_dose),
            step=0.5,
            help="Modify the insulin dose for this intervention"
        )
        
        # Additional interventions
        st.sidebar.markdown("---")
        add_intervention = st.sidebar.checkbox("Add New Intervention")
        
        if add_intervention:
            new_time = st.sidebar.time_input(
                "New Intervention Time",
                value=datetime.combine(start_date, datetime.min.time()).time()
            )
            new_intervention_dose = st.sidebar.slider(
                "New Intervention Dose",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5
            )
        
        # Apply modifications
        modified_data = day_data.copy()
        
        # Modify selected intervention
        selected_data_idx = modified_data.index.get_loc(selected_time)
        modified_data.iloc[selected_data_idx, modified_data.columns.get_loc('insulin')] = new_dose
        
        # Add new intervention if specified
        if add_intervention:
            new_datetime = pd.Timestamp.combine(start_date, new_time)
            if new_datetime in modified_data.index:
                new_data_idx = modified_data.index.get_loc(new_datetime)
                modified_data.iloc[new_data_idx, modified_data.columns.get_loc('insulin')] = new_intervention_dose
        
        # Simulate counterfactual outcomes
        with st.spinner("Calculating counterfactual predictions..."):
            # Find the index in the full timeline for prediction
            prediction_start_idx = modified_data.index.get_loc(selected_time)
            
            # Create modified insulin schedule for prediction
            modified_insulin = modified_data['insulin'].values
            
            # Simulate glucose response for the rest of the day after the intervention
            n_steps = len(modified_data) - prediction_start_idx
            counterfactual_data = analyzer.simulate_glucose_response(
                modified_data, 
                modified_insulin, 
                prediction_start_idx, 
                n_steps=n_steps
            )
        
        # Create visualization
        st.markdown("---")
        
        # Main timeline plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Glucose Timeline", "Insulin Interventions"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Plot glucose timeline
        fig.add_trace(
            go.Scatter(
                x=day_data.index,
                y=day_data['glucose'],
                name="Original Glucose",
                line=dict(color='blue', width=2),
                mode='lines'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=counterfactual_data.index,
                y=counterfactual_data['predicted_glucose'],
                name="Modified Glucose",
                line=dict(color='red', width=2, dash='dash'),
                mode='lines'
            ),
            row=1, col=1
        )
        
        # Add glucose range guidelines
        fig.add_hline(y=100, line_dash="dot", line_color="green", annotation_text="Target", row=1, col=1)
        
        # Plot meals
        meal_times = day_data[day_data['carbs'] > 0]
        if len(meal_times) > 0:
            fig.add_trace(
                go.Scatter(
                    x=meal_times.index,
                    y=day_data.loc[meal_times.index, 'glucose'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name="Meals",
                    text=[f"{carbs}g" for carbs in meal_times['carbs']],
                    hovertemplate="Meal: %{text}<br>Time: %{x}<br>Glucose: %{y} mg/dL"
                ),
                row=1, col=1
            )
        
        # Add prominent marker for selected intervention on glucose timeline
        selected_glucose_value = day_data.loc[selected_time, 'glucose']
        fig.add_trace(
            go.Scatter(
                x=[selected_time],
                y=[selected_glucose_value],
                mode='markers',
                marker=dict(
                    symbol='diamond', 
                    size=20, 
                    color='yellow', 
                    line=dict(color='black', width=3)
                ),
                name="Selected Intervention Time",
                hovertemplate="Intervention Time: %{x}<br>Glucose: %{y} mg/dL<br>Dose: " + f"{new_dose:.1f}u",
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add vertical line to show intervention timing
        fig.add_shape(
            type="line",
            x0=selected_time, x1=selected_time,
            y0=0, y1=1,
            yref="y domain",
            line=dict(color="yellow", width=3, dash="dash"),
            row=1, col=1
        )
        
        # Add annotation for intervention
        fig.add_annotation(
            x=selected_time,
            y=selected_glucose_value + 20,
            text=f"Intervention: {new_dose:.1f}u",
            showarrow=True,
            arrowhead=2,
            arrowcolor="yellow",
            bgcolor="yellow",
            bordercolor="black",
            font=dict(color="black", size=12),
            row=1, col=1
        )
        
        # Plot insulin interventions
        fig.add_trace(
            go.Scatter(
                x=day_data.index,
                y=day_data['insulin'],
                name="Original Insulin",
                line=dict(color='blue'),
                mode='lines+markers',
                fill='tozeroy',
                fillcolor='rgba(0,100,80,0.2)'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=modified_data.index,
                y=modified_data['insulin'],
                name="Modified Insulin",
                line=dict(color='red', dash='dash'),
                mode='lines+markers',
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)'
            ),
            row=2, col=1
        )
        
        # Highlight selected intervention
        fig.add_trace(
            go.Scatter(
                x=[selected_time],
                y=[new_dose],
                mode='markers',
                marker=dict(symbol='star', size=15, color='yellow', line=dict(color='black', width=2)),
                name="Modified Intervention",
                showlegend=False
            ),
            row=2, col=1
        )
        
        
        # Update layout
        fig.update_layout(
            height=700,
            title=f"Interactive Counterfactual Analysis - {start_date.strftime('%Y-%m-%d')}",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1)
        fig.update_yaxes(title_text="Insulin (units)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        glucose_diff = counterfactual_data['predicted_glucose'] - day_data['glucose']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Glucose Change",
                f"{glucose_diff.mean():.1f} mg/dL",
                delta=f"{glucose_diff.mean():.1f}"
            )
        
        with col2:
            original_time_in_range = np.mean((day_data['glucose'] >= 70) & (day_data['glucose'] <= 180)) * 100
            modified_time_in_range = np.mean((counterfactual_data['predicted_glucose'] >= 70) & (counterfactual_data['predicted_glucose'] <= 180)) * 100
            st.metric(
                "Time in Range",
                f"{modified_time_in_range:.1f}%",
                delta=f"{modified_time_in_range - original_time_in_range:.1f}%"
            )
        
        with col3:
            dose_change = new_dose - original_dose
            st.metric(
                "Insulin Dose Change",
                f"{new_dose:.1f} units",
                delta=f"{dose_change:.1f}"
            )
        
        # Detailed analysis table
        st.markdown("---")
        st.subheader("📊 Detailed Analysis")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Time': day_data.index,
            'Original Glucose': day_data['glucose'],
            'Modified Glucose': counterfactual_data['predicted_glucose'],
            'Glucose Difference': glucose_diff,
            'Original Insulin': day_data['insulin'],
            'Modified Insulin': modified_data['insulin']
        })
        
        # Filter to show only relevant time window around intervention
        window_start = selected_time - pd.Timedelta(hours=2)
        window_end = selected_time + pd.Timedelta(hours=6)
        windowed_df = comparison_df[(comparison_df['Time'] >= window_start) & (comparison_df['Time'] <= window_end)]
        
        st.dataframe(
            windowed_df.style.format({
                'Original Glucose': '{:.1f}',
                'Modified Glucose': '{:.1f}',
                'Glucose Difference': '{:.1f}',
                'Original Insulin': '{:.1f}',
                'Modified Insulin': '{:.1f}'
            }),
            use_container_width=True
        )
        
        # Export options
        st.markdown("---")
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Analysis Data"):
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"counterfactual_analysis_{start_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Generate Report"):
                st.success("Report generation feature coming soon!")
    
    else:
        st.warning("No insulin interventions found for the selected date.")

if __name__ == "__main__":
    main()