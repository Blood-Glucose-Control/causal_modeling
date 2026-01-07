import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os
import json
from plotly.subplots import make_subplots

class HealthDataVisualizer:
    def __init__(self, data_path="output/merged_health_data.csv"):
        print("Loading data...")
        self.df = pd.read_csv(data_path, low_memory=False)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Debug information
        print("\nAvailable columns:")
        print(self.df.columns.tolist())
        print("\nFirst few rows:")
        print(self.df.head())
        
    def plot_bgl_time_series(self):
        """Generate interactive blood glucose time series plot."""
        print("\nPlotting interactive blood glucose time series...")
        
        # Debug: Print data statistics
        print("\nData statistics:")
        for col in sorted(self.df.columns):
            non_null = self.df[col].count()
            if non_null > 0:
                print(f"- {col}: {non_null:,} non-null values")
                if col == 'timestamp':
                    # Analyze timestamp distribution
                    timestamps = pd.to_datetime(self.df['timestamp'].dropna())
                    time_diffs = timestamps.sort_values().diff()
                    print("  Timestamp intervals:")
                    print("  - Min:", time_diffs.min())
                    print("  - Max:", time_diffs.max())
                    print("  - Median:", time_diffs.median())
                    interval_counts = time_diffs.value_counts().sort_index()
                    print("  Most common intervals:")
                    print(interval_counts.head().to_string())
        
        # Create figure with two subplots sharing x-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.6, 0.4],  # BGL plot gets more height
            subplot_titles=('Blood Glucose', 'Health Metrics')
        )
        
        # Add main BGL trace to top subplot
        readings = self.df[self.df['bgl'].notna()].copy()
        # Sort by timestamp to ensure correct line connections
        readings = readings.sort_values('timestamp')
        
        # Add BGL line trace
        fig.add_trace(go.Scatter(
            x=readings['timestamp'],
            y=readings['bgl'],
            mode='lines',
            name='Blood Glucose',
            line=dict(color='blue', width=2),
            hovertemplate='<b>BGL:</b> %{y:.1f} mg/dL<br><b>Time:</b> %{x}',
            legendgroup='glucose',
            legendgrouptitle_text="Glucose"
        ), row=1, col=1)
        
        # Add markers for actual readings to show exact data points
        fig.add_trace(go.Scatter(
            x=readings['timestamp'],
            y=readings['bgl'],
            mode='markers',
            name='BGL Readings',
            marker=dict(color='blue', size=4),
            hovertemplate='<b>BGL:</b> %{y:.1f} mg/dL<br><b>Time:</b> %{x}',
            legendgroup='glucose',
            showlegend=False
        ), row=1, col=1)
        
        # Add BGL events to top subplot
        events = readings[readings['msg_type'].notna()]
        event_colors = {
            'DOSE_INSULIN': 'red',
            'DOSE_BASAL_INSULIN': 'darkred',
            'ANNOUNCE_MEAL': 'green',
            'INTERVENTION_SNACK': 'orange',
            'BGL_FP_READING': 'purple',
            'NEW_PEN': 'brown',
            'NEW_SENSOR': 'gray',
            'ANNOUNCE_EXERCISE': 'blue',
            'MEDICAL_TEST_RESULT': 'black'
        }
        
        # Create descriptions dictionary for each metric and event type
        descriptions = {
            'Blood Glucose': {
                'original_scale': 'mg/dL (70-180 target range)',
                'description': 'Continuous glucose monitor readings showing blood glucose levels.',
                'modifications': 'No scaling applied. Shown in original units.',
                'frequency': 'Every 5 minutes from CGM, or on-demand for finger prick readings.'
            },
            'DOSE_INSULIN': {
                'description': 'Regular insulin doses administered.',
                'modifications': 'Multiple doses at same timestamp are summed.',
                'frequency': 'Event-based (when insulin is administered).'
            },
            'DOSE_BASAL_INSULIN': {
                'description': 'Long-acting basal insulin doses.',
                'modifications': 'Multiple doses at same timestamp are summed.',
                'frequency': 'Typically once or twice daily.'
            },
            'ANNOUNCE_MEAL': {
                'description': 'Meal announcements with carbohydrate amounts.',
                'modifications': 'Multiple food amounts at same timestamp are summed.',
                'frequency': 'Event-based (when meals are consumed).'
            },
            'INTERVENTION_SNACK': {
                'description': 'Fast-acting carbohydrates for low blood sugar treatment.',
                'modifications': 'Multiple food amounts at same timestamp are summed.',
                'frequency': 'Event-based (when treating low blood sugar).'
            },
            'Sleep Score': {
                'original_scale': '0-100',
                'description': 'Overall sleep quality score.',
                'modifications': 'No scaling applied. Original 0-100 scale.',
                'frequency': 'Once daily (calculated after each sleep session).'
            },
            'Deep Sleep': {
                'original_scale': 'minutes (typically 0-500)',
                'description': 'Time spent in deep sleep phase.',
                'modifications': 'No scaling applied. Shown in original minutes.',
                'frequency': 'Once daily (total minutes from sleep session).'
            },
            'REM Sleep': {
                'original_scale': 'minutes (typically 0-500)',
                'description': 'Time spent in REM sleep phase.',
                'modifications': 'No scaling applied. Shown in original minutes.',
                'frequency': 'Once daily (total minutes from sleep session).'
            },
            'Stress Score': {
                'original_scale': '0-100',
                'description': 'Overall stress level based on various metrics.',
                'modifications': 'No scaling applied. Original 0-100 scale.',
                'frequency': 'Updated throughout the day based on HRV and activity.'
            },
            'HRV (RMSSD)': {
                'original_scale': 'milliseconds (typically 0-150)',
                'description': 'Heart Rate Variability - Root Mean Square of Successive Differences. Measured during sleep/rest periods for highest accuracy.',
                'modifications': 'Scaled to 0-100 range (multiplied by 100/150) for visualization. Only high-quality readings (coverage ≥ 80%) are shown.',
                'frequency': 'Every 5 minutes during sleep/rest periods only.'
            },
            'Non-REM Heart Rate': {
                'original_scale': 'beats per minute (typically 40-100)',
                'description': 'Heart rate measured during non-REM sleep phases.',
                'modifications': 'No scaling applied. Original beats per minute.',
                'frequency': 'Once per sleep session.'
            },
            'Readiness Score': {
                'original_scale': '0-100',
                'description': 'Overall readiness score indicating recovery and preparedness for activity.',
                'modifications': 'Scaled to 0-10 range (divided by 10) for better visualization of daily variations.',
                'frequency': 'Updated daily based on sleep, activity, and recovery metrics.'
            }
        }
        
        for msg_type in events['msg_type'].unique():
            if pd.isna(msg_type):
                continue
            
            type_events = events[events['msg_type'] == msg_type]
            color = event_colors.get(msg_type, 'gray')
            
            # Create customdata array based on event type
            if msg_type == 'ANNOUNCE_MEAL':
                customdata = type_events[['trend', 'text', 'food_g', 'food_glycemic_index']].fillna('').values
                hovertemplate = (
                    '<b>BGL:</b> %{y:.1f} mg/dL<br>' +
                    '<b>Time:</b> %{x}<br>' +
                    '<b>Trend:</b> %{customdata[0]}<br>' +
                    '<b>Food:</b> %{customdata[1]}<br>' +
                    '<b>Carbs:</b> %{customdata[2]:.1f}g<br>' +
                    '<b>Glycemic Index:</b> %{customdata[3]:.1f}'
                )
            else:
                customdata = type_events[['trend', 'text']].fillna('').values
                hovertemplate = (
                    '<b>BGL:</b> %{y:.1f} mg/dL<br>' +
                    '<b>Time:</b> %{x}<br>' +
                    '<b>Trend:</b> %{customdata[0]}<br>' +
                    '<b>Note:</b> %{customdata[1]}'
                )
            
            fig.add_trace(go.Scatter(
                x=type_events['timestamp'],
                y=type_events['bgl'],
                mode='markers',
                name=msg_type,
                marker=dict(size=8, color=color),
                customdata=customdata,
                hovertemplate=hovertemplate,
                visible='legendonly',
                legendgroup='glucose',
                legendgrouptitle_text="Glucose"
            ), row=1, col=1)
        
        # Add target range for BGL to top subplot
        fig.add_hrect(
            y0=70, y1=180,
            fillcolor="lightgreen", opacity=0.1,
            layer="below", line_width=0,
            name="Target Range (70-180)",
            legendgroup='glucose',
            row=1, col=1
        )
        
        # Add Sleep Metrics to bottom subplot
        sleep_metrics = {
            'sleep_score_overall_score': {'color': 'purple', 'name': 'Sleep Score', 'scale': 1},
            'sleep_score_deep_sleep_in_minutes': {'color': 'darkblue', 'name': 'Deep Sleep (minutes)', 'scale': 1},
            'sleep_profile_rem_sleep': {'color': 'lightblue', 'name': 'REM Sleep (minutes)', 'scale': 1}
        }
        
        for metric, props in sleep_metrics.items():
            metric_data = self.df[self.df[metric].notna()]
            if not metric_data.empty:
                original_values = metric_data[metric].values
                scaled_values = original_values * props['scale']
                
                is_duration = 'sleep' in metric and 'score' not in metric
                hover_template = (
                    f'<b>{props["name"]}:</b><br>' +
                    ('Duration: %{y:.1f} minutes<br>' if is_duration else 'Score: %{y:.1f}/100<br>') +
                    '<b>Time:</b> %{x}'
                )
                
                fig.add_trace(go.Scatter(
                    x=metric_data['timestamp'],
                    y=scaled_values,
                    mode='lines+markers',
                    name=props['name'],
                    line=dict(color=props['color']),
                    hovertemplate=hover_template,
                    visible='legendonly',
                    legendgroup='sleep',
                    legendgrouptitle_text="Sleep"
                ), row=2, col=1)  # Add to second subplot
        
        # Add Sleep Stability to bottom subplot
        if 'sleep_profile_sleep_stability' in self.df.columns:
            stability_data = self.df[self.df['sleep_profile_sleep_stability'].notna()]
            if not stability_data.empty:
                fig.add_trace(go.Scatter(
                    x=stability_data['timestamp'],
                    y=stability_data['sleep_profile_sleep_stability'],
                    mode='lines+markers',
                    name='Sleep Stability',
                    line=dict(color='darkgreen'),
                    customdata=np.column_stack((
                        stability_data['sleep_profile_sleep_stability'],
                        stability_data['sleep_profile_nights_with_long_awakenings'],
                        stability_data['sleep_profile_days_with_naps']
                    )),
                    hovertemplate=(
                        '<b>Sleep Stability:</b><br>' +
                        'Wake Events: %{y:.1f} per hour<br>' +
                        'Long Awakenings: %{customdata[1]:.1f}%<br>' +
                        'Days with Naps: %{customdata[2]:.1f}%<br>' +
                        '<b>Time:</b> %{x}'
                    ),
                    visible='legendonly',
                    legendgroup='sleep',
                    legendgrouptitle_text="Sleep"
                ), row=2, col=1)
        
        # Add SPO2 data to bottom subplot
        if 'spo2_value' in self.df.columns:
            spo2_data = self.df[self.df['spo2_value'].notna()]
            if not spo2_data.empty:
                fig.add_trace(go.Scatter(
                    x=spo2_data['timestamp'],
                    y=spo2_data['spo2_value'],
                    mode='markers',
                    name='SpO2',
                    marker=dict(size=4, color='cyan'),
                    customdata=np.column_stack((
                        spo2_data['spo2_value'],
                        spo2_data['spo2_spo2_daily_average_value'].fillna(''),
                        spo2_data['spo2_spo2_daily_lower_bound'].fillna(''),
                        spo2_data['spo2_spo2_daily_upper_bound'].fillna('')
                    )),
                    hovertemplate=(
                        '<b>SpO2:</b> %{y:.1f}%<br>' +
                        '<b>Daily Average:</b> %{customdata[1]:.1f}%<br>' +
                        '<b>Range:</b> %{customdata[2]:.1f}% - %{customdata[3]:.1f}%<br>' +
                        '<b>Time:</b> %{x}'
                    ),
                    visible='legendonly',
                    legendgroup='vitals',
                    legendgrouptitle_text="Vital Signs"
                ), row=2, col=1)
        
        # Update Stress Score to include more details
        stress_data = self.df[self.df['stress_score_STRESS_SCORE'].notna()]
        if not stress_data.empty:
            fig.add_trace(go.Scatter(
                x=stress_data['timestamp'],
                y=stress_data['stress_score_STRESS_SCORE'],
                mode='lines+markers',
                name='Stress Score',
                line=dict(color='red'),
                customdata=np.column_stack((
                    stress_data['stress_score_STRESS_SCORE'],
                    stress_data['stress_score_SLEEP_POINTS'].fillna(0),
                    stress_data['stress_score_RESPONSIVENESS_POINTS'].fillna(0),
                    stress_data['stress_score_EXERTION_POINTS'].fillna(0),
                    stress_data['stress_score_STATUS'].fillna('')
                )),
                hovertemplate=(
                    '<b>Stress Score:</b> %{customdata[0]:.1f}/100<br>' +
                    '<b>Components:</b><br>' +
                    'Sleep: %{customdata[1]:.1f}<br>' +
                    'Responsiveness: %{customdata[2]:.1f}<br>' +
                    'Exertion: %{customdata[3]:.1f}<br>' +
                    '<b>Status:</b> %{customdata[4]}<br>' +
                    '<b>Time:</b> %{x}'
                ),
                visible='legendonly',
                legendgroup='stress',
                legendgrouptitle_text="Stress"
            ), row=2, col=1)
        
        # Update Readiness Score to include state information
        readiness_data = self.df[self.df['daily_readiness_daily_readiness_readiness_score_value'].notna()]
        if not readiness_data.empty:
            customdata = np.column_stack((
                readiness_data['daily_readiness_daily_readiness_readiness_score_value'],
                readiness_data['daily_readiness_daily_readiness_readiness_score_value'] / 10,
                readiness_data['daily_readiness_daily_readiness_readiness_state'].fillna(''),
                readiness_data['daily_readiness_daily_readiness_activity_subcomponent'].fillna(0),
                readiness_data['daily_readiness_daily_readiness_sleep_subcomponent'].fillna(0),
                readiness_data['daily_readiness_daily_readiness_hrv_subcomponent'].fillna(0),
                readiness_data['daily_readiness_activity_state'].fillna(''),
                readiness_data['daily_readiness_sleep_state'].fillna(''),
                readiness_data['daily_readiness_hrv_state'].fillna('')
            ))
            
            fig.add_trace(go.Scatter(
                x=readiness_data['timestamp'],
                y=customdata[:, 1],  # Use scaled values (0-10)
                mode='lines+markers',
                name='Readiness Score',
                line=dict(color='orange'),
                customdata=customdata,
                hovertemplate=(
                    '<b>Readiness Score:</b><br>' +
                    'Score: %{customdata[0]:.1f}/100 (%{customdata[1]:.1f}/10)<br>' +
                    'State: %{customdata[2]}<br>' +
                    '<b>Components:</b><br>' +
                    'Activity: %{customdata[3]:.1f} (%{customdata[6]})<br>' +
                    'Sleep: %{customdata[4]:.1f} (%{customdata[7]})<br>' +
                    'HRV: %{customdata[5]:.1f} (%{customdata[8]})<br>' +
                    '<b>Time:</b> %{x}'
                ),
                visible=True,
                legendgroup='readiness',
                legendgrouptitle_text="Readiness"
            ), row=2, col=1)
        
        # Add HRV (RMSSD) data
        if 'hrv_details_rmssd' in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df['timestamp'],
                y=self.df['hrv_details_rmssd'] * (100/150),  # Scale to 0-100
                mode='markers',
                name='HRV (RMSSD)',
                marker=dict(size=6, color='purple'),
                legendgroup='hrv',
                legendgrouptitle_text="Heart Metrics"
            ), row=2, col=1)  # Add to second subplot

        # Add Non-REM Heart Rate data
        if 'hrv_summary_nremhr' in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df['timestamp'],
                y=self.df['hrv_summary_nremhr'],
                mode='markers',
                name='Non-REM Heart Rate',
                marker=dict(size=6, color='darkred'),
                legendgroup='hrv'
            ), row=2, col=1)  # Add to second subplot
        
        # Update layout
        fig.update_layout(
            height=1000,  # Increased height for better visualization
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor="rgba(255, 255, 255, 0.8)",
                groupclick="toggleitem"
            ),
            margin=dict(r=150, t=60, b=100),  # Added bottom margin for range selector
            # Bottom subplot x-axis (with range selector)
            xaxis2=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=12, label="12h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=3, label="3d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    y=1.1,  # Position above the plot
                    x=0,  # Align to left
                    font=dict(size=12),
                    bgcolor='rgba(150, 200, 250, 0.4)',
                    activecolor='rgba(150, 200, 250, 0.8)'
                )
            ),
            # Top subplot x-axis (no controls, just shared zoom)
            xaxis=dict(
                showticklabels=False  # Hide x-axis labels for top subplot
            ),
            # Add buttons for showing/hiding all traces
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1.05,
                    y=1.1,
                    showactive=True,
                    buttons=[
                        dict(
                            label="Show All",
                            method="update",
                            args=[{"visible": True}]
                        ),
                        dict(
                            label="Hide All",
                            method="update",
                            args=[{"visible": False}]
                        )
                    ]
                )
            ]
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Blood Glucose (mg/dL)", row=1, col=1)
        fig.update_yaxes(title_text="Scores / Duration (minutes)", row=2, col=1)
        
        # Create description text with grid styling
        description_html = """
        <div style='margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;'>
            <h2 style='margin: 0 0 15px 0; color: #2c3e50; font-size: 1.2em;'>📊 Active Metrics</h2>
            <div id='metric-descriptions' style='display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px;'>
                <p><i>Select metrics in the legend above to view their details here.</i></p>
            </div>
        </div>
        """
        
        # Add JavaScript to update descriptions based on selected traces
        description_script = """
        <script>
            const descriptions = %s;
            const gd = document.getElementById('health-metrics');
            let lastUpdate = 0;
            
            function formatDescription(traceName, desc) {
                return `
                    <div class="metric-card" style="background: white; padding: 12px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="color: #2c3e50; font-weight: 600; font-size: 1.1em; margin-bottom: 8px;">
                            ${traceName}
                        </div>
                        ${desc.original_scale ? 
                            `<div style="font-size: 0.9em; color: #666; margin-bottom: 6px;">
                                <span style="color: #2c3e50; font-weight: 500;">Scale:</span> ${desc.original_scale}
                            </div>` : ''
                        }
                        <div style="font-size: 0.9em; color: #666; margin-bottom: 6px;">
                            <span style="color: #2c3e50; font-weight: 500;">Description:</span> ${desc.description}
                        </div>
                        <div style="font-size: 0.9em; color: #666; margin-bottom: 6px;">
                            <span style="color: #2c3e50; font-weight: 500;">Frequency:</span> ${desc.frequency}
                        </div>
                        <div style="font-size: 0.9em; color: #666;">
                            <span style="color: #2c3e50; font-weight: 500;">Processing:</span> ${desc.modifications}
                        </div>
                    </div>
                `.trim();
            }
            
            function updateDescriptions() {
                const now = Date.now();
                if (now - lastUpdate < 100) return; // Debounce updates
                lastUpdate = now;
                
                const activeTraces = gd.data
                    .filter(trace => trace.visible === true)
                    .map(trace => trace.name)
                    .filter(name => descriptions[name]);
                
                const descriptionDiv = document.getElementById('metric-descriptions');
                if (activeTraces.length === 0) {
                    descriptionDiv.innerHTML = '<p><i>Select metrics in the legend above to view their details here.</i></p>';
                    return;
                }
                
                const cards = activeTraces
                    .map(name => formatDescription(name, descriptions[name]))
                    .join('');
                
                descriptionDiv.innerHTML = cards;
            }
            
            // Initial update
            setTimeout(updateDescriptions, 1000);
            
            // Update on legend clicks
            gd.on('plotly_legendclick', updateDescriptions);
            gd.on('plotly_legenddoubleclick', updateDescriptions);
        </script>
        """ % json.dumps(descriptions)
        
        # Save plot with descriptions
        os.makedirs('docs', exist_ok=True)
        html_content = fig.to_html(
            include_plotlyjs=True,
            full_html=True,
            div_id='health-metrics'
        )
        # Insert description section and script before </body>
        html_content = html_content.replace('</body>', description_html + description_script + '</body>')
        
        with open('docs/index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("Interactive plot saved to docs/index.html")

        # Add sleep periods as shaded regions
        if 'sleep_profile_sleep_start_time' in self.df.columns and 'sleep_profile_sleep_duration' in self.df.columns:
            sleep_data = self.df[self.df['sleep_profile_sleep_start_time'].notna()]
            for idx, row in sleep_data.iterrows():
                start_time = pd.to_datetime(row['sleep_profile_sleep_start_time'])
                duration_mins = row['sleep_profile_sleep_duration']
                end_time = start_time + pd.Timedelta(minutes=duration_mins)
                
                # Add shaded region for sleep period
                fig.add_vrect(
                    x0=start_time,
                    x1=end_time,
                    fillcolor="gray",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    name="Sleep Period",
                    legendgroup="activities",
                    legendgrouptitle_text="Activities"
                )

        # Add workout periods based on stress score exertion points
        if 'stress_score_EXERTION_POINTS' in self.df.columns:
            workout_data = self.df[self.df['stress_score_EXERTION_POINTS'] > 80]  # High exertion indicates workout
            for idx, row in workout_data.iterrows():
                # Add markers for workout periods
                fig.add_vline(
                    x=row['timestamp'],
                    line_width=2,
                    line_color="orange",
                    opacity=0.5,
                    name="High Activity",
                    legendgroup="activities"
                )

        # Add annotations for activity status
        if 'stress_score_STATUS' in self.df.columns:
            status_data = self.df[self.df['stress_score_STATUS'].notna()]
            for status in status_data['stress_score_STATUS'].unique():
                status_times = status_data[status_data['stress_score_STATUS'] == status]
                fig.add_trace(go.Scatter(
                    x=status_times['timestamp'],
                    y=[0] * len(status_times),  # Place at bottom of plot
                    mode='markers',
                    name=f"Status: {status}",
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color=dict(
                            REST='blue',
                            ACTIVITY='orange',
                            STRESS='red'
                        ).get(status, 'gray')
                    ),
                    legendgroup="activities",
                    showlegend=True
                ), row=1, col=1)

    def generate_all_plots(self):
        """Generate all visualization plots."""
        os.makedirs('docs', exist_ok=True)
        
        # Generate interactive time series plot
        self.plot_bgl_time_series()
        
        print("\nAll visualizations have been generated in the 'docs' directory.")

def main():
    visualizer = HealthDataVisualizer()
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main() 