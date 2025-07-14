#!/usr/bin/env python3
"""
Simple Glucose & Counterfactual Visualizer

Interactive tool to visualize blood glucose data with insulin interventions
and explore counterfactual scenarios.

Usage:
1. View glucose timeline with interventions
2. Select an intervention point
3. Choose counterfactual scenario (dose change)
4. Compare factual vs counterfactual glucose evolution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button, Slider
import sys
from datetime import datetime, timedelta

# Add the diabetes-data-api to path
sys.path.append('diabetes-data-api')
from main import DiabetesAnalyzer

class GlucoseVisualizer:
    """
    Interactive visualizer for glucose data and counterfactuals.
    """
    
    def __init__(self):
        self.analyzer = DiabetesAnalyzer(seed=42)
        self.patient_data = None
        self.interventions = None
        self.selected_intervention = None
        self.counterfactual_data = None
        
        # GUI elements
        self.fig = None
        self.ax_glucose = None
        self.ax_insulin = None
        
    def generate_data(self, days=7):
        """Generate patient data for visualization."""
        print(f"Generating {days} days of patient data...")
        
        self.patient_data = self.analyzer.generate_patient_data(
            n_days=days, 
            start_date='2024-01-01'
        )
        
        self.interventions = self.analyzer.counterfactual_model.list_interventions(self.patient_data)
        
        print(f"✓ Generated {len(self.patient_data)} data points")
        print(f"✓ Found {len(self.interventions)} insulin interventions")
        
    def setup_plot(self):
        """Setup the interactive plot."""
        
        # Create figure with subplots
        self.fig, (self.ax_glucose, self.ax_insulin) = plt.subplots(
            2, 1, figsize=(14, 10), sharex=True, 
            gridspec_kw={'height_ratios': [3, 1]}
        )
        
        self.fig.suptitle('Interactive Glucose & Counterfactual Visualizer', 
                         fontsize=16, fontweight='bold')
        
        # Enable interactive navigation (zoom, pan)
        self.fig.canvas.toolbar_visible = True
        
        # Plot initial data
        self.plot_baseline_data()
        
        # Add control buttons
        self.add_controls()
        
        # Setup event handlers
        self.setup_events()
        
    def plot_baseline_data(self):
        """Plot the baseline glucose and insulin data."""
        
        # Clear previous plots
        self.ax_glucose.clear()
        self.ax_insulin.clear()
        
        times = self.patient_data.index
        
        # Plot glucose
        self.ax_glucose.plot(times, self.patient_data['glucose'], 
                           'b-', linewidth=2, label='Blood Glucose', alpha=0.8)
        
        # Plot insulin interventions as points (not lines)
        intervention_times = []
        intervention_doses = []
        intervention_glucose = []
        
        for intervention in self.interventions:
            time = intervention['timestamp']
            dose = intervention['dose']
            
            # Find glucose value at intervention time
            time_idx = abs(times - time).argmin()
            glucose_val = self.patient_data['glucose'].iloc[time_idx]
            
            intervention_times.append(time)
            intervention_doses.append(dose)
            intervention_glucose.append(glucose_val)
        
        # Plot intervention points
        self.ax_glucose.scatter(intervention_times, intervention_glucose, 
                              c='red', s=100, marker='o', alpha=0.8, 
                              label='Insulin Interventions', zorder=5)
        
        # Add dose labels next to points
        for time, dose, glucose in zip(intervention_times, intervention_doses, intervention_glucose):
            self.ax_glucose.annotate(f'{dose:.1f}u', 
                                   xy=(time, glucose), 
                                   xytext=(8, 8), textcoords='offset points',
                                   fontsize=9, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        
        # Plot insulin doses
        insulin_mask = self.patient_data['insulin'] > 0
        insulin_times = self.patient_data.index[insulin_mask]
        insulin_doses = self.patient_data['insulin'][insulin_mask]
        
        self.ax_insulin.stem(insulin_times, insulin_doses, 
                           linefmt='red', markerfmt='ro', basefmt=' ')
        
        # Plot meals as background
        meal_mask = self.patient_data['carbs'] > 0
        meal_times = self.patient_data.index[meal_mask]
        meal_carbs = self.patient_data['carbs'][meal_mask]
        
        self.ax_insulin.bar(meal_times, meal_carbs/10, width=timedelta(minutes=30), 
                          alpha=0.3, color='brown', label='Meals (carbs/10)')
        
        # Formatting
        self.ax_glucose.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
        self.ax_glucose.legend(loc='upper right')
        self.ax_glucose.grid(True, alpha=0.3)
        self.ax_glucose.set_ylim(40, 250)
        
        self.ax_insulin.set_ylabel('Insulin (units)', fontsize=12)
        self.ax_insulin.set_xlabel('Time', fontsize=12)
        self.ax_insulin.legend(loc='upper right')
        self.ax_insulin.grid(True, alpha=0.3)
        
        # Format time axis
        self.ax_insulin.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        self.ax_insulin.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(self.ax_insulin.xaxis.get_majorticklabels(), rotation=45)
        
        # Add intervention selection info
        self.ax_glucose.text(0.02, 0.98, 'Click on a red point to select intervention. Use toolbar to zoom/pan.', 
                           transform=self.ax_glucose.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
    def add_controls(self):
        """Add control buttons and sliders."""
        
        # Add dose factor slider
        ax_slider = plt.axes([0.2, 0.02, 0.5, 0.03])
        self.dose_slider = Slider(ax_slider, 'Dose Factor', 0.5, 2.0, valinit=1.0, 
                                 valfmt='%.1fx')
        self.dose_slider.on_changed(self.update_counterfactual)
        
        # Add control buttons
        ax_reset = plt.axes([0.8, 0.02, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_plot)
        
        ax_generate = plt.axes([0.88, 0.02, 0.1, 0.04])
        self.btn_generate = Button(ax_generate, 'New Data')
        self.btn_generate.on_clicked(self.generate_new_data)
        
    def setup_events(self):
        """Setup mouse click events."""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        """Handle mouse clicks to select interventions."""
        
        if event.inaxes != self.ax_glucose:
            return
            
        if event.button != 1:  # Only left clicks
            return
            
        click_time = mdates.num2date(event.xdata)
        
        # Ensure click_time is timezone-naive for comparison
        if click_time.tzinfo is not None:
            click_time = click_time.replace(tzinfo=None)
        
        # Find closest intervention
        closest_intervention = None
        min_distance = float('inf')
        
        for intervention in self.interventions:
            intervention_time = intervention['timestamp']
            
            # Ensure intervention timestamp is timezone-naive for comparison
            if hasattr(intervention_time, 'tzinfo') and intervention_time.tzinfo is not None:
                intervention_time = intervention_time.replace(tzinfo=None)
            
            distance = abs((intervention_time - click_time).total_seconds())
            if distance < min_distance:
                min_distance = distance
                closest_intervention = intervention
        
        # Only select if click is within 1 hour of intervention
        if min_distance < 3600:  # 1 hour in seconds
            self.selected_intervention = closest_intervention
            print(f"Selected intervention: {closest_intervention['timestamp']} - {closest_intervention['dose']:.2f} units")
            self.update_counterfactual(self.dose_slider.val)
            
    def update_counterfactual(self, dose_factor):
        """Update counterfactual visualization based on selected intervention and dose factor."""
        
        if self.selected_intervention is None:
            return
            
        # Generate counterfactual
        try:
            cf_data = self.analyzer.analyze_intervention(
                self.patient_data,
                intervention_id=self.selected_intervention['id'],
                analysis_type='dose',
                dose_factor=dose_factor,
                before_minutes=120,
                after_minutes=360
            )
            
            self.counterfactual_data = cf_data
            self.plot_with_counterfactual(dose_factor)
            
        except Exception as e:
            print(f"Error generating counterfactual: {e}")
            
    def plot_with_counterfactual(self, dose_factor):
        """Plot baseline data with counterfactual overlay."""
        
        # Replot baseline
        self.plot_baseline_data()
        
        if self.counterfactual_data is None:
            return
            
        # Get counterfactual metadata
        cf_meta = list(self.counterfactual_data.attrs.values())[-1]
        cf_col = f"cf{cf_meta['cf_number']}_glucose"
        
        # Plot counterfactual glucose
        times = self.counterfactual_data.index
        cf_glucose = self.counterfactual_data[cf_col]
        
        self.ax_glucose.plot(times, cf_glucose, 'g--', linewidth=3, 
                           label=f'Counterfactual ({dose_factor:.1f}x dose)', alpha=0.8)
        
        # Highlight the intervention window
        window_start = cf_meta['window_start']
        window_end = cf_meta['window_end']
        
        self.ax_glucose.axvspan(window_start, window_end, alpha=0.1, color='gray', 
                              label='Analysis Window')
        
        # Highlight selected intervention
        intervention_time = self.selected_intervention['timestamp']
        original_dose = self.selected_intervention['dose']
        new_dose = original_dose * dose_factor
        
        self.ax_glucose.axvline(intervention_time, color='red', linewidth=3, 
                              label=f'Selected: {original_dose:.1f}u → {new_dose:.1f}u')
        
        # Add difference annotation
        intervention_idx = abs(times - intervention_time).argmin()
        
        # Show glucose difference at key time points
        for minutes_after in [60, 120, 180]:
            target_time = intervention_time + timedelta(minutes=minutes_after)
            target_idx = abs(times - target_time).argmin()
            
            if target_idx < len(cf_glucose):
                baseline_val = self.counterfactual_data['glucose'].iloc[target_idx]
                cf_val = cf_glucose.iloc[target_idx]
                diff = cf_val - baseline_val
                
                self.ax_glucose.annotate(f'{diff:+.0f}', 
                                       xy=(target_time, cf_val),
                                       xytext=(10, 10), textcoords='offset points',
                                       bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
                                       fontsize=9, fontweight='bold')
        
        # Update legend
        self.ax_glucose.legend(loc='upper right')
        
        # Update title with scenario info
        self.fig.suptitle(f'Glucose Visualization - {dose_factor:.1f}x Dose Scenario '
                         f'({original_dose:.1f}u → {new_dose:.1f}u)', 
                         fontsize=16, fontweight='bold')
        
        plt.draw()
        
    def reset_plot(self, event):
        """Reset to baseline view."""
        self.selected_intervention = None
        self.counterfactual_data = None
        self.dose_slider.reset()
        self.plot_baseline_data()
        self.fig.suptitle('Interactive Glucose & Counterfactual Visualizer', 
                         fontsize=16, fontweight='bold')
        plt.draw()
        
    def generate_new_data(self, event):
        """Generate new patient data."""
        self.generate_data(days=7)
        self.reset_plot(event)
        
    def show(self):
        """Display the interactive plot."""
        plt.show()

def main():
    """Run the glucose visualizer."""
    
    print("Interactive Glucose & Counterfactual Visualizer")
    print("=" * 50)
    print("Instructions:")
    print("1. View blood glucose timeline with insulin interventions (red points)")
    print("2. Use toolbar to zoom/pan for detailed view of time windows")
    print("3. Click on any red intervention point to select it")
    print("4. Use the dose factor slider to explore counterfactual scenarios")
    print("5. Green dashed line shows counterfactual glucose evolution")
    print("6. Numbers show glucose difference at key time points")
    print("7. Use 'Reset' to clear selection, 'New Data' to generate new patient")
    print()
    
    # Create and run visualizer
    viz = GlucoseVisualizer()
    viz.generate_data(days=7)
    viz.setup_plot()
    viz.show()

if __name__ == "__main__":
    main()