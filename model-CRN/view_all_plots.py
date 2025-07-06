#!/usr/bin/env python3

"""
Display information about all available plots and visualizations.
"""

import os
from pathlib import Path

def list_all_visualizations():
    """List all available plots and visualizations"""
    
    print("📊 AVAILABLE PLOTS AND VISUALIZATIONS")
    print("=" * 60)
    
    # 1. CRN Test Results Plots (just created)
    print("\n🎯 CRN MODEL TEST RESULTS:")
    print("-" * 30)
    test_results_dir = Path("test_results")
    if test_results_dir.exists():
        plot_files = list(test_results_dir.glob("*.png")) + list(test_results_dir.glob("*.pdf"))
        for plot_file in sorted(plot_files):
            print(f"  📈 {plot_file}")
            if "comparison" in str(plot_file):
                print("      → 6-panel comparison: RMSE, MAE, Range Accuracy, Win rates")
            elif "heatmap" in str(plot_file):
                print("      → Detailed heatmaps: Performance by scenario")
    
    # 2. Synthetic Data Visualizations
    print("\n🧬 SYNTHETIC GLUCOSE DATA VISUALIZATIONS:")
    print("-" * 40)
    viz_dir = Path("../synthetic_data/visualizations")
    if viz_dir.exists():
        
        # Main timeline plots
        main_plots = ["full_timeline.html", "sample_period.html"]
        for plot in main_plots:
            plot_path = viz_dir / plot
            if plot_path.exists():
                print(f"  🩸 {plot_path}")
                if "full" in plot:
                    print("      → 90-day synthetic glucose timeline with meals, insulin, exercise")
                elif "sample" in plot:
                    print("      → 3-day detailed view for analysis")
        
        # Dose counterfactuals
        dose_dir = viz_dir / "dose_counterfactuals"
        if dose_dir.exists():
            print(f"\n  💉 DOSE COUNTERFACTUAL ANALYSIS:")
            dose_plots = list(dose_dir.glob("*.html"))
            for plot in sorted(dose_plots):
                print(f"      📊 {plot}")
                if "comparison" in str(plot):
                    print("          → Compare different insulin dose scenarios")
                elif "sample" in str(plot):
                    print("          → 24-hour sample with dose variations")
        
        # Timing counterfactuals  
        timing_dir = viz_dir / "timing_counterfactuals"
        if timing_dir.exists():
            print(f"\n  ⏰ TIMING COUNTERFACTUAL ANALYSIS:")
            timing_plots = list(timing_dir.glob("*.html"))
            for plot in sorted(timing_plots):
                print(f"      📊 {plot}")
                if "comparison" in str(plot):
                    print("          → Compare different insulin timing scenarios")
                elif "sample" in str(plot):
                    print("          → 24-hour sample with timing variations")
    
    # 3. Data Files
    print("\n💾 DATA FILES:")
    print("-" * 15)
    
    # Test results data
    test_json = Path("test_results/crn_test_results.json")
    if test_json.exists():
        print(f"  📋 {test_json}")
        print("      → Detailed CRN test results in JSON format")
    
    # Synthetic data files
    data_dir = Path("../synthetic_data/data")
    if data_dir.exists():
        data_files = ["ml_dataset.csv", "full_dataset.csv"]
        for data_file in data_files:
            data_path = data_dir / data_file
            if data_path.exists():
                print(f"  📊 {data_path}")
                if "ml_dataset" in data_file:
                    print("      → Processed dataset for ML training (25,920 time points)")
                elif "full_dataset" in data_file:
                    print("      → Raw synthetic glucose data with all features")
    
    print("\n" + "=" * 60)
    print("💡 HOW TO VIEW PLOTS:")
    print("   • PNG/PDF files: Open directly in file viewer")
    print("   • HTML files: Open in web browser for interactive plots")
    print("   • JSON files: Contains raw numerical results")
    
    print("\n🔗 QUICK ACCESS COMMANDS:")
    print("   # View CRN comparison plots")
    print("   open test_results/crn_comparison_plots.png")
    print("   ")
    print("   # View synthetic data timeline")
    print("   open ../synthetic_data/visualizations/full_timeline.html")
    print("   ")
    print("   # View dose counterfactuals")
    print("   open ../synthetic_data/visualizations/dose_counterfactuals/")

def main():
    list_all_visualizations()

if __name__ == '__main__':
    main()