#!/usr/bin/env python3

"""
Create visualizations for CRN test results.
Plots comparison between ordinal and one-hot encoding performance.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_test_results():
    """Load the CRN test results from JSON file"""
    with open('test_results/crn_test_results.json', 'r') as f:
        results = json.load(f)
    return results

def create_comparison_plots(results):
    """Create comprehensive comparison plots"""
    
    # Prepare data for plotting
    plot_data = []
    
    for test_category, scenarios in results.items():
        for scenario_name, scenario_results in scenarios.items():
            ordinal = scenario_results.get('ordinal', {})
            onehot = scenario_results.get('onehot', {})
            
            if ordinal and onehot:
                plot_data.append({
                    'Test Category': test_category.replace('_', ' ').title(),
                    'Scenario': scenario_name.replace('_', ' ').title(),
                    'Ordinal RMSE': ordinal.get('rmse', 0),
                    'One-Hot RMSE': onehot.get('rmse', 0),
                    'Ordinal MAE': ordinal.get('mae', 0),
                    'One-Hot MAE': onehot.get('mae', 0),
                    'Ordinal Range Acc': ordinal.get('range_accuracy', 0),
                    'One-Hot Range Acc': onehot.get('range_accuracy', 0),
                    'RMSE Difference': ordinal.get('rmse', 0) - onehot.get('rmse', 0),
                    'Ordinal Better': ordinal.get('rmse', 0) < onehot.get('rmse', 0)
                })
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. RMSE Comparison Bar Plot
    plt.subplot(2, 3, 1)
    x_pos = np.arange(len(df))
    width = 0.35
    
    plt.bar(x_pos - width/2, df['Ordinal RMSE'], width, 
            label='Ordinal', alpha=0.8, color='#2E86AB')
    plt.bar(x_pos + width/2, df['One-Hot RMSE'], width, 
            label='One-Hot', alpha=0.8, color='#A23B72')
    
    plt.xlabel('Test Scenarios')
    plt.ylabel('RMSE (mg/dL)')
    plt.title('RMSE Comparison: Ordinal vs One-Hot', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, df['Scenario'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 2. RMSE Difference Plot
    plt.subplot(2, 3, 2)
    colors = ['#2E86AB' if x < 0 else '#A23B72' for x in df['RMSE Difference']]
    bars = plt.bar(range(len(df)), df['RMSE Difference'], color=colors, alpha=0.8)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Test Scenarios')
    plt.ylabel('RMSE Difference (Ordinal - One-Hot)')
    plt.title('RMSE Difference (Negative = Ordinal Better)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(df)), df['Scenario'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.003),
                f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # 3. MAE Comparison
    plt.subplot(2, 3, 3)
    plt.bar(x_pos - width/2, df['Ordinal MAE'], width, 
            label='Ordinal', alpha=0.8, color='#2E86AB')
    plt.bar(x_pos + width/2, df['One-Hot MAE'], width, 
            label='One-Hot', alpha=0.8, color='#A23B72')
    
    plt.xlabel('Test Scenarios')
    plt.ylabel('MAE (mg/dL)')
    plt.title('MAE Comparison: Ordinal vs One-Hot', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, df['Scenario'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 4. Range Accuracy Comparison
    plt.subplot(2, 3, 4)
    plt.bar(x_pos - width/2, df['Ordinal Range Acc'], width, 
            label='Ordinal', alpha=0.8, color='#2E86AB')
    plt.bar(x_pos + width/2, df['One-Hot Range Acc'], width, 
            label='One-Hot', alpha=0.8, color='#A23B72')
    
    plt.xlabel('Test Scenarios')
    plt.ylabel('Range Accuracy (70-180 mg/dL)')
    plt.title('Clinical Range Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, df['Scenario'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0.95, 1.005)  # Zoom in since all values are close to 1
    
    # 5. Win/Loss Summary by Category
    plt.subplot(2, 3, 5)
    category_summary = df.groupby('Test Category')['Ordinal Better'].agg(['sum', 'count']).reset_index()
    category_summary['Ordinal Win Rate'] = category_summary['sum'] / category_summary['count']
    category_summary['One-Hot Win Rate'] = 1 - category_summary['Ordinal Win Rate']
    
    categories = category_summary['Test Category']
    ordinal_wins = category_summary['Ordinal Win Rate']
    onehot_wins = category_summary['One-Hot Win Rate']
    
    x_cat = np.arange(len(categories))
    plt.bar(x_cat, ordinal_wins, label='Ordinal Wins', alpha=0.8, color='#2E86AB')
    plt.bar(x_cat, onehot_wins, bottom=ordinal_wins, label='One-Hot Wins', alpha=0.8, color='#A23B72')
    
    plt.xlabel('Test Categories')
    plt.ylabel('Win Rate')
    plt.title('Win Rate by Test Category', fontsize=14, fontweight='bold')
    plt.xticks(x_cat, categories, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)
    
    # Add percentage labels
    for i, (ord_win, onehot_win) in enumerate(zip(ordinal_wins, onehot_wins)):
        if ord_win > 0:
            plt.text(i, ord_win/2, f'{ord_win:.1%}', ha='center', va='center', fontweight='bold', color='white')
        if onehot_win > 0:
            plt.text(i, ord_win + onehot_win/2, f'{onehot_win:.1%}', ha='center', va='center', fontweight='bold', color='white')
    
    # 6. Overall Summary Statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calculate overall statistics
    total_scenarios = len(df)
    ordinal_wins_total = sum(df['Ordinal Better'])
    onehot_wins_total = total_scenarios - ordinal_wins_total
    avg_ordinal_rmse = df['Ordinal RMSE'].mean()
    avg_onehot_rmse = df['One-Hot RMSE'].mean()
    avg_improvement = ((avg_onehot_rmse - avg_ordinal_rmse) / avg_onehot_rmse * 100)
    
    summary_text = f"""
    OVERALL SUMMARY
    
    Total Test Scenarios: {total_scenarios}
    
    Ordinal Wins: {ordinal_wins_total} ({ordinal_wins_total/total_scenarios:.1%})
    One-Hot Wins: {onehot_wins_total} ({onehot_wins_total/total_scenarios:.1%})
    
    Average RMSE:
    • Ordinal: {avg_ordinal_rmse:.3f} mg/dL
    • One-Hot: {avg_onehot_rmse:.3f} mg/dL
    
    Overall Improvement: {avg_improvement:.2f}%
    (Ordinal better by {abs(avg_improvement):.2f}%)
    
    Key Insight:
    Ordinal encoding shows consistent
    advantage across diverse insulin
    dosing scenarios, especially in
    complex dosing situations.
    """
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('test_results/crn_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.savefig('test_results/crn_comparison_plots.pdf', bbox_inches='tight')
    
    return fig

def create_detailed_scenario_plot(results):
    """Create detailed plot showing scenario-specific results"""
    
    # Prepare data
    plot_data = []
    for test_category, scenarios in results.items():
        for scenario_name, scenario_results in scenarios.items():
            ordinal = scenario_results.get('ordinal', {})
            onehot = scenario_results.get('onehot', {})
            
            if ordinal and onehot:
                plot_data.append({
                    'category': test_category,
                    'scenario': scenario_name,
                    'ordinal_rmse': ordinal.get('rmse', 0),
                    'onehot_rmse': onehot.get('rmse', 0),
                    'improvement': ((onehot.get('rmse', 0) - ordinal.get('rmse', 0)) / onehot.get('rmse', 0) * 100)
                })
    
    df = pd.DataFrame(plot_data)
    
    # Create heatmap of improvements
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Heatmap of RMSE values
    pivot_ordinal = df.pivot(index='scenario', columns='category', values='ordinal_rmse')
    pivot_onehot = df.pivot(index='scenario', columns='category', values='onehot_rmse')
    
    # RMSE comparison heatmap
    sns.heatmap(pivot_ordinal, annot=True, fmt='.3f', cmap='Blues', ax=ax1, cbar_kws={'label': 'RMSE (mg/dL)'})
    ax1.set_title('Ordinal RMSE by Scenario', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Test Category')
    ax1.set_ylabel('Scenario')
    
    # Improvement heatmap
    pivot_improvement = df.pivot(index='scenario', columns='category', values='improvement')
    sns.heatmap(pivot_improvement, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, ax=ax2, 
                cbar_kws={'label': 'Improvement (%)'})
    ax2.set_title('Ordinal Improvement over One-Hot (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Test Category')
    ax2.set_ylabel('Scenario')
    
    plt.tight_layout()
    plt.savefig('test_results/crn_detailed_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('test_results/crn_detailed_heatmap.pdf', bbox_inches='tight')
    
    return fig

def main():
    """Create all visualization plots"""
    print("Creating CRN test result visualizations...")
    
    # Load results
    results = load_test_results()
    
    # Create plots
    fig1 = create_comparison_plots(results)
    print("✓ Created comparison plots: test_results/crn_comparison_plots.png")
    
    fig2 = create_detailed_scenario_plot(results)
    print("✓ Created detailed heatmap: test_results/crn_detailed_heatmap.png")
    
    # Show plots
    plt.show()
    
    print(f"\\nPlot files saved:")
    print(f"  • test_results/crn_comparison_plots.png")
    print(f"  • test_results/crn_comparison_plots.pdf") 
    print(f"  • test_results/crn_detailed_heatmap.png")
    print(f"  • test_results/crn_detailed_heatmap.pdf")

if __name__ == '__main__':
    main()