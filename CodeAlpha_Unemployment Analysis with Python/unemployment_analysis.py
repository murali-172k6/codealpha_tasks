"""
Task 1: Unemployment Analysis in India
CodeAlpha Data Science Internship
-----------------------------------
This script loads, cleans, normalizes, and analyzes unemployment datasets in India.
It produces detailed summary statistics and exports high-quality visualizations into outputs/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual aesthetic configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette('crest')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean_data():
    """Load both unemployment datasets, normalize headers, clean dates and null values."""
    path1 = os.path.join(BASE_DIR, 'Unemployment in India.csv')
    path2 = os.path.join(BASE_DIR, 'Unemployment_Rate_upto_11_2020.csv')
    
    # Load dataset 1
    df1 = pd.read_csv(path1)
    df1.columns = df1.columns.str.strip().str.lower().str.replace(' ', '_')
    df1 = df1.dropna(subset=['region']).copy()
    df1['date'] = pd.to_datetime(df1['date'].str.strip(), format='%d-%m-%Y')
    df1['area'] = df1['area'].str.strip()
    df1['region'] = df1['region'].str.strip()
    
    # Load dataset 2
    df2 = pd.read_csv(path2)
    df2.columns = df2.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'region.1' in df2.columns:
        df2.rename(columns={'region.1': 'zone'}, inplace=True)
    df2['date'] = pd.to_datetime(df2['date'].str.strip(), format='%d-%m-%Y')
    df2['region'] = df2['region'].str.strip()
    
    return df1, df2

def print_eda_summary(df1, df2):
    """Compute and display summary statistics across regions and area types (Rural vs Urban)."""
    print("=" * 70)
    print("TASK 1: UNEMPLOYMENT ANALYSIS - SUMMARY STATISTICS")
    print("=" * 70)
    
    print("\n--- DATASET 1: Summary Statistics (Rural vs Urban) ---")
    area_stats = df1.groupby('area')[['estimated_unemployment_rate_(%)', 'estimated_employed', 'estimated_labour_participation_rate_(%)']].agg(['mean', 'median', 'std'])
    print(area_stats.round(2))
    
    print("\n--- DATASET 2: Top 10 Regions by Average Unemployment Rate (%) ---")
    region_stats = df2.groupby('region')['estimated_unemployment_rate_(%)'].agg(['mean', 'median', 'max']).sort_values(by='mean', ascending=False)
    print(region_stats.head(10).round(2))
    
    print("\n--- COVID-19 Lockdown Period Analysis (March - June 2020) ---")
    lockdown_df = df2[(df2['date'] >= '2020-03-01') & (df2['date'] <= '2020-06-30')]
    pre_lockdown = df2[df2['date'] < '2020-03-01']
    print(f"Pre-Lockdown Mean Unemployment Rate: {pre_lockdown['estimated_unemployment_rate_(%)'].mean():.2f}%")
    print(f"Lockdown Peak Mean Unemployment Rate: {lockdown_df['estimated_unemployment_rate_(%)'].mean():.2f}%")

def generate_visualizations(df1, df2):
    """Generate and save visual analysis plots."""
    
    # 1. Time-Series Trends of Key Metrics (from Dataset 2)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    monthly_trend = df2.groupby('date')[['estimated_unemployment_rate_(%)', 'estimated_employed', 'estimated_labour_participation_rate_(%)']].mean().reset_index()
    
    # Plot Unemployment Rate
    sns.lineplot(ax=axes[0], data=monthly_trend, x='date', y='estimated_unemployment_rate_(%)', marker='o', color='#d9534f', linewidth=2.5)
    axes[0].set_title('Time-Series Trend: National Average Unemployment Rate (%)', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_ylabel('Unemployment Rate (%)', fontsize=12)
    axes[0].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-01'), color='#ff9999', alpha=0.3, label='COVID-19 Lockdown')
    axes[0].legend(loc='upper left')
    
    # Plot Estimated Employed
    sns.lineplot(ax=axes[1], data=monthly_trend, x='date', y='estimated_employed', marker='s', color='#0275d8', linewidth=2.5)
    axes[1].set_title('Time-Series Trend: Estimated Employed Population', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_ylabel('Employed Count', fontsize=12)
    axes[1].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-01'), color='#ff9999', alpha=0.3)
    
    # Plot Labour Participation Rate
    sns.lineplot(ax=axes[2], data=monthly_trend, x='date', y='estimated_labour_participation_rate_(%)', marker='^', color='#5cb85c', linewidth=2.5)
    axes[2].set_title('Time-Series Trend: Labour Participation Rate (%)', fontsize=14, fontweight='bold', pad=10)
    axes[2].set_ylabel('Participation Rate (%)', fontsize=12)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-01'), color='#ff9999', alpha=0.3)
    
    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, 'unemployment_time_series_trends.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")
    
    # 2. Rural vs Urban Comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df1, x='area', y='estimated_unemployment_rate_(%)', hue='area', legend=False, palette=['#5bc0de', '#f0ad4e'])
    plt.title('Unemployment Rate Distribution: Rural vs. Urban Areas', fontsize=14, fontweight='bold')
    plt.xlabel('Area Type', fontsize=12)
    plt.ylabel('Unemployment Rate (%)', fontsize=12)
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, 'rural_vs_urban_unemployment.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")
    
    # 3. Top Affected Indian States (Lockdown Peak vs Overall)
    plt.figure(figsize=(12, 8))
    top_states = df2.groupby('region')['estimated_unemployment_rate_(%)'].mean().sort_values(ascending=False).head(12).reset_index()
    sns.barplot(data=top_states, y='region', x='estimated_unemployment_rate_(%)', hue='region', legend=False, palette='magma')
    plt.title('Top 12 Most Affected Indian States by Average Unemployment Rate (2020)', fontsize=14, fontweight='bold')
    plt.xlabel('Average Unemployment Rate (%)', fontsize=12)
    plt.ylabel('State / Region', fontsize=12)
    plt.tight_layout()
    plot3_path = os.path.join(OUTPUT_DIR, 'top_affected_states.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_cols = df2.select_dtypes(include=[np.number]).columns
    corr = df2[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap - Unemployment Dataset Metrics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot4_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved: {plot4_path}")

def main():
    df1, df2 = load_and_clean_data()
    print_eda_summary(df1, df2)
    generate_visualizations(df1, df2)
    print("\nTask 1 execution completed successfully!\n")

if __name__ == '__main__':
    main()
