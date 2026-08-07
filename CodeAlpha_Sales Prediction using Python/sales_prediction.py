"""
Task 3: Sales Prediction using Python
CodeAlpha Data Science Internship
-----------------------------------
This script loads Advertising.csv, cleans un-named index columns,
performs correlation matrix analysis to identify key sales revenue drivers,
splits data into 80/20 train/test sets, builds Multi-Variable Linear Regression
and Random Forest Regressor models, evaluates R², MAE, RMSE, and saves output visualizations into outputs/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Set visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_clean_data():
    """Load Advertising dataset, remove index column & duplicates, normalize headers."""
    path = os.path.join(BASE_DIR, 'Advertising.csv')
    df = pd.read_csv(path)
    
    # Drop unnamed index column if present
    unnamed_cols = [c for c in df.columns if 'Unnamed' in c or 'unnamed' in c]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.drop_duplicates().copy()
    
    return df

def analyze_correlations(df):
    """Correlation matrix analysis to identify driving advertising channel."""
    corr = df.corr()
    print("=" * 70)
    print("TASK 3: SALES PREDICTION - CORRELATION ANALYSIS WITH SALES")
    print("=" * 70)
    sales_corr = corr['sales'].sort_values(ascending=False)
    print(sales_corr)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='viridis', fmt='.3f', linewidths=0.8, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix: Advertising Spend vs Sales Revenue', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'sales_correlation_heatmap.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")

def train_and_evaluate_models(df):
    """Split into train/test, fit Linear Regression & Random Forest, and report metrics."""
    X = df[['tv', 'radio', 'newspaper']]
    y = df['sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Multi-Variable Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION METRICS (80/20 Train-Test Split)")
    print("=" * 70)
    
    best_r2 = -float('inf')
    best_name = None
    best_model = None
    best_y_pred = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        print(f"\nModel: {name}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE:      {mae:.4f}")
        print(f"  RMSE:     {rmse:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model
            best_y_pred = preds
            
    # Save best model binary
    model_pkl_path = os.path.join(OUTPUT_DIR, 'sales_prediction_model.pkl')
    joblib.dump(best_model, model_pkl_path)
    print(f"\nSaved best model ({best_name}) binary to: {model_pkl_path}")
    
    return best_name, best_model, X_train, y_test, best_y_pred

def generate_visualizations(best_name, best_model, X_train, y_test, best_y_pred):
    """Generate feature importance and actual vs predicted sales plots."""
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(9, 5))
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
    else:
        importances = pd.Series(np.abs(best_model.coef_), index=X_train.columns).sort_values(ascending=True)
        
    importances.plot(kind='barh', color='#5cb85c')
    plt.title(f'Feature Importance / Channel Impact ({best_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Impact Score', fontsize=12)
    plt.ylabel('Advertising Channel', fontsize=12)
    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, 'feature_importance_sales.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")
    
    # 2. Actual vs Predicted Sales Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_y_pred, alpha=0.85, color='#0275d8', edgecolors='k', s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal 1:1 Line')
    plt.title(f'Actual vs Predicted Sales Revenue ({best_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Sales', fontsize=12)
    plt.ylabel('Predicted Sales', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, 'actual_vs_predicted_sales.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

def main():
    df = load_and_clean_data()
    analyze_correlations(df)
    best_name, best_model, X_tr, y_te, best_y_pred = train_and_evaluate_models(df)
    generate_visualizations(best_name, best_model, X_tr, y_te, best_y_pred)
    print("\nTask 3 execution completed successfully!\n")

if __name__ == '__main__':
    main()
