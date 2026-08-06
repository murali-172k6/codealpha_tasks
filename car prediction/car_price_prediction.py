"""
Task 2: Car Price Prediction with Machine Learning
CodeAlpha Data Science Internship
-----------------------------------
This script loads car price data, performs feature engineering (Car_Age),
encodes categorical variables, trains Linear Regression, Random Forest, and Gradient Boosting models,
evaluates R², MAE, RMSE, exports the best model binary to outputs/car_price_model.pkl,
and generates performance plots.
"""

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Set visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess_data():
    """Load car data CSV, normalize headers, compute car_age, and one-hot encode categorical features."""
    csv_path = os.path.join(BASE_DIR, 'car data.csv')
    df = pd.read_csv(csv_path)
    
    # 1. Header normalization
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # 2. Feature Engineering: Car_Age
    current_year = datetime.datetime.now().year
    df['car_age'] = current_year - df['year']
    
    # Drop car_name and year (since car_age represents year information)
    feature_df = df.drop(columns=['car_name', 'year'])
    
    # 3. Categorical encoding (One-Hot Encoding)
    categorical_cols = ['fuel_type', 'selling_type', 'transmission']
    encoded_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=True)
    
    X = encoded_df.drop(columns=['selling_price'])
    y = encoded_df['selling_price']
    
    return encoded_df, X, y

def train_and_evaluate_models(X, y):
    """Split data into 80/20 train/test sets, train regression models, and evaluate performance."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model_obj = None
    best_y_pred = None
    
    print("=" * 70)
    print("TASK 2: CAR PRICE PREDICTION - MODEL PERFORMANCE EVALUATION")
    print("=" * 70)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
        print(f"\nModel: {name}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE:      {mae:.4f} Lakhs")
        print(f"  RMSE:     {rmse:.4f} Lakhs")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_obj = model
            best_y_pred = y_pred

    print("\n" + "=" * 70)
    print(f"BEST PERFORMING MODEL: {best_model_name} (R² = {best_r2:.4f})")
    print("=" * 70)
    
    # Save best model binary
    model_pkl_path = os.path.join(OUTPUT_DIR, 'car_price_model.pkl')
    joblib.dump(best_model_obj, model_pkl_path)
    print(f"Saved best model binary to: {model_pkl_path}")
    
    return best_model_name, best_model_obj, X_train, X_test, y_train, y_test, best_y_pred

def generate_visualizations(model_name, model_obj, X_train, y_test, y_pred):
    """Generate and save feature importance plot and actual vs predicted prices visualization."""
    
    # 1. Feature Importance plot (for tree-based models)
    if hasattr(model_obj, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = pd.Series(model_obj.feature_importances_, index=X_train.columns).sort_values(ascending=True)
        importances.plot(kind='barh', color='#0275d8')
        plt.title(f'Feature Importance ({model_name})', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.tight_layout()
        plot1_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(plot1_path, dpi=300)
        plt.close()
        print(f"Saved: {plot1_path}")
        
    # 2. Actual vs Predicted Prices Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.8, color='#d9534f', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal 1:1 Line')
    plt.title(f'Actual vs Predicted Car Selling Price ({model_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Selling Price (in Lakhs)', fontsize=12)
    plt.ylabel('Predicted Selling Price (in Lakhs)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, 'actual_vs_predicted_car_prices.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

def main():
    encoded_df, X, y = load_and_preprocess_data()
    best_name, best_model, X_tr, X_te, y_tr, y_te, y_pred = train_and_evaluate_models(X, y)
    generate_visualizations(best_name, best_model, X_tr, y_te, y_pred)
    print("\nTask 2 execution completed successfully!\n")

if __name__ == '__main__':
    main()
