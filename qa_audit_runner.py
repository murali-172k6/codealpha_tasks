import os
import sys
import joblib
import pandas as pd
import numpy as np

print("=" * 80)
print("PRINCIPAL QA LEAD & SENIOR DATA AUDITOR - SUITE VERIFICATION")
print("=" * 80)

def audit_task1():
    print("\n--- AUDITING TASK 1: Unemployment Analysis (unemployment/) ---")
    script_path = os.path.join("unemployment", "unemployment_analysis.py")
    ret = os.system(f'python "{script_path}"')
    if ret != 0:
        raise RuntimeError(f"Task 1 execution failed with exit code {ret}")
    
    out_dir = os.path.join("unemployment", "outputs")
    outputs = os.listdir(out_dir) if os.path.exists(out_dir) else []
    expected_files = [
        "unemployment_time_series_trends.png",
        "rural_vs_urban_unemployment.png",
        "top_affected_states.png",
        "correlation_heatmap.png"
    ]
    for ef in expected_files:
        assert ef in outputs, f"Missing output file: {ef}"
        fsize = os.path.getsize(os.path.join(out_dir, ef))
        assert fsize > 1000, f"File {ef} appears corrupted or empty (size: {fsize} bytes)"
        print(f"  [PASS] Verified artifact: {ef} ({fsize / 1024:.1f} KB)")
        
    print("  [SUCCESS] Task 1 Unemployment Analysis Audit PASSED 100%")

def audit_task2():
    print("\n--- AUDITING TASK 2: Car Price Prediction (car prediction/) ---")
    script_path = os.path.join("car prediction", "car_price_prediction.py")
    ret = os.system(f'python "{script_path}"')
    if ret != 0:
        raise RuntimeError(f"Task 2 execution failed with exit code {ret}")
        
    out_dir = os.path.join("car prediction", "outputs")
    outputs = os.listdir(out_dir) if os.path.exists(out_dir) else []
    expected_files = ["car_price_model.pkl", "feature_importance.png", "actual_vs_predicted_car_prices.png"]
    for ef in expected_files:
        assert ef in outputs, f"Missing output file: {ef}"
        fsize = os.path.getsize(os.path.join(out_dir, ef))
        print(f"  [PASS] Verified artifact: {ef} ({fsize / 1024:.1f} KB)")
        
    # Verify model deserialization
    model_path = os.path.join(out_dir, "car_price_model.pkl")
    model = joblib.load(model_path)
    print(f"  [PASS] Successfully deserialized model: {type(model).__name__}")
    
    # Test dummy inference on model
    # Model expects 8 features: present_price, driven_kms, owner, car_age, fuel_type_Diesel, fuel_type_Petrol, selling_type_Individual, transmission_Manual
    dummy_input = pd.DataFrame([{
        'present_price': 5.5,
        'driven_kms': 25000,
        'owner': 0,
        'car_age': 7,
        'fuel_type_Diesel': 0,
        'fuel_type_Petrol': 1,
        'selling_type_Individual': 0,
        'transmission_Manual': 1
    }])
    pred = model.predict(dummy_input)
    print(f"  [PASS] Model test inference returned predicted price: {pred[0]:.2f} Lakhs")
    print("  [SUCCESS] Task 2 Car Price Prediction Audit PASSED 100%")

def audit_task3():
    print("\n--- AUDITING TASK 3: Sales Prediction (Sales Prediction/) ---")
    script_path = os.path.join("Sales Prediction", "sales_prediction.py")
    ret = os.system(f'python "{script_path}"')
    if ret != 0:
        raise RuntimeError(f"Task 3 execution failed with exit code {ret}")
        
    out_dir = os.path.join("Sales Prediction", "outputs")
    outputs = os.listdir(out_dir) if os.path.exists(out_dir) else []
    expected_files = [
        "sales_prediction_model.pkl",
        "sales_correlation_heatmap.png",
        "feature_importance_sales.png",
        "actual_vs_predicted_sales.png"
    ]
    for ef in expected_files:
        assert ef in outputs, f"Missing output file: {ef}"
        fsize = os.path.getsize(os.path.join(out_dir, ef))
        print(f"  [PASS] Verified artifact: {ef} ({fsize / 1024:.1f} KB)")
        
    # Verify model deserialization
    model_path = os.path.join(out_dir, "sales_prediction_model.pkl")
    model = joblib.load(model_path)
    print(f"  [PASS] Successfully deserialized model: {type(model).__name__}")
    
    # Dummy inference test: tv, radio, newspaper
    dummy_input = pd.DataFrame([{'tv': 200.0, 'radio': 40.0, 'newspaper': 20.0}])
    pred = model.predict(dummy_input)
    print(f"  [PASS] Model test inference returned predicted sales: {pred[0]:.2f} units")
    print("  [SUCCESS] Task 3 Sales Prediction Audit PASSED 100%")

def main():
    audit_task1()
    audit_task2()
    audit_task3()
    print("\n" + "=" * 80)
    print("ALL 3 TASKS PASSED COMPREHENSIVE QA AUDIT & VERIFICATION WITH 0 ERRORS")
    print("=" * 80)

if __name__ == '__main__':
    main()
