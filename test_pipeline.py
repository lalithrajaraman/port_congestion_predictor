import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def run_tests():
    print("==================================================")
    print("   PORT CONGESTION ML PIPELINE VERIFICATION TEST  ")
    print("==================================================")
    
    # Check 1: Feature CSV existence and shape
    project_root = Path(__file__).resolve().parent
    feature_csv = project_root / "data" / "processed_features.csv"
    assert feature_csv.exists(), f"FAIL: processed_features.csv does not exist at {feature_csv}."
    print("PASS: processed_features.csv exists.")
    
    df = pd.read_csv(feature_csv)
    assert len(df) > 0, "FAIL: processed_features.csv is empty."
    print(f"PASS: Dataset loaded successfully with {len(df)} rows.")
    
    # Check 2: Check expected column names
    expected_cols = [
        'LAT', 'LON', 'SOG', 'COG', 'Heading', 
        'VesselType', 'Length', 'Width', 'Draft', 'Cargo', 
        'port_active_density', 'port_stationary_density', 'port_total_density', 
        'dist_to_port', 'size_area', 'draft_ratio', 'nearest_port_id', 'dwell_time'
    ]
    for col in expected_cols:
        assert col in df.columns, f"FAIL: Expected column '{col}' missing from data."
    print("PASS: All engineered feature columns are present.")
    
    # Check 3: Check model and metadata existence
    model_path = project_root / "models" / "best_model.pkl"
    meta_path = project_root / "models" / "model_metadata.pkl"
    assert model_path.exists(), f"FAIL: best_model.pkl does not exist at {model_path}."
    assert meta_path.exists(), f"FAIL: model_metadata.pkl does not exist at {meta_path}."
    print("PASS: Trained model and metadata pickle files exist.")
    
    # Check 4: Load model and perform mock inference
    model = joblib.load(str(model_path))
    meta = joblib.load(str(meta_path))
    print(f"PASS: Successfully loaded best model ({meta['best_model_name']}).")
    
    # Make a mock dataframe matching feature_names in metadata
    feature_names = meta['feature_names']
    print(f"Model features: {feature_names}")
    
    mock_input = pd.DataFrame([{
        'LAT': 29.6,
        'LON': -94.9,
        'SOG': 5.0,
        'COG': 180.0,
        'Heading': 180.0,
        'VesselType': 1024.0,
        'Length': 150.0,
        'Width': 20.0,
        'Draft': 8.0,
        'Cargo': 70.0,
        'port_active_density': 40.0,
        'port_stationary_density': 80.0,
        'port_total_density': 120.0,
        'dist_to_port': 15.0,
        'size_area': 3000.0,
        'draft_ratio': 8.0 / 150.0,
        'nearest_port_id': 0, # Houston
        'hour': 12
    }])
    
    # Reorder/align columns
    mock_input = mock_input[feature_names]
    
    prediction = model.predict(mock_input)[0]
    print(f"Mock Input Prediction Result: {prediction:.4f} hours")
    assert isinstance(prediction, (float, np.float64, np.float32)), "FAIL: Prediction is not a numeric type."
    assert prediction >= 0 or np.isnan(prediction), f"FAIL: Predicted dwell time ({prediction}) should be positive."
    print("PASS: Mock inference test completed successfully.")
    
    print("\n==================================================")
    print("        ALL TESTS COMPLETED SUCCESSFULLY!         ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
