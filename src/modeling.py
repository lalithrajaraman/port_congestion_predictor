import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

def train_and_evaluate(data_path, model_dir):
    print("Starting nationwide model training pipeline...")
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows of processed features across all ports.")
    
    # Handle missing values
    df['Cargo'] = df['Cargo'].fillna(-1)
    df['VesselType'] = df['VesselType'].fillna(-1)
    
    # Extract temporal features
    df['hour'] = pd.to_datetime(df['BaseDateTime']).dt.hour
    
    # Feature columns (incorporating nearest_port_id)
    feature_cols = [
        'LAT', 'LON', 'SOG', 'COG', 'Heading', 
        'VesselType', 'Length', 'Width', 'Draft', 'Cargo', 
        'port_active_density', 'port_stationary_density', 'port_total_density', 
        'dist_to_port', 'size_area', 'draft_ratio', 'nearest_port_id', 'hour'
    ]
    target_col = 'dwell_time'
    
    # Split by MMSI to prevent data leakage of the same ship's trajectory points
    unique_mmsis = df['MMSI'].unique()
    train_mmsis, test_mmsis = train_test_split(unique_mmsis, test_size=0.2, random_state=42)
    
    train_mask = df['MMSI'].isin(train_mmsis)
    test_mask = df['MMSI'].isin(test_mmsis)
    
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]
    
    print(f"Training set: {len(X_train)} rows ({len(train_mmsis)} vessels)")
    print(f"Testing set: {len(X_test)} rows ({len(test_mmsis)} vessels)")
    
    # Define models
    models = {
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=150, 
            learning_rate=0.06, 
            max_depth=6,
            num_leaves=31,
            min_child_samples=30,
            reg_alpha=2.0,
            reg_lambda=2.0,
            random_state=42,
            verbose=-1
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=80, 
            max_depth=10, 
            min_samples_split=8,
            random_state=42, 
            n_jobs=-1
        )
    }
    
    best_rmse = float('inf')
    best_model_name = None
    best_model = None
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Categorical feature specification for LightGBM
        if name == "LightGBM":
            model.fit(X_train, y_train, categorical_feature=['nearest_port_id'])
        else:
            model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        print(f"{name} Evaluation:")
        print(f"  RMSE: {rmse:.4f} hours")
        print(f"  MAE:  {mae:.4f} hours")
        print(f"  R2:   {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} with RMSE: {best_rmse:.4f} hours")
    
    # Save model
    model_path = os.path.join(model_dir, "best_model.pkl")
    metadata_path = os.path.join(model_dir, "model_metadata.pkl")
    
    joblib.dump(best_model, model_path)
    
    metadata = {
        "feature_names": feature_cols,
        "best_model_name": best_model_name,
        "results": results,
        "feature_importances": dict(zip(feature_cols, best_model.feature_importances_)) if hasattr(best_model, 'feature_importances_') else None
    }
    joblib.dump(metadata, metadata_path)
    print(f"Best model saved to {model_path}")
    print(f"Model metadata saved to {metadata_path}")
    
    if metadata["feature_importances"]:
        print("\nFeature Importances:")
        sorted_fi = sorted(metadata["feature_importances"].items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_fi:
            print(f"  {feat}: {imp:.4f}")

if __name__ == "__main__":
    default_root = Path(__file__).resolve().parent.parent
    data_csv = default_root / "data" / "processed_features.csv"
    m_dir = default_root / "models"
    train_and_evaluate(str(data_csv), str(m_dir))
