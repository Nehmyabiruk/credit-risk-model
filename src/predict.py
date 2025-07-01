# Inference script
import pandas as pd
import mlflow.sklearn
from data_hemorrhq import load_data, create_rfm_features, feature_pipeline

def predict(data_path):
    # Load and preprocess data
    df = load_data(data_path)
    rfm = create_rfm_features(df)
    X = rfm.drop(['is_high_risk', 'TransactionStartTime'], axis=1, errors='ignore')
    
    # Load model from MLflow
    model = mlflow.sklearn.load_model('models:/GradientBoosting/1')
    
    # Apply feature pipeline
    pipeline = feature_pipeline()
    X_transformed = pipeline.fit_transform(X)
    
    # Predict
    predictions = model.predict_proba(X_transformed)[:, 1]  # Probability of high-risk
    return predictions
