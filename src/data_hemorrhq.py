# Feature engineering script
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def create_rfm_features(df):
    # Placeholder for RFM feature engineering
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': 'max',  # Recency
        'TransactionId': 'count',       # Frequency
        'Amount': 'sum'                 # Monetary
    }).reset_index()
    return rfm

# Example pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler())
])
