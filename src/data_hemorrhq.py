import pandas as pd
import os
from datetime import datetime, timezone

def load_data():
    """Load the Xente dataset"""
    data_path = "data/raw/xente_dataset.csv"
    
    if not os.path.exists(data_path):
        print("❌ Dataset not found!")
        print(f"Expected path: {os.path.abspath(data_path)}")
        raise FileNotFoundError("xente_dataset.csv is missing")
    
    print("📂 Loading dataset...")
    # Robust loading
    df = pd.read_csv(data_path, on_bad_lines='skip', encoding='utf-8')
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    return df

def create_rfm_features(df):
    """Create RFM features with timezone handling"""
    df = df.copy()
    
    # Convert to datetime and remove timezone
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True).dt.tz_convert(None)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': 'max',     # Recency
        'TransactionId': 'count',          # Frequency
        'Amount': ['sum', 'mean']          # Monetary
    }).reset_index()
    
    rfm.columns = ['CustomerId', 'LastTransaction', 'Frequency', 'TotalAmount', 'AvgAmount']
    
    # Calculate Recency using timezone-naive datetime
    current_time = datetime.now().replace(tzinfo=None)
    rfm['Recency'] = (current_time - rfm['LastTransaction']).dt.days
    
    print(f"✅ RFM features created for {len(rfm)} customers")
    return rfm