# Unit tests for feature engineering
import pytest
import pandas as pd
from src.data_hemorrhq import create_rfm_features

def test_rfm_features():
    # Sample data
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionId': ['T1', 'T2', 'T3'],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Amount': [100, 200, 150]
    })
    
    # Create RFM features
    rfm = create_rfm_features(data)
    
    # Assertions
    assert len(rfm) == 2, "RFM should have 2 customers"
    assert rfm.loc[rfm['CustomerId'] == 'C1', 'TransactionId'].iloc[0] == 2, "C1 should have 2 transactions"
    assert rfm.loc[rfm['CustomerId'] == 'C1', 'Amount'].iloc[0] == 300, "C1 monetary value should be 300"
