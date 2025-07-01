# Feature engineering script
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    return pd.read_csv(file_path)

def create_rfm_features(df):
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': 'max',  # Recency
        'TransactionId': 'count',       # Frequency
        'Amount': 'sum'                 # Monetary
    }).reset_index()
    rfm['TransactionStartTime'] = pd.to_datetime(rfm['TransactionStartTime'])
    rfm['Recency'] = (pd.Timestamp.now() - rfm['TransactionStartTime']).dt.days
    return rfm

def create_temporal_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Hour'] = df['TransactionStartTime'].dt.hour
    df['Day'] = df['TransactionStartTime'].dt.day
    df['Month'] = df['TransactionStartTime'].dt.month
    return df

def feature_pipeline():
    numerical_features = ['Amount', 'Value']
    categorical_features = ['ProductCategory', 'ChannelId']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    return pipeline
