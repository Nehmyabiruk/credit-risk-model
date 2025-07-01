# Model training script
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_hemorrhq import load_data, create_rfm_features, feature_pipeline

# Load and preprocess data
df = load_data('data/raw/xente_dataset.csv')
rfm = create_rfm_features(df)
X = rfm.drop(['is_high_risk', 'TransactionStartTime'], axis=1, errors='ignore')
y = rfm['is_high_risk'] if 'is_high_risk' in rfm else pd.Series(0, index=rfm.index)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature pipeline
pipeline = feature_pipeline()

# Model training
models = {
    'LogisticRegression': LogisticRegression(),
    'GradientBoosting': GradientBoostingClassifier()
}

# Hyperparameter tuning
param_grids = {
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'GradientBoosting': {'n_estimators': [100, 200], 'max_depth': [3, 5]}
}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        grid = GridSearchCV(model, param_grids[model_name], cv=5, scoring='f1')
        grid.fit(X_train, y_train)
        
        # Log parameters and metrics
        mlflow.log_params(grid.best_params_)
        y_pred = grid.predict(X_test)
        mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
        mlflow.log_metric('precision', precision_score(y_test, y_pred))
        mlflow.log_metric('recall', recall_score(y_test, y_pred))
        mlflow.log_metric('f1', f1_score(y_test, y_pred))
        mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_pred))
        
        # Register model
        if model_name == 'GradientBoosting':
            mlflow.sklearn.log_model(grid.best_estimator_, 'model')
