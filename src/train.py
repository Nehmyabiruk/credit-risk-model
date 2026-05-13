import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Fix encoding for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_hemorrhq import load_data, create_rfm_features


def train_model():
    """Main training function"""

    mlflow.set_experiment("Credit_Risk_Xente")

    print("Loading and preparing data...")

    df = load_data()
    rfm = create_rfm_features(df)

    # Target variable
    rfm["is_high_risk"] = (
        (rfm["Recency"] > 45) & (rfm["Frequency"] < 8)
    ).astype(int)

    # Features
    features = ["Recency", "Frequency", "TotalAmount", "AvgAmount"]

    X = rfm[features]
    y = rfm["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print(f"Training shape: {X_train.shape} | Test shape: {X_test.shape}")

    with mlflow.start_run(run_name="GradientBoosting"):

        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Save to MLflow
        mlflow.sklearn.log_model(model, "model")

        # -----------------------------
        # ALSO SAVE LOCAL MODEL (IMPORTANT FIX)
        # -----------------------------
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/credit_model.pkl")

        print("\nModel training completed successfully!")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC AUC   : {roc_auc:.4f}")
        print("Model saved to models/credit_model.pkl")

    return model


if __name__ == "__main__":
    train_model()