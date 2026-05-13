from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Credit Risk Model API",
    version="1.0"
)

# -----------------------------
# Load Trained Model
# -----------------------------
model = None

try:
    model_path = "models/credit_model.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully!")
    else:
        print("Model file not found. Please train the model first.")

except Exception as e:
    print(f"Error loading model: {e}")

# -----------------------------
# Request Schema
# -----------------------------
class RiskRequest(BaseModel):
    CustomerId: str
    Frequency: int
    TotalAmount: float
    AvgAmount: float
    Recency: int

# -----------------------------
# Response Schema
# -----------------------------
class RiskResponse(BaseModel):
    CustomerId: str
    RiskProbability: float
    RiskLevel: str
    Recommendation: str

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
async def root():
    return {
        "status": "Credit Risk Model API is running"
    }

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict", response_model=RiskResponse)
async def predict_risk(request: RiskRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run python src/train.py first."
        )

    # Create dataframe for prediction
    input_df = pd.DataFrame([{
        "Recency": request.Recency,
        "Frequency": request.Frequency,
        "TotalAmount": request.TotalAmount,
        "AvgAmount": request.AvgAmount
    }])

    # Predict probability
    probability = float(model.predict_proba(input_df)[0][1])

    # Risk logic
    if probability > 0.6:
        risk_level = "High Risk"
        recommendation = "Reject / High Caution"

    elif probability > 0.35:
        risk_level = "Medium Risk"
        recommendation = "Approve with Caution"

    else:
        risk_level = "Low Risk"
        recommendation = "Approve"

    return RiskResponse(
        CustomerId=request.CustomerId,
        RiskProbability=round(probability, 4),
        RiskLevel=risk_level,
        Recommendation=recommendation
    )

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )