📊 Credit Risk Model (Bati Bank BNPL)
🚀 Overview

This project builds a Credit Risk Prediction System for Bati Bank’s Buy-Now-Pay-Later (BNPL) service using the Xente transaction dataset.

It covers the full ML lifecycle:

Data preprocessing & feature engineering (RFM)
Proxy target creation (since default labels are not available)
Model training (Gradient Boosting)
Experiment tracking (MLflow)
Model deployment via FastAPI
CI/CD ready project structure

The goal is to classify customers into Low, Medium, and High credit risk to support lending decisions.

🧠 Business Understanding (Credit Scoring Context)
1. Basel II Compliance

Financial institutions must use:

Interpretable models (e.g., Logistic Regression, WoE)
Transparent decision-making
Auditable risk scoring systems

This project balances:

✔ Interpretability
✔ Predictive performance
2. Proxy Target Problem

The dataset has no explicit default label, so we construct a proxy:

High risk = customers with:
High Recency (inactive users)
Low Frequency (low engagement)

⚠️ Risk: Proxy labels may introduce noise and bias, affecting model accuracy.

3. Model Trade-offs
Model	Pros	Cons
Logistic Regression	Highly interpretable	May underfit complex patterns
Gradient Boosting (Used)	High accuracy	Less interpretable

We selected Gradient Boosting Classifier for better predictive performance.

📁 Project Structure
credit-risk-model/
│
├── README.md
├── final_report.pdf
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
│
├── notebooks/
│   └── 10-eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_hemorrhq.py
│   ├── train.py
│   └── predict.py
│
├── api/
│   ├── main.py
│   └── pydantic_models.py
│
├── tests/
│   └── test_data_hemorrhq.py
│
└── .github/
    └── workflows/
        └── ci.yml
⚙️ Setup Instructions
1. Clone Repository
git clone https://github.com/Nehmyabiruk/credit-risk-model.git
cd credit-risk-model
2. Install Dependencies
pip install -r requirements.txt
3. Train Model
python src/train.py

This will:

Train Gradient Boosting model
Log metrics with MLflow
Save model to:
models/credit_model.pkl
4. Run API (FastAPI)
uvicorn api.main:app --reload
5. Run with Docker (Optional)
docker-compose up
📡 API Usage
Endpoint
POST /predict
Request Body
{
  "CustomerId": "CUST001",
  "Recency": 15,
  "Frequency": 10,
  "TotalAmount": 5000,
  "AvgAmount": 500
}
Response
{
  "CustomerId": "CUST001",
  "RiskProbability": 0.23,
  "RiskLevel": "Low Risk",
  "Recommendation": "Approve"
}
📊 Model Performance
Accuracy: ~0.85+
F1 Score: Balanced for imbalanced classification
ROC-AUC: Strong ranking performance
🧪 Testing

Run unit tests:

pytest tests/
🔄 CI/CD Pipeline

GitHub Actions workflow:

Runs tests automatically
Validates code structure
Ensures reproducibility
🧰 Tech Stack
Python 🐍
Pandas / NumPy
Scikit-learn
MLflow
FastAPI
Docker
GitHub Actions
📌 Key Features

✔ RFM Feature Engineering
✔ Proxy Label Creation
✔ MLflow Experiment Tracking
✔ REST API with FastAPI
✔ Dockerized Deployment
✔ CI/CD Pipeline

👨‍💻 Author

Nehmya Biruk
Computer Science Student | AI/ML Engineer
GitHub: Nehmyabiruk

📜 License

This project is for educational purposes (10 Academy Week Project).
