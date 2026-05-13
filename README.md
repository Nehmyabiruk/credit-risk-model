📊 Credit Risk Model (Bati Bank BNPL)
🚀 Overview

This project builds a Credit Risk Prediction System for Bati Bank’s Buy-Now-Pay-Later (BNPL) service using the Xente dataset.

It covers the full ML lifecycle:

Data preprocessing & feature engineering (RFM)
Proxy target creation
Model training (Gradient Boosting)
MLflow tracking
FastAPI deployment
Docker + CI/CD ready structure

🧠 Business Understanding
1. Basel II Compliance

Financial institutions require:

Interpretable models
Transparent decision-making
Auditable systems
2. Proxy Target Problem

No default label exists, so we define:

High risk = high Recency + low Frequency

⚠️ This introduces some labeling noise.

3. Model Choice
Model	Pros	Cons
Logistic Regression	Interpretable	Lower accuracy
Gradient Boosting	High performance	Less interpretable

We use Gradient Boosting Classifier.

📁 Project Structure
credit-risk-model/
│
├── notebooks/
│   └── 10-eda.ipynb
│
├── src/
│   ├── data_hemorrhq.py
│   ├── train.py
│   └── predict.py
│
├── api/
│   └── main.py
│
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

⚙️ Setup

1. Clone repo
git clone https://github.com/Nehmyabiruk/credit-risk-model.git
cd credit-risk-model

3. Install dependencies
pip install -r requirements.txt

5. Train model
python src/train.py

7. Run API
 python api/main.py
   
📡 API Usage
Endpoint
POST /predict
Request
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
Accuracy: 85%+
F1 Score: Balanced
ROC-AUC: Strong ranking power

🧰 Tech Stack
Python
Scikit-learn
Pandas
MLflow
FastAPI
Docker
GitHub Actions

👨‍💻 Author

Nehmya Biruk
GitHub: https://github.com/Nehmyabiruk
