# Credit Risk Model

## Overview
This project develops a credit risk model for Bati Bank’s buy-now-pay-later service using the Xente dataset. It includes EDA, feature engineering, proxy target creation, model training, and deployment with CI/CD.

## Credit Scoring Business Understanding
1. Basel II Accord: Requires interpretable models (e.g., Logistic Regression with WoE) for regulatory compliance, ensuring transparency and auditability.
2. Proxy Variable: Necessary due to missing “default” label; risks include misclassification leading to financial losses or regulatory penalties.
3. Model Trade-offs: Logistic Regression is interpretable but may underfit; Gradient Boosting offers higher accuracy but less interpretability.

## Project Structure

credit-risk-model/ ├── README.md ├── final_report.pdf ├── notebooks/ │   ├── 10-eda.ipynb ├── src/ │   ├── init.py │   ├── data_hemorrhq.py │   ├── train.py │   ├── predict.py ├── api/ │   ├── main.py │   ├── pydantic_models.py ├── tests/ │   ├── test_data_hemorrhq.py ├── github/workflows/ │   ├── ci.yml ├── Dockerfile ├── docker-compose.yml ├── .gitignore ├── requirements.txt
## Setup

1. Clone the repository.
2. Install dependencies: pip install -r requirements.txt.
3. Run the API: docker-compose up.

## Usage
- Run EDA: notebooks/10-eda.ipynb.
- Train model: python src/train.py.
- Make predictions: python src/predict.py.
- Access API: http://localhost:8000/predict.
