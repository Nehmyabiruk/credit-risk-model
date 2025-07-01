# Credit Risk Model

## Overview
This project develops a credit risk model for Bati Bank’s buy-now-pay-later service using the Xente dataset. It includes EDA, feature engineering, proxy target creation, model training, and deployment with CI/CD.

## Credit Scoring Business Understanding
1. Basel II Accord: Requires interpretable models (e.g., Logistic Regression with WoE) for regulatory compliance, ensuring transparency and auditability.
2. Proxy Variable: Necessary due to missing “default” label; risks include misclassification leading to financial losses or regulatory penalties.
3. Model Trade-offs: Logistic Regression is interpretable but may underfit; Gradient Boosting offers higher accuracy but less interpretability.

## Project Structure
