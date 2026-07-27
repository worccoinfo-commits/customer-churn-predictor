# Customer Churn Predictor

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-00e676?style=for-the-badge)](LICENSE)

A machine learning project that predicts customer churn and provides actionable business recommendations. Built with the IBM Telco Customer Churn dataset, this project demonstrates end-to-end data science workflow with a focus on business impact.

---

## Business Problem

Customer churn is one of the biggest challenges for subscription-based businesses. Acquiring a new customer costs **5-7x more** than retaining an existing one. This project helps businesses:

- Identify customers at risk of leaving **before** they churn
- Understand **why** customers are leaving (key drivers)
- Take **targeted action** with data-driven retention strategies
- Measure the **ROI** of retention programs

## Key Findings

| Insight | Business Action |
|---------|----------------|
| Month-to-month contracts have **42% churn rate** vs 3% for 2-year | Offer discounts for annual commitments |
| First **6 months** is the highest-risk period | Structured onboarding with milestone check-ins |
| Customers without Online Security churn **2x more** | Bundle security services for free first year |
| Electronic check users churn **3x more** than auto-pay users | $5/month discount for switching to auto-pay |
| Higher monthly charges ($70+) increase churn risk | Loyalty discounts at 12-month milestone |

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8125 | 0.8542 | 0.7912 | 0.8215 | 0.8930 |
| Random Forest | 0.8432 | 0.8721 | 0.8345 | 0.8528 | 0.9214 |
| **XGBoost** | **0.8517** | **0.8834** | **0.8492** | **0.8660** | **0.9352** |

## Business Impact

```
Total customers:            7,043
Current churn rate:         26.5%
At-risk customers/year:     ~1,868
Preventable with program:   1,121 (60% retention rate)
Annual revenue saved:       $874,380
Retention program cost:     $140,860
Net annual impact:          $733,520
ROI:                        520.7%
```

## Project Structure

```
customer-churn-predictor/
├── src/
│   ├── train.py          # Training pipeline (EDA, models, evaluation)
│   └── predict.py        # CLI tool for predictions
├── notebooks/
│   └── churn_analysis.ipynb  # Full EDA with visualizations
├── data/                 # Dataset (auto-downloaded)
├── models/              # Saved models (generated)
├── reports/             # Plots and reports (generated)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Predict churn for a single customer (interactive mode)
python src/predict.py

# Predict from a CSV file
python src/predict.py --csv data/new_customers.csv

# Predict from JSON
python src/predict.py --json '{"tenure": 12, "contract": "Month-to-month", ...}'
```

## Training Pipeline

The training script (`src/train.py`) runs:

1. **Data Loading** — Downloads the IBM Telco dataset
2. **Exploratory Data Analysis** — Statistics, distributions, correlations
3. **Visualization** — 5+ plots saved to `reports/` directory
4. **Preprocessing** — Encoding, scaling, feature engineering
5. **Model Training** — Logistic Regression, Random Forest, XGBoost
6. **Evaluation** — Cross-validation, ROC curves, feature importance
7. **Business Report** — ROI analysis with actionable recommendations

## Prediction CLI

The prediction tool (`src/predict.py`) provides:

- **Interactive mode** — Answer questions about a customer, get instant prediction
- **CSV batch mode** — Predict churn for multiple customers from a file
- **JSON API** — Integrate with other applications
- **Risk levels** — HIGH / MEDIUM / LOW with specific retention recommendations

## Dataset

IBM Telco Customer Churn Dataset — 7,043 customers with 21 features including demographics, account information, and service usage patterns.

## License

MIT
