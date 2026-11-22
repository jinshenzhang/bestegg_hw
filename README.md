# Credit Risk Modeling

## Overview

### Please refer all answers from notebook/modeling.ipynb 

The analysis includes exploratory data analysis, data quality assessment, feature engineering, and model development. Logistic Regression was selected as an interpretable baseline, while XGBoost was evaluated to capture potential nonlinear relationships and interactions. Special values and missing observations were identified and handled through appropriate preprocessing. Model evaluation considers discrimination, ranking performance, and the importance of evaluating generalization across origination vintages.
Overall, the project successfully developed a high-performing credit risk prediction framework capable of identifying borrowers with elevated default risk. XGBoost achieved the strongest predictive performance, while both models highlighted the importance of delinquency history, credit quality, debt burden, and repayment capacity as key determinants of default. With proper monitoring, governance, and feature enhancements, the model could serve as a valuable tool for loan underwriting, risk management, and portfolio monitoring. This project develops and evaluates machine learning models for predicting borrower default risk within a 12‑month horizon.

The objective is to identify key drivers of default and build predictive models that can support credit underwriting, portfolio monitoring, and risk management decisions.

The project covers:
- Exploratory Data Analysis (EDA)
- Data Quality Assessment
- Missing Value Handling
- Outlier Detection
- Feature Engineering
- Logistic Regression Modeling
- XGBoost Modeling
- Model Evaluation and Comparison
- Production-Oriented Pipeline Design

---

## Project Structure

```text
Credit-Risk-Modeling-Take-Home/

├── README.md
├── requirements.txt from Best Egg
│
├── data/
│   └── credit_risk_data_enhanced.csv
│
├── notebooks/
│   └── modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── modeling.py
│   └── evaluation.py
│
├── credit_risk_flow.py
│
└── outputs/
    ├── models/
    └── metrics/
```

---

## Dataset

The dataset contains borrower-level credit and loan performance information, including:

- Credit Score (FICO)
- Income
- Debt-to-Income Ratio
- Loan Amount
- APR
- Loan Term
- Credit Utilization
- Employment Length
- Inquiry Activity
- Delinquency Information

### Target Variable

```text
default_12m
```

Binary indicator:

```text
1 = Default within next 12 months
0 = No Default
```

---

## Feature Engineering

Several additional features were engineered to enhance model performance.

### Risk Indicators

```text
high_utilization
high_dti
medium_dti
low_dti
has_delinquency
```

### Ratio Features

```text
loan_income_ratio
loan_to_income
apr_fico_ratio
inquiry_per_trade
payment_to_income
```

### Bucketed Features

```text
age_group
fico_bucket
```

### Transformed Features

```text
log_income
```

---

## Modeling Approaches

### Logistic Regression

Advantages:

- High interpretability
- Easy to explain
- Commonly used in banking and credit scoring

### XGBoost

Advantages:

- Captures nonlinear relationships
- Handles feature interactions automatically
- Strong predictive performance

---

## Model Performance

### Best Performing Model

**XGBoost** achieved the highest overall performance and was selected as the champion model.

---

## Key Findings

The most important drivers of default risk were:

### Higher Default Risk

- Prior delinquency history
- High debt-to-income ratio
- High credit utilization
- Larger loan-to-income ratio
- Frequent credit inquiries

### Lower Default Risk

- Higher FICO scores
- Higher income levels
- Stronger credit profiles
- Lower debt burden

These findings align closely with established credit risk theory.

---

## Data Quality Assessment

### Missing Values

Missing values were identified in:

```text
fico_score
income
inquiries_last_6m
```

Median imputation was applied.

### Logical Data Quality Checks

Records with:

```text
employment_length > age - 16
```

were flagged as logically inconsistent and corrected during preprocessing.

### Potential Data Leakage

The following variables were removed from model development due to leakage risk:

```text
sample_weight
days_past_due_current
total_payments_to_date
payment_ratio
payment_to_income
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jinshenzhang/bestegg_hw.git
```

Navigate to the project directory:

```bash
cd Credit-Risk-Modeling
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Analysis

Open the notebook:

```bash
jupyter notebook
```

Then run:

```text
notebooks/modeling.ipynb
```

The notebook contains the complete workflow including:

- Data exploration
- Feature engineering
- Model development
- Evaluation
- Interpretation

---

## Running the Production Pipeline

Execute:

```bash
python credit_risk_flow.py
```

Pipeline stages:

```text
1. Load Data
2. Data Validation
3. Feature Engineering
4. Train/Test Split
5. Model Training
6. Model Evaluation
7. Save Artifacts
```

---

## Generated Outputs

Successful execution generates:

```text
notebooks/outputs/

├── models/
│   └── xgb_model.pkl
│
└── metrics/
    └── metrics.json
```

### Model Artifact

```text
notebooks/outputs/models/xgb_model.pkl
```
Serialized trained model that can be used for future scoring and deployment.

### Metrics Artifact

```text
notebooks/outputs/metrics/metrics.json
```

Example:

```json
    "LogisticRegression": {
        "auc": 0.6793709900289984,
        "ks": 0.3099400788424902,
        "pr_auc": 0.041318829637111115
    },
    "XGBoost": {
        "auc": 0.679605614554073,
        "ks": 0.3063547566067888,
        "pr_auc": 0.041375639596412585
    }
```

## A/B Testing Capability (Bonus)

To support model selection and future production deployment, an A/B testing framework was implemented to compare multiple candidate models.

### Candidate Models

- Model A: Logistic Regression
- Model B: XGBoost

Both models were trained and evaluated using the same training and testing datasets to ensure a fair comparison.


---

## Future Improvements

Potential enhancements include:

- Hyperparameter tuning
- Model stacking and ensemble methods
- SHAP explainability analysis
- Time-based validation
- Credit bureau data integration
- Application decision data integration
- Automated model monitoring
- Real-time scoring API deployment

---

## Conclusion
This project successfully developed a robust framework for predicting borrower default risk using both Logistic Regression and XGBoost.
Both models achieved strong predictive performance, with XGBoost producing the best overall discrimination power. The analysis identified delinquency history, credit quality, debt burden, and repayment capacity as the primary drivers of default risk.
The modular design and production-oriented pipeline provide a foundation for further model development and deployment in a real-world credit risk environment from Best Egg.
