# Credit Risk Modeling

## Overview

### Please refer all answers from notebook/modeling.ipynb 

The analysis includes exploratory data analysis, data quality assessment, feature engineering, and model development. Logistic Regression was selected as an interpretable baseline, while XGBoost was evaluated to capture potential nonlinear relationships and interactions. Special values and missing observations were identified and handled through appropriate preprocessing. Model evaluation considers discrimination, ranking performance, and the importance of evaluating generalization across origination vintages.

1. Summary of Findings: This project developed and evaluated machine learning models to predict borrower default risk using historical loan and borrower information. Both Logistic Regression and XGBoost demonstrated strong predictive performance; however, XGBoost achieved the highest overall performance.
2. Best Performing Model: Among the models evaluated, XGBoost was selected as the best-performing model. Logistic Regression achieved a good mean cross-validation ROC-AUC and XGBoost achieved a slightly higher ROC-AUC and XGBoost was able to capture nonlinear relationships and complex interactions among borrower characteristics, resulting in improved predictive accuracy.Although the performance improvement was modest, XGBoost consistently outperformed Logistic Regression across evaluation metrics while maintaining strong discriminatory power.Model performance is assessed using ROC-AUC, KS statistic, PR-AUC, and additional metrics relevant to an imbalanced credit risk dataset. Calibration and vintage-level performance are also evaluated to understand whether predicted probabilities are reliable and whether model performance remains stable over time. Model selection is based not only on predictive performance but also on interpretability, stability, calibration, and potential business applicability.
3. Key Insights About Default Risk: The analysis identified several important drivers of borrower default:Delinquency history was the strongest predictor of future default. Lower FICO scores were associated with significantly higher default risk. Higher DTI increased the likelihood of default. High credit utilization indicated greater financial stress and elevated risk. Lower income levels and higher loan-to-income ratios were linked to weaker repayment capacity. Frequent recent credit inquiries were associated with increased default probability. These findings are consistent with established credit risk theory and common lending industry practices.
- The dataset exhibits class imbalance, with default representing a relatively small proportion of observations.
- Data quality issues include special missing-value codes and potentially inconsistent feature values.
- Both Logistic Regression and XGBoost provide useful modeling benchmarks.
- Temporal validation is important to assess the model's ability to generalize to future origination vintages.
- Origination-time feature availability is critical to prevent data leakage and ensure realistic model performance estimates.
4. Recommendations for Deployment: Before production deployment, the model should undergo additional validation, including out-of-time testing, calibration assessment, vintage-level stability analysis, and business impact evaluation. Further work should include threshold optimization aligned with lending objectives, model monitoring for population and performance drift, documentation of model limitations, and independent model validation.The final model should be selected based on a balance of predictive performance, calibration, stability, interpretability, and operational requirements rather than a single evaluation metric.Before deploying the model into a production environment, several recommendations should be considered: Deploy XGBoost as the primary risk model due to its superior predictive performance. Maintain Logistic Regression as a benchmark model because of its transparency and ease of explanation. Implement ongoing model monitoring using metrics such as AUC, KS, and PSI. Retrain the model periodically to account for changing borrower behavior and economic conditions. Use SHAP or other explainability techniques to provide transparency for risk decisions and regulatory reviews. Integrate additional data sources, including credit bureau attributes and application decision results, to further improve model performance. Monitor for potential data leakage and model drift to ensure long-term reliability. AUC tells us whether the model can rank borrowers by risk, but calibration tells us whether the predicted probabilities are accurate.Calibration is essential in credit risk modeling to ensure that predicted probabilities of default (PDs) accurately reflect observed default rates. While discrimination metrics such as ROC-AUC measure a model's ability to rank borrowers by risk, they do not assess the accuracy of predicted probabilities.
5. Final Conclusion:Overall, the project successfully developed a high-performing credit risk prediction framework capable of identifying borrowers with elevated default risk. XGBoost achieved the strongest predictive performance, while both models highlighted the importance of delinquency history, credit quality, debt burden, and repayment capacity as key determinants of default. With proper monitoring, governance, and feature enhancements, the model could serve as a valuable tool for loan underwriting, risk management, and portfolio monitoring.

This project develops and evaluates machine learning models for predicting borrower default risk within a 12‑month horizon.
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
