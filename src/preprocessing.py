# src/preprocessing.py

import numpy as np
import pandas as pd


def prepare_features(df):

    df = df.copy()

    # -------------------------
    # Date Features
    # -------------------------

    df['origination_date'] = pd.to_datetime(
        df['origination_date']
    )

    df['origination_year'] = (
        df['origination_date'].dt.year
    )

    df['origination_month'] = (
        df['origination_date'].dt.month
    )

    df['origination_quarter'] = (
        df['origination_date'].dt.quarter
    )

    # -------------------------
    # Logical Data Quality Check
    # -------------------------

    invalid_emp = (
        df['employment_length']
        >
        (df['age'] - 16)
    )

    df.loc[
        invalid_emp,
        'employment_length'
    ] = np.nan

    df['employment_length'] = (
        df['employment_length']
        .fillna(
            df['employment_length'].median()
        )
    )

    # -------------------------
    # Risk Features
    # -------------------------

    df['high_utilization'] = (
        df['utilization_rate'] > 0.8
    ).astype(int)

    df['high_dti'] = (
        df['debt_to_income'] > 0.8
    ).astype(int)

    df['medium_dti'] = (
        (df['debt_to_income'] > 0.4)
        &
        (df['debt_to_income'] <= 0.8)
    ).astype(int)

    df['low_dti'] = (
        df['debt_to_income'] <= 0.4
    ).astype(int)

    df['payment_ratio'] = (
        df['total_payments_to_date']
        /
        df['loan_amount'].clip(lower=1)
    )

    df['apr_fico_ratio'] = (
        df['apr']
        /
        df['fico_score'].clip(lower=1)
    )

    df['inquiry_per_trade'] = (
        df['inquiries_last_6m']
        /
        (df['num_open_trades'] + 1)
    )

    df['loan_income_ratio'] = (
        df['loan_amount']
        /
        df['income'].clip(lower=1000)
    )

    df['has_delinquency'] = (
        df['days_past_due_current'] > 0
    ).astype(int)

    df['log_income'] = (
        np.log1p(df['income'])
    )

    df['loan_to_income'] = (
        df['loan_amount']
        /
        df['income'].clip(lower=1000)
    )

    # -------------------------
    # Monthly Payment
    # -------------------------

    rate_m = df['apr'] / 100 / 12

    n_pmts = df['term']

    df['est_monthly_payment'] = np.where(
        rate_m == 0,
        df['loan_amount'] / n_pmts,
        (
            df['loan_amount']
            * rate_m
            * (1 + rate_m) ** n_pmts
        ) /
        (
            (1 + rate_m) ** n_pmts - 1
        )
    )

    df['payment_to_income'] = (
        df['est_monthly_payment'] * 12
    ) / df['income'].clip(lower=1000)

    # -------------------------
    # Buckets
    # -------------------------

    df['age_group'] = pd.cut(
        df['age'],
        bins=[18, 30, 45, 60, 100],
        labels=[
            'Young',
            'Middle',
            'Mature',
            'Senior'
        ]
    )

    df['fico_bucket'] = pd.cut(
        df['fico_score'],
        bins=[300, 580, 670, 740, 800, 850],
        labels=[
            'Poor',
            'Fair',
            'Good',
            'Very_Good',
            'Excellent'
        ]
    )

    # -------------------------
    # Remove Leakage
    # -------------------------

    leak_cols = ['sample_weight',
    'origination_year',
    'origination_month',
    'origination_quarter',
     'has_delinquency',
    'days_past_due_current',
    'total_payments_to_date',
    'payment_ratio',
    'payment_to_income']

    df.drop(
        columns=leak_cols,
        errors='ignore',
        inplace=True
    )

    # -------------------------
    # Drop Raw Fields
    # -------------------------

    df.drop(
        columns=[
            'loan_id',
            'origination_date'
        ],
        errors='ignore',
        inplace=True
    )

    # -------------------------
    # One Hot Encoding
    # -------------------------

    categorical_features = [
        'channel',
        'product_type',
        'state',
        'age_group',
        'fico_bucket'
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_features,
        drop_first=True
    )

    bool_cols = df.select_dtypes(
        include='bool'
    ).columns

    df[bool_cols] = (
        df[bool_cols]
        .astype(int)
    )

    return df