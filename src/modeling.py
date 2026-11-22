# src/modeling.py

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_logistic_regression(
        X_train,
        y_train
):

    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def train_xgboost(
        X_train,
        y_train
):

    scale_pos_weight = (
        (y_train == 0).sum()
        /
        (y_train == 1).sum()
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(
        X_train,
        y_train
    )

    return model