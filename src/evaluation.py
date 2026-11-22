# src/evaluation.py
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
def calculate_ks(
        y_true,
        y_prob
):
    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )
    return np.max(
        tpr - fpr
    )
def evaluate_model(
        model,
        X_test,
        y_test
):
    y_pred = model.predict(
        X_test
    )
    y_prob = model.predict_proba(
        X_test
    )[:, 1]
    metrics = {
        "auc":
            roc_auc_score(
                y_test,
                y_prob
            ),
        "ks":
            calculate_ks(
                y_test,
                y_prob
            ),
        "pr_auc":
            average_precision_score(
                y_test,
                y_prob
            ),
        "precision":
            precision_score(
                y_test,
                y_pred
            ),
        "recall":
            recall_score(
                y_test,
                y_pred
            ),
        "f1":
            f1_score(
                y_test,
                y_pred
            )
    }
    return metrics