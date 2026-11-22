from metaflow import FlowSpec, step, Parameter, card
import pandas as pd
import warnings
import os
import json
import joblib

warnings.filterwarnings("ignore")

from src.preprocessing import prepare_features
from src.modeling import (
    train_logistic_regression,
    train_xgboost
)
from src.evaluation import evaluate_model


class CreditRiskFlow(FlowSpec):

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------

    data_path = Parameter(
        "data_path",
        default="data/credit_risk_data_enhanced.csv",
        help="Path to dataset"
    )

    model_type = Parameter(
        "model_type",
        default="xgb",
        help="Model type: lr or xgb"
    )

    train_vintage_end = Parameter(
        "train_vintage_end",
        default="202212",
        help="Last vintage included in training"
    )

    test_vintage_start = Parameter(
        "test_vintage_start",
        default="202401",
        help="First vintage included in testing"
    )

    # --------------------------------------------------
    # STEP 1
    # --------------------------------------------------

    @card
    @step
    def start(self):

        print("=" * 60)
        print("STEP 1: DATA LOADING")
        print("=" * 60)

        self.df = pd.read_csv(
            self.data_path
        )

        required_cols = [
            "loan_id",
            "fico_score",
            "income",
            "loan_amount",
            "default_12m"
        ]

        missing_cols = [
            c for c in required_cols
            if c not in self.df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}"
            )

        print(
            f"Records Loaded: {len(self.df):,}"
        )

        print(
            f"Default Rate: "
            f"{self.df['default_12m'].mean():.2%}"
        )

        print(
            f"Vintage Range: "
            f"{self.df['vintage'].min()} "
            f"to "
            f"{self.df['vintage'].max()}"
        )

        self.next(self.preprocess)

    # --------------------------------------------------
    # STEP 2
    # --------------------------------------------------

    @step
    def preprocess(self):

        print("\n" + "=" * 60)
        print("STEP 2: TRAIN / TEST SPLIT")
        print("=" * 60)

        df = self.df.copy()

        df["vintage"] = (
            df["vintage"]
            .astype(str)
        )

        self.df_train = df[
            df["vintage"]
            <= str(self.train_vintage_end)
        ].copy()

        self.df_test = df[
            df["vintage"]
            >= str(self.test_vintage_start)
        ].copy()

        print(
            f"Train Shape: "
            f"{self.df_train.shape}"
        )

        print(
            f"Test Shape: "
            f"{self.df_test.shape}"
        )

        print(
            f"Train Default Rate: "
            f"{self.df_train['default_12m'].mean():.2%}"
        )

        print(
            f"Test Default Rate: "
            f"{self.df_test['default_12m'].mean():.2%}"
        )

        self.next(self.feature_engineering)

    # --------------------------------------------------
    # STEP 3
    # --------------------------------------------------

    @step
    def feature_engineering(self):

        print("\n" + "=" * 60)
        print("STEP 3: FEATURE ENGINEERING")
        print("=" * 60)

        self.df_train = prepare_features(
            self.df_train
        )

        self.df_test = prepare_features(
            self.df_test
        )

        self.X_train = self.df_train.drop(
            "default_12m",
            axis=1
        )

        self.y_train = self.df_train[
            "default_12m"
        ]

        self.X_test = self.df_test.drop(
            "default_12m",
            axis=1
        )

        self.y_test = self.df_test[
            "default_12m"
        ]

        print(
            f"Training Features: "
            f"{self.X_train.shape}"
        )

        print(
            f"Testing Features: "
            f"{self.X_test.shape}"
        )

        self.next(self.train)

    # --------------------------------------------------
    # STEP 4
    # --------------------------------------------------

    @step
    def train(self):

        print("\n" + "=" * 60)
        print("STEP 4: MODEL TRAINING")
        print("=" * 60)

        if self.model_type.lower() == "lr":

            self.model = (
                train_logistic_regression(
                    self.X_train,
                    self.y_train
                )
            )

        else:

            self.model = (
                train_xgboost(
                    self.X_train,
                    self.y_train
                )
            )

        print(
            f"Model Selected: "
            f"{self.model_type.upper()}"
        )

        self.next(self.evaluate)

    # --------------------------------------------------
    # STEP 5
    # --------------------------------------------------

    @card
    @step
    def evaluate(self):

        print("\n" + "=" * 60)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 60)

        self.metrics = evaluate_model(
            self.model,
            self.X_test,
            self.y_test
        )

        print("\nModel Metrics:")

        for k, v in self.metrics.items():

            print(
                f"{k}: {v:.4f}"
            )

        self.next(self.end)

    # --------------------------------------------------
    # STEP 6
    # --------------------------------------------------

    @step
    def end(self):

        print("\n" + "=" * 60)
        print("STEP 6: SAVE ARTIFACTS")
        print("=" * 60)

        os.makedirs(
            "outputs/models",
            exist_ok=True
        )

        os.makedirs(
            "outputs/metrics",
            exist_ok=True
        )

        model_path = (
            f"outputs/models/"
            f"{self.model_type}_model.pkl"
        )

        metric_path = (
            "outputs/metrics/"
            "metrics.json"
        )

        joblib.dump(
            self.model,
            model_path
        )

        with open(
            metric_path,
            "w"
        ) as f:

            json.dump(
                self.metrics,
                f,
                indent=4
            )

        print(
            f"Model Saved: {model_path}"
        )

        print(
            f"Metrics Saved: {metric_path}"
        )

        print("\nFinal Results")

        for k, v in self.metrics.items():

            print(
                f"{k}: {v:.4f}"
            )

        print("\nPipeline execution completed.")


if __name__ == "__main__":
    CreditRiskFlow()