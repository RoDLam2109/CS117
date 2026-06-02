"""Training and inference utilities for module 2 lifestyle risk detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.preprocessing.preprocess_depression import DEPRESSION_FEATURE_COLUMNS


@dataclass
class ModelRunResult:
    """Structured output for one trained model."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


class LifestyleRiskModel:
    """Compare baseline models and produce lifestyle risk scores for module 2."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_columns = list(DEPRESSION_FEATURE_COLUMNS)
        self.numeric_features = [
            "Academic Pressure",
            "CGPA",
            "Study Satisfaction",
            "Sleep Duration",
            "Dietary Habits",
            "Have you ever had suicidal thoughts ?",
            "Work/Study Hours",
            "Financial Stress",
            "Family History of Mental Illness",
            "Age",
        ]
        self.categorical_features = ["Gender"]
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", encoder),
                        ]
                    ),
                    self.categorical_features,
                ),
            ]
        )
        self.model_candidates = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "random_forest": RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
            "svm_rbf": SVC(probability=True, random_state=random_state),
        }
        self.pipeline: Pipeline | None = None
        self.best_model_name: str | None = None
        self.is_trained = False

    def _build_pipeline(self, estimator: Any) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", clone(self.preprocessor)),
                ("model", estimator),
            ]
        )

    def _evaluate_predictions(
        self, model_name: str, y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series
    ) -> ModelRunResult:
        return ModelRunResult(
            model_name=model_name,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_true, y_proba),
        )

    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame | pd.Series]]:
        """Train several baselines and keep the best one by F1, then ROC-AUC."""
        X_train, X_test, y_train, y_test = train_test_split(
            X[self.feature_columns],
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state,
        )

        results: list[ModelRunResult] = []
        fitted_pipelines: dict[str, Pipeline] = {}

        for model_name, estimator in self.model_candidates.items():
            pipeline = self._build_pipeline(estimator)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            results.append(self._evaluate_predictions(model_name, y_test, y_pred, y_proba))
            fitted_pipelines[model_name] = pipeline

        results_df = pd.DataFrame([result.__dict__ for result in results]).sort_values(
            by=["f1_score", "roc_auc", "recall"],
            ascending=False,
        )
        self.best_model_name = str(results_df.iloc[0]["model_name"])
        self.pipeline = fitted_pipelines[self.best_model_name]
        self.is_trained = True

        holdout = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        return results_df.reset_index(drop=True), holdout

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, model_name: str | None = None):
        """Train a selected model or the current best model."""
        chosen_model = model_name or self.best_model_name or "gradient_boosting"
        if chosen_model not in self.model_candidates:
            raise ValueError(f"Unknown model '{chosen_model}'")
        self.pipeline = self._build_pipeline(self.model_candidates[chosen_model])
        self.pipeline.fit(X_train[self.feature_columns], y_train)
        self.best_model_name = chosen_model
        self.is_trained = True
        return self

    def predict(self, X: pd.DataFrame):
        """Predict binary lifestyle risk labels."""
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Model must be trained first")
        return self.pipeline.predict(X[self.feature_columns])

    def predict_proba(self, X: pd.DataFrame):
        """Return class probabilities for lifestyle risk."""
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Model must be trained first")
        return self.pipeline.predict_proba(X[self.feature_columns])

    def predict_scores(self, X: pd.DataFrame) -> pd.Series:
        """Return the positive-class probability as module 2 lifestyle_score."""
        scores = self.predict_proba(X)[:, 1]
        return pd.Series(scores, index=X.index, name="lifestyle_score")
