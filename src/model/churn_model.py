import json
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.core.config import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    HISTORY_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
)
from src.exceptions import DataPreparationError, ModelNotFoundError
from src.preprocessing.churn_preprocessor import build_preprocessor, split_data
from src.schemas.churn import TrainingConfigChurn


def build_model(
    preprocessor,
    model_type: str,
    hyperparameters: dict[str, Any]
) -> Pipeline:
    parameters = {"random_state": 42, **hyperparameters}

    if model_type in ("logistic_regression", "logreg"):
        classifier = LogisticRegression(**{"max_iter": 1000, **parameters})
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(**parameters)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", classifier)])


def train_churn_model(dataset: pd.DataFrame, config: TrainingConfigChurn) -> Pipeline:
    try:
        model = build_model(
            build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES),
            config.model_type,
            config.hyperparameters,
        )

        model.fit(dataset[FEATURE_NAMES], dataset["churn"])

    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise DataPreparationError(
            "Не удалось обучить модель: проверьте данные и гиперпараметры",
            {"reason": str(exc)},
        ) from exc

    return model


def train_and_evaluate(dataset: pd.DataFrame, config: TrainingConfigChurn) -> dict:
    X_train, X_test, _, y_test = split_data(
        dataset[FEATURE_NAMES],
        dataset["churn"]
    )

    model = train_churn_model(dataset.loc[X_train.index], config)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(
        X_test)[:, list(model.classes_).index(1)]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    return {
        "model": model,
        "date": datetime.now(timezone.utc),
        "metrics": metrics,
        "config": config.model_dump(),
        "feature_names": FEATURE_NAMES,
    }


def save_churn_model(data) -> bool:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary = MODEL_PATH.with_suffix(".tmp")

    try:
        joblib.dump(data, temporary)
        temporary.replace(MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)

    return True


def load_training_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []

    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list):
        raise ValueError("Training history must be a list")

    return history


def save_to_history(data):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = load_training_history()
    history.append(data)

    temporary = HISTORY_PATH.with_suffix(".tmp")

    try:
        temporary.write_text(json.dumps(
            history, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(HISTORY_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def load_churn_model() -> dict:
    try:
        saved = joblib.load(MODEL_PATH)
    except FileNotFoundError as error:
        raise ModelNotFoundError("Trained churn model not found") from error

    if (
        not isinstance(saved, dict)
        or not isinstance(saved.get("model"), Pipeline)
        or set(saved.get("feature_names", [])) != set(FEATURE_NAMES)
    ):
        raise ValueError("Invalid saved churn model")

    return saved
