import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.exceptions import ModelNotFoundError

MODEL_PATH = Path("models/churn_model.joblib")
HISTORY_PATH = Path("data/training_history.json")


def build_model(
    preprocessor,
    model_type: str,
    hyperparameters: dict[str, Any]
) -> Pipeline:

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            **hyperparameters
        )

    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            **hyperparameters
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier)
        ]
    )


def save_churn_model(data) -> bool:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    return bool(joblib.dump(data, MODEL_PATH))


def load_training_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []

    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_to_history(data):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = load_training_history()
    history.append(data)

    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            history,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_churn_model() -> dict:
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError as error:
        raise ModelNotFoundError(
            "Trained churn model not found"
        ) from error
