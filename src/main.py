import logging
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import pandas
from fastapi import FastAPI
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.dataset.dataset import ChurnDataset
from src.error_handlers import *
from src.logger import setup_logging
from src.model.churn_model import *
from src.preprocessing.churn_preprocessor import build_preprocessor, split_data
from src.schemas.churn import *

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_exception_handler(
    HTTPException,
    http_exception_handler,
)

app.add_exception_handler(
    DataPreparationError,
    data_preparation_exception_handler,
)

app.add_exception_handler(
    ModelPredictionError,
    model_prediction_exception_handler,
)

app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler,
)

app.add_exception_handler(
    EmptyDatasetError,
    empty_dataset_exception_handler,
)

app.add_exception_handler(
    ModelNotFoundError,
    model_not_found_exception_handler,
)


ModelType = Literal[
    "logistic_regression",
    "random_forest",
]


@app.get("/")
async def read_root():
    return {"message": "ml churn service is running"}


# Day 7
@app.post("/predict")
async def predict_churn(data: list[FeatureVectorChurn]) -> PredictionResponseChurn:
    logger.info(
        "Prediction requested objects=%s",
        len(data),
    )

    saved = load_churn_model()

    model: Pipeline = saved["model"]
    feature_names: list[str] = saved["feature_names"]

    X = pandas.DataFrame([
        item.model_dump()
        for item in data
    ])

    missing = set(feature_names) - set(X.columns)
    extra = set(X.columns) - set(feature_names)

    if missing:
        raise DataPreparationError(
            message="Не удалось подготовить данные",
            details={
                "missing_features": missing,
            },
        )

    if extra:
        raise DataPreparationError(
            message="Не удалось подготовить данные",
            details={
                "extra_features": extra,
            },
        )

    X = X[feature_names]

    result = model.predict(X)
    probabilities = model.predict_proba(X)

    classes = {}

    for index, _ in enumerate(data):
        classes[index] = {}

        for class_index, class_label in enumerate(model.classes_):
            classes[index][int(class_label)] = round(
                float(probabilities[index][class_index]),
                2,
            )

    logger.info(
        "Prediction completed objects=%s",
        len(data),
    )

    return PredictionResponseChurn(
        churn=result.tolist(),
        classes=classes,
    )


@app.get("/dataset/preview")
async def preview_dataset():
    dataset = ChurnDataset(path="data/churn_dataset.csv")
    dataset.load()

    return dataset.get_preview()


@app.get("/dataset/info")
async def dataset_info() -> dict:
    dataset = ChurnDataset(path="data/churn_dataset.csv")
    dataset.load()

    return dataset.get_info()


@app.get("/dataset/split-info")
async def split_info() -> dict:
    dataset = ChurnDataset(path="data/churn_dataset.csv")
    dataset.load()

    X, y = dataset.get_xy()

    X_train, X_test, y_train, y_test = split_data(X, y)

    return {
        "X_train": len(X_train),
        "X_test": len(X_test),
        "y_train": y_train.value_counts(normalize=True).to_dict(),
        "y_test": y_test.value_counts(normalize=True).to_dict()
    }


# Day 5,7
@app.post("/model/train")
async def train(config: TrainingConfigChurn) -> dict:
    logger.info(
        "Model training started model_type=%s hyperparameters=%s",
        config.model_type,
        config.hyperparameters,
    )

    dataset = ChurnDataset(path="data/churn_dataset.csv")
    dataset.load()

    X, y = dataset.get_xy()

    categorical_features = [
        "region",
        "device_type",
        "payment_method",
        "autopay_enabled",
    ]

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = build_model(
        preprocessor,
        config.model_type,
        config.hyperparameters
    )

    X_train, X_test, y_train, y_test = split_data(X, y)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc
    }

    logger.info(
        "Model training completed model_type=%s accuracy=%.4f f1=%.4f roc_auc=%.4f",
        config.model_type,
        accuracy,
        f1,
        roc_auc
    )

    date = datetime.now(ZoneInfo("Europe/Moscow"))

    save_to_history({
        "timestamp": date.isoformat(),
        "model_type": config.model_type,
        "hyperparameters": config.hyperparameters,
        "metrics": metrics,
    })

    save_churn_model({
        "model": model,
        "date": date,
        "metrics": metrics,
        "config": config.model_dump(),
        "feature_names": categorical_features + numeric_features,
    })

    return metrics


# Day 9
@app.get("/model/schema")
async def get_schema() -> dict:
    return FeatureVectorChurn.model_json_schema()


# Day 6
@app.get("/model/status")
async def get_status() -> dict:
    data = load_churn_model()

    if not data:
        return {
            "fitted": False,
            "last_fitted": False,
            "metrics": False,
            "config": False
        }

    return {
        "fitted": True,
        "last_fitted": data.get("date"),
        "metrics": data.get("metrics"),
        "config": data.get("config"),
        "feature_names": data.get("feature_names")
    }


# Day11
@app.get("/model/metrics")
async def get_metrics(model_type: ModelType | None = None) -> dict:
    history = load_training_history()

    if not history:
        return {
            "latest": None,
            "history": [],
        }

    if model_type is not None:
        history = [
            item
            for item in history
            if item.get("model_type") == model_type
        ]

    return {
        "latest": history[-1],
        "history": history[-5:],
    }


@app.get("/health")
async def check_health():
    model_available = False
    dataset_loaded = False

    try:
        load_churn_model()
        model_available = True
    except Exception:
        pass

    try:
        dataset = ChurnDataset(
            path="data/churn_dataset.csv"
        )
        dataset.load()

        dataset_loaded = True
    except Exception:
        pass

    healthy = (
        model_available
        and dataset_loaded
    )

    return {
        "status": "ok" if healthy else "degraded",
        "model_available": model_available,
        "dataset_loaded": dataset_loaded,
    }
