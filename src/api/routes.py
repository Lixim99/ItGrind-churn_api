import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Body, Query, Request
from pydantic import Field

from src.core.config import DATASET_PATH
from src.dataset.dataset import ChurnDataset
from src.exceptions import DataPreparationError, EmptyDatasetError, ModelNotFoundError, ModelPredictionError
from src.model.churn_model import load_training_history, save_churn_model, save_to_history, train_and_evaluate
from src.preprocessing.churn_preprocessor import split_data
from src.schemas.churn import (
    FEATURE_EXAMPLE, DatasetRowChurn, FeatureVectorChurn, ModelType,
    PredictionResponseChurn, TrainingConfigChurn,
)
from src.schemas.error import ErrorResponse

logger = logging.getLogger(__name__)
ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Ошибка данных или гиперпараметров", "content": {
        "application/json": {"example": {"code": "EMPTY_DATASET", "message": "Датасет пуст", "details": None}}
    }},
    404: {"model": ErrorResponse, "description": "Модель ещё не обучена", "content": {
        "application/json": {"example": {"code": "MODEL_NOT_FOUND", "message": "Обученная модель отсутствует", "details": None}}
    }},
    422: {"model": ErrorResponse, "description": "Неверный формат запроса", "content": {
        "application/json": {"example": {"code": "VALIDATION_ERROR", "message": "Некорректные входные данные", "details": []}}
    }},
    500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервиса"},
}
router = APIRouter(responses=ERROR_RESPONSES)
PredictionInput = FeatureVectorChurn | Annotated[list[FeatureVectorChurn], Field(min_length=1)]


def read_dataset() -> ChurnDataset:
    dataset = ChurnDataset(DATASET_PATH)
    dataset.load()
    return dataset


@router.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@router.get("/dataset/preview", response_model=list[DatasetRowChurn])
def preview_dataset(count: int = Query(default=5, ge=1, le=1000)):
    """Первые count строк CSV. Пропуски возвращаются как null."""
    return read_dataset().get_preview(count)


@router.get("/dataset/info")
def dataset_info() -> dict:
    return read_dataset().get_info()


@router.get("/dataset/split-info")
def split_info() -> dict:
    X, y = read_dataset().get_xy()
    X_train, X_test, y_train, y_test = split_data(X, y)
    return {
        "X_train": len(X_train), "X_test": len(X_test),
        "y_train": y_train.value_counts(normalize=True).to_dict(),
        "y_test": y_test.value_counts(normalize=True).to_dict(),
    }


@router.post("/model/train")
def train(config: TrainingConfigChurn, request: Request) -> dict:
    """Обучить и сохранить pipeline. Ошибки CSV и параметров возвращаются в формате code/message/details."""
    with request.app.state.training_lock:
        logger.info("Model training started model_type=%s", config.model_type)
        dataset = read_dataset()
        saved = train_and_evaluate(dataset.df, config)
        # Проверяем историю перед заменой рабочей модели.
        load_training_history()
        save_churn_model(saved)
        request.app.state.model = saved
        save_to_history({
            "timestamp": saved["date"].isoformat(), **saved["config"], "metrics": saved["metrics"],
        })
        logger.info("Model training completed model_type=%s metrics=%s", config.model_type, saved["metrics"])
    return saved["metrics"]


@router.post("/predict", response_model=PredictionResponseChurn)
def predict_churn(
    data: Annotated[PredictionInput, Body(openapi_examples={
        "single": {"summary": "Один клиент", "value": FEATURE_EXAMPLE},
        "batch": {"summary": "Несколько клиентов", "value": [FEATURE_EXAMPLE, FEATURE_EXAMPLE]},
    })],
    request: Request,
) -> PredictionResponseChurn:
    """Один клиент или непустой список. Ответ всегда содержит списки классов и вероятности по индексам клиентов."""
    items = [data] if isinstance(data, FeatureVectorChurn) else data
    saved = request.app.state.model
    if saved is None:
        raise ModelNotFoundError("Trained churn model not found")
    logger.info("Prediction requested objects=%s", len(items))
    try:
        frame = pd.DataFrame([item.model_dump() for item in items], columns=saved["feature_names"])
        model = saved["model"]
        result = model.predict(frame)
        probabilities = model.predict_proba(frame)
        response = PredictionResponseChurn(
            churn=result.tolist(),
            classes={i: {int(label): float(probability) for label, probability in zip(model.classes_, row)}
                     for i, row in enumerate(probabilities)},
        )
    except Exception as exc:
        raise ModelPredictionError("Не удалось получить предсказание модели") from exc
    logger.info("Prediction completed objects=%s", len(items))
    return response


@router.get("/model/schema")
def get_schema() -> dict:
    return FeatureVectorChurn.model_json_schema()


@router.get("/model/status")
def get_status(request: Request) -> dict:
    saved = request.app.state.model
    return {
        "fitted": saved is not None,
        "last_fitted": saved["date"].isoformat() if saved and saved.get("date") else None,
        "metrics": saved.get("metrics") if saved else None,
        "config": saved.get("config") if saved else None,
        "feature_names": saved.get("feature_names") if saved else None,
    }


@router.get("/model/metrics")
def get_metrics(model_type: ModelType | None = None, limit: int = Query(default=5, ge=1, le=100)) -> dict:
    history = load_training_history()
    if model_type is not None:
        normalize = lambda name: "logistic_regression" if name == "logreg" else name
        history = [item for item in history if normalize(item.get("model_type")) == normalize(model_type)]
    return {"latest": history[-1] if history else None, "history": history[-limit:]}


@router.get("/health")
def check_health(request: Request):
    dataset_loaded = False
    try:
        read_dataset()
        dataset_loaded = True
    except (DataPreparationError, EmptyDatasetError) as exc:
        logger.warning("Dataset unavailable during health check: %s", exc)
    model_available = request.app.state.model is not None
    return {
        "status": "ok" if model_available and dataset_loaded else "degraded",
        "model_available": model_available, "dataset_loaded": dataset_loaded,
    }
