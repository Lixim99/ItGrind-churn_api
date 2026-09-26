import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from src.exceptions import (
    DataPreparationError,
    EmptyDatasetError,
    ModelNotFoundError,
    ModelPredictionError,
)
from src.schemas.error import ErrorResponse

logger = logging.getLogger(__name__)


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):
    logger.error(
        "Model prediction error path=%s",
        request.url.path,
    )

    error = ErrorResponse(
        code="HTTP_ERROR",
        message=str(exc.detail),
        details={
            "status_code": exc.status_code,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error.model_dump(),
    )


async def data_preparation_exception_handler(
    request: Request,
    exc: DataPreparationError,
):
    logger.error(
        "Model prediction error path=%s message=%s details=%s",
        request.url.path,
        exc.message,
        exc.details,
    )

    error = ErrorResponse(
        code="DATA_PREPARATION_ERROR",
        message=exc.message,
        details=exc.details,
    )

    return JSONResponse(
        status_code=400,
        content=error.model_dump(),
    )


async def model_prediction_exception_handler(
    request: Request,
    exc: ModelPredictionError,
):
    logger.error(
        "Model prediction error path=%s message=%s details=%s",
        request.url.path,
        exc.message,
        exc.details,
    )

    error = ErrorResponse(
        code="MODEL_PREDICTION_ERROR",
        message=exc.message,
        details=exc.details,
    )

    return JSONResponse(
        status_code=500,
        content=error.model_dump(),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    logger.error(
        "Model prediction error path=%s",
        request.url.path,
    )

    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": "Некорректные входные данные",
            "details": exc.errors(),
        },
    )


async def empty_dataset_exception_handler(
    request: Request,
    exc: EmptyDatasetError,
):
    logger.error(
        "Model prediction error path=%s",
        request.url.path,
    )

    return JSONResponse(
        status_code=400,
        content={
            "code": "EMPTY_DATASET",
            "message": "Датасет пуст",
            "details": None,
        },
    )


async def model_not_found_exception_handler(
    request: Request,
    exc: ModelNotFoundError,
):
    logger.error(
        "Model prediction error path=%s ",
        request.url.path,
    )

    return JSONResponse(
        status_code=404,
        content={
            "code": "MODEL_NOT_FOUND",
            "message": "Обученная модель отсутствует",
            "details": None,
        },
    )
