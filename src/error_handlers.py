import json
import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from src.exceptions import DataPreparationError, EmptyDatasetError, ModelNotFoundError, ModelPredictionError
from src.schemas.error import ErrorResponse

logger = logging.getLogger(__name__)


def error_response(request: Request, status: int, code: str, message: str, details=None, headers=None):
    logger.error("Request failed path=%s code=%s message=%s", request.url.path, code, message)
    error = ErrorResponse(code=code, message=message, details=details)
    # JSON mode converts nonfinite validation inputs to null as well.
    return JSONResponse(status_code=status, content=json.loads(error.model_dump_json()), headers=headers)


async def http_exception_handler(request: Request, exc: HTTPException):
    return error_response(request, exc.status_code, "HTTP_ERROR", str(exc.detail),
                          {"status_code": exc.status_code}, headers=exc.headers)


async def data_preparation_exception_handler(request: Request, exc: DataPreparationError):
    return error_response(request, 400, "DATA_PREPARATION_ERROR", exc.message, exc.details)


async def model_prediction_exception_handler(request: Request, exc: ModelPredictionError):
    logger.error("Prediction failed", exc_info=exc)
    return error_response(request, 500, "MODEL_PREDICTION_ERROR", exc.message, exc.details)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [{"loc": item["loc"], "type": item["type"], "msg": item["msg"]} for item in exc.errors()]
    return error_response(request, 422, "VALIDATION_ERROR", "Некорректные входные данные", details)


async def empty_dataset_exception_handler(request: Request, exc: EmptyDatasetError):
    return error_response(request, 400, "EMPTY_DATASET", "Датасет пуст")


async def model_not_found_exception_handler(request: Request, exc: ModelNotFoundError):
    return error_response(request, 404, "MODEL_NOT_FOUND", "Обученная модель отсутствует")


async def unexpected_exception_handler(request: Request, exc: Exception):
    logger.error("Unexpected error path=%s", request.url.path, exc_info=exc)
    return error_response(request, 500, "INTERNAL_ERROR", "Внутренняя ошибка сервиса")
