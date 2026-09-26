import logging
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from src.api.routes import router
from src.error_handlers import (
    data_preparation_exception_handler, empty_dataset_exception_handler,
    http_exception_handler, model_not_found_exception_handler,
    model_prediction_exception_handler, unexpected_exception_handler, validation_exception_handler,
)
from src.exceptions import DataPreparationError, EmptyDatasetError, ModelNotFoundError, ModelPredictionError
from src.logger import setup_logging
from src.model.churn_model import load_churn_model

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.training_lock = Lock()
    try:
        app.state.model = load_churn_model()
        logger.info("Saved churn model loaded")
    except ModelNotFoundError:
        logger.info("No saved model; train it via POST /model/train")
    except Exception:
        logger.exception("Could not load saved model; retraining is available")
    yield
    app.state.model = None


def create_app() -> FastAPI:
    app = FastAPI(title="ML Churn Service", version="1.0.0", lifespan=lifespan)
    for exception, handler in [
        (HTTPException, http_exception_handler),
        (DataPreparationError, data_preparation_exception_handler),
        (ModelPredictionError, model_prediction_exception_handler),
        (RequestValidationError, validation_exception_handler),
        (EmptyDatasetError, empty_dataset_exception_handler),
        (ModelNotFoundError, model_not_found_exception_handler),
        (Exception, unexpected_exception_handler),
    ]:
        app.add_exception_handler(exception, handler)
    app.include_router(router)
    return app


app = create_app()
