import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.core.config import FEATURE_NAMES
from src.exceptions import DataPreparationError, EmptyDatasetError
from src.schemas.churn import DatasetRowChurn

logger = logging.getLogger(__name__)


class ChurnDataset:
    def __init__(self, path: str | Path):
        self.path = path
        self.df: pd.DataFrame | None = None
        self.rows: list[DatasetRowChurn] = []

    def load(self) -> None:
        logger.info("Loading churn dataset path=%s", self.path)

        self.df, self.rows = None, []

        try:
            frame = pd.read_csv(self.path)
        except pd.errors.EmptyDataError as exc:
            raise EmptyDatasetError("Dataset is empty") from exc
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise DataPreparationError(
                "Не удалось прочитать CSV датасета") from exc

        if frame.empty:
            raise EmptyDatasetError("Dataset is empty")

        expected = set(FEATURE_NAMES) | {"churn"}

        if set(frame.columns) != expected:
            raise DataPreparationError("Неверный набор столбцов датасета", {
                "missing_features": sorted(expected - set(frame.columns)),
                "extra_features": sorted(set(frame.columns) - expected),
            })

        records = frame.astype(object).where(
            frame.notna(), None).to_dict("records")

        try:
            self.rows = [DatasetRowChurn(**row) for row in records]
        except ValidationError as exc:
            raise DataPreparationError(
                "Некорректные значения в CSV датасете",
                exc.errors(include_url=False, include_context=False,
                           include_input=False),
            ) from exc

        self.df = pd.DataFrame([
            row.model_dump()
            for row in self.rows
        ], columns=frame.columns)

        self.df = self.df.fillna(np.nan)

        logger.info("Churn dataset loaded rows=%s columns=%s", *self.df.shape)

    def _require_loaded(self) -> pd.DataFrame:
        if self.df is None:
            raise DataPreparationError("Dataset is not loaded")

        return self.df

    def get_xy(self) -> tuple[pd.DataFrame, pd.Series]:
        frame = self._require_loaded()

        return frame[FEATURE_NAMES], frame["churn"]

    def get_info(self) -> dict:
        frame = self._require_loaded()

        return {
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "columns": list(frame.columns),
            "churn_distribution": frame["churn"].value_counts().to_dict(),
        }

    def get_preview(self, count: int = 5) -> list[DatasetRowChurn]:
        self._require_loaded()

        return self.rows[:count]
