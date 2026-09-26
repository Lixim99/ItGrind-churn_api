import logging

import pandas as pd

from src.exceptions import EmptyDatasetError
from src.schemas.churn import DatasetRowChurn

logger = logging.getLogger(__name__)


class ChurnDataset:
    def __init__(self, path: str):
        self.path = path
        self.df: pd.DataFrame | None = None
        self.rows: list[DatasetRowChurn] = []

    def load(self) -> None:
        logger.info(
            "Loading churn dataset path=%s",
            self.path,
        )

        try:
            self.df = pd.read_csv(self.path)

            if self.df.empty:
                logger.warning(
                    "Churn dataset is empty path=%s",
                    self.path,
                )

                raise EmptyDatasetError(
                    "Dataset is empty"
                )

            logger.info(
                "Churn dataset loaded rows=%s columns=%s",
                self.df.shape[0],
                self.df.shape[1],
            )

        except Exception:
            logger.exception(
                "Failed to load churn dataset path=%s",
                self.path,
            )
            raise

        self.rows = [
            DatasetRowChurn(**row)
            for row in self.df.to_dict(orient="records")
        ]

    def get_xy(self) -> tuple[pd.DataFrame, pd.Series]:
        if self.df is None:
            raise RuntimeError("Dataset is not loaded")

        X = self.df.drop(columns=["churn"])
        y = self.df["churn"]

        return X, y

    def get_info(self) -> list[dict]:
        if self.df is None:
            raise RuntimeError("Dataset is not loaded")

        return [
            {
                "row_count": len(self.df),
                "column_count": len(self.df.columns),
                "columns": list(self.df.columns),
                "churn_distribution": self.df["churn"].value_counts().to_dict(),
            }
        ]

    def get_preview(self, count: int = 5) -> list[dict]:
        if self.df is None:
            raise RuntimeError("Dataset is not loaded")

        return self.rows[:count]
