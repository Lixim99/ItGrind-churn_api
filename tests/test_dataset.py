import pandas as pd
import pytest
from pydantic import ValidationError

from src.dataset.dataset import ChurnDataset


def test_load_csv_and_get_xy(dataset_path, churn_frame):
    dataset = ChurnDataset(str(dataset_path))
    dataset.load()

    X, y = dataset.get_xy()

    pd.testing.assert_frame_equal(X, churn_frame.drop(columns="churn"))
    pd.testing.assert_series_equal(y, churn_frame["churn"])
    assert [row.model_dump() for row in dataset.rows] == churn_frame.to_dict("records")
    # Получение признаков не должно удалять target из исходного датасета.
    pd.testing.assert_frame_equal(dataset.df, churn_frame)


def test_dataset_preview_and_info(dataset_path, churn_frame):
    dataset = ChurnDataset(str(dataset_path))
    dataset.load()

    assert [row.model_dump() for row in dataset.get_preview(3)] == churn_frame.head(3).to_dict("records")
    assert len(dataset.get_preview()) == 5
    assert dataset.get_info() == [{
        "row_count": 40,
        "column_count": 10,
        "columns": list(churn_frame.columns),
        "churn_distribution": {0: 20, 1: 20},
    }]


@pytest.mark.parametrize("method", ["get_xy", "get_preview", "get_info"])
def test_dataset_must_be_loaded(method):
    dataset = ChurnDataset("missing.csv")

    with pytest.raises(RuntimeError, match="Dataset is not loaded"):
        getattr(dataset, method)()


def test_load_missing_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChurnDataset(str(tmp_path / "missing.csv")).load()


@pytest.mark.parametrize("invalid_column", ["churn", "monthly_fee"])
def test_load_rejects_invalid_csv(dataset_path, churn_frame, invalid_column):
    churn_frame[invalid_column] = "not-a-number"
    churn_frame.to_csv(dataset_path, index=False)

    with pytest.raises(ValidationError) as error:
        ChurnDataset(str(dataset_path)).load()

    assert error.value.errors()[0]["loc"] == (invalid_column,)
