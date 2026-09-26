import pandas as pd
import pytest


@pytest.fixture
def churn_frame():
    """Маленький детерминированный датасет с обоими классами."""
    return pd.DataFrame([
        {
            "monthly_fee": 10.0 + i,
            "usage_hours": 5.0 + i % 7,
            "support_requests": i % 4,
            "account_age_months": i + 1,
            "failed_payments": i % 2,
            "region": ["america", "europe", "asia"][i % 3],
            "device_type": ["desktop", "mobile", "tablet"][i % 3],
            "payment_method": "card" if i % 3 else "paypal",
            "autopay_enabled": 1 - i % 2,
            "churn": i % 2,
        }
        for i in range(40)
    ])


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    # Пути приложения относительные: реальные CSV, модель и история не затрагиваются.
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def dataset_path(isolated_workspace, churn_frame):
    path = isolated_workspace / "data" / "churn_dataset.csv"
    path.parent.mkdir()
    churn_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def feature_columns():
    return (
        ["monthly_fee", "usage_hours", "support_requests", "account_age_months", "failed_payments"],
        ["region", "device_type", "payment_method", "autopay_enabled"],
    )


@pytest.fixture(params=["logistic_regression", "random_forest"])
def training_config(request):
    parameters = {"random_state": 42}
    if request.param == "logistic_regression":
        parameters["max_iter"] = 1000
    else:
        parameters.update(n_estimators=20, max_depth=4, n_jobs=1)
    return {"model_type": request.param, "hyperparameters": parameters}


@pytest.fixture
def prediction_payload(churn_frame):
    return churn_frame.drop(columns="churn").iloc[:3].to_dict(orient="records")


@pytest.fixture
def client(isolated_workspace):
    # Unit-тесты не импортируют и не запускают FastAPI.
    from fastapi.testclient import TestClient

    from src.main import app

    with TestClient(app) as test_client:
        yield test_client
