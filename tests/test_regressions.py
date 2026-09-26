import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from src.model.churn_model import train_churn_model
from src.preprocessing.churn_preprocessor import build_preprocessor
from src.schemas.churn import TrainingConfigChurn

CONFIG = {"model_type": "logreg", "hyperparameters": {"max_iter": 1000}}


def test_dataset_routes_and_schema(client, dataset_path, churn_frame):
    assert client.get("/").json() == {"message": "ml churn service is running"}
    assert len(client.get("/dataset/preview?count=7").json()) == 7
    assert client.get("/dataset/preview?count=0").status_code == 422
    info = client.get("/dataset/info")
    assert info.status_code == 200
    assert info.json()["row_count"] == 40
    assert info.json()["churn_distribution"] == {"0": 20, "1": 20}
    assert client.get("/dataset/split-info").json() == {
        "X_train": 32, "X_test": 8,
        "y_train": {"0": 0.5, "1": 0.5}, "y_test": {"0": 0.5, "1": 0.5},
    }
    schema = client.get("/model/schema").json()
    assert set(schema["required"]) == set(churn_frame.columns) - {"churn"}
    assert schema["properties"]["region"]["enum"] == ["europe", "asia", "america", "africa"]
    assert client.get("/docs").status_code == 200
    openapi = client.get("/openapi.json").json()
    for path in ("/model/train", "/predict"):
        assert openapi["paths"][path]["post"]["responses"]["422"]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")


def test_status_and_health_before_training(client, dataset_path):
    assert client.get("/model/status").json() == {
        "fitted": False, "last_fitted": None, "metrics": None, "config": None, "feature_names": None,
    }
    assert client.get("/health").json() == {
        "status": "degraded", "model_available": False, "dataset_loaded": True,
    }


def test_restart_loads_model_once(dataset_path, prediction_payload, monkeypatch):
    with TestClient(create_app()) as first:
        assert first.post("/model/train", json=CONFIG).status_code == 200
        before = first.post("/predict", json=prediction_payload).json()
        status = first.get("/model/status").json()
    with TestClient(create_app()) as restarted:
        def unexpected_load():
            raise AssertionError("The model must stay in memory after startup")
        monkeypatch.setattr("src.main.load_churn_model", unexpected_load)
        assert restarted.get("/model/status").json() == status
        assert restarted.get("/health").json()["status"] == "ok"
        assert restarted.post("/predict", json=prediction_payload).json() == before
        single = restarted.post("/predict", json=dict(reversed(list(prediction_payload[0].items()))))
        assert single.status_code == 200
        assert single.json() == {"churn": before["churn"][:1], "classes": {"0": before["classes"]["0"]}}
        assert restarted.get("/model/metrics").json()["latest"]["metrics"] == status["metrics"]


@pytest.mark.parametrize("kind", ["missing", "empty", "header_only", "missing_column", "extra_column", "invalid_value", "missing_target", "one_class", "tiny", "rare_class", "test_without_minority", "malformed"])
def test_bad_csv_returns_structured_error(client, dataset_path, churn_frame, kind):
    if kind == "missing":
        dataset_path.unlink()
    elif kind == "empty":
        dataset_path.write_text("")
    elif kind == "header_only":
        churn_frame.iloc[:0].to_csv(dataset_path, index=False)
    elif kind == "malformed":
        dataset_path.write_text('"unterminated')
    else:
        if kind == "missing_column":
            churn_frame = churn_frame.drop(columns="monthly_fee")
        elif kind == "extra_column":
            churn_frame["extra"] = 1
        elif kind == "invalid_value":
            churn_frame["churn"] = 3
        elif kind == "missing_target":
            churn_frame.loc[0, "churn"] = np.nan
        elif kind == "one_class":
            churn_frame["churn"] = 1
        elif kind == "tiny":
            churn_frame = churn_frame.iloc[:4]
        elif kind in ("rare_class", "test_without_minority"):
            churn_frame["churn"] = 0
            churn_frame.loc[0, "churn"] = 1
            if kind == "test_without_minority":
                churn_frame.loc[1, "churn"] = 1
        churn_frame.to_csv(dataset_path, index=False)
    response = client.post("/model/train", json=CONFIG)
    assert response.status_code == 400, response.text
    assert set(response.json()) == {"code", "message", "details"}
    assert response.json()["code"] == ("EMPTY_DATASET" if kind in ("empty", "header_only") else "DATA_PREPARATION_ERROR")
    assert client.get("/model/status").json()["fitted"] is False
    assert client.get("/model/metrics").json() == {"latest": None, "history": []}


@pytest.mark.parametrize("column", ["monthly_fee", "support_requests", "region", "autopay_enabled"])
@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_features_can_be_imputed(client, dataset_path, churn_frame, prediction_payload, column, all_missing):
    if all_missing:
        churn_frame[column] = np.nan
    else:
        churn_frame.loc[0, column] = np.nan
    churn_frame.to_csv(dataset_path, index=False)
    preview = client.get("/dataset/preview")
    assert preview.status_code == 200
    assert preview.json()[0][column] is None
    response = client.post("/model/train", json=CONFIG)
    assert response.status_code == 200, response.text
    assert client.post("/predict", json=prediction_payload).status_code == 200


@pytest.mark.parametrize("config, status", [
    ({"model_type": "unknown"}, 422),
    ({"model_type": "logreg", "hyperparameters": {"unknown": 1}}, 400),
    ({"model_type": "logreg", "hyperparameters": {"C": -1}}, 400),
    ({"model_type": "random_forest", "hyperparameters": {"n_estimators": 0}}, 400),
])
def test_invalid_config_preserves_current_model(client, dataset_path, prediction_payload, config, status):
    client.post("/model/train", json=CONFIG)
    before = client.post("/predict", json=prediction_payload).json()
    history = client.get("/model/metrics").json()
    response = client.post("/model/train", json=config)
    assert response.status_code == status, response.text
    assert set(response.json()) == {"code", "message", "details"}
    assert client.post("/predict", json=prediction_payload).json() == before
    assert client.get("/model/metrics").json() == history


@pytest.mark.parametrize("field, value", [("monthly_fee", -1), ("monthly_fee", "NaN"), ("usage_hours", "Infinity"), ("autopay_enabled", 2), ("region", "unknown")])
def test_invalid_feature_values(client, prediction_payload, field, value):
    prediction_payload[0][field] = value
    response = client.post("/predict", json=prediction_payload[0])
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_history_filter_without_matches(client, dataset_path):
    assert client.get("/model/metrics?model_type=random_forest").json() == {"latest": None, "history": []}
    client.post("/model/train", json=CONFIG)
    assert client.get("/model/metrics?model_type=random_forest").json() == {"latest": None, "history": []}
    assert client.get("/model/metrics?model_type=logistic_regression").json()["latest"]["model_type"] == "logreg"
    client.post("/model/train", json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 10}})
    history = client.get("/model/metrics?limit=1").json()
    assert len(history["history"]) == 1
    assert history["latest"]["model_type"] == "random_forest"
    assert len(client.get("/model/metrics").json()["history"]) == 2


def test_training_function_without_api(churn_frame):
    model = train_churn_model(churn_frame, TrainingConfigChurn(**CONFIG))
    assert set(model.predict(churn_frame.drop(columns="churn"))) == {0, 1}


def test_imputer_uses_training_data_only(churn_frame, feature_columns):
    frame = churn_frame.drop(columns="churn")
    frame.loc[0, "monthly_fee"] = np.nan
    preprocessor = build_preprocessor(*feature_columns)
    preprocessor.fit(frame.iloc[:30])
    imputer = preprocessor.named_transformers_["num"]["imputer"]
    expected = frame.iloc[:30][feature_columns[0]].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, expected)
    preprocessor.transform(frame.iloc[30:].assign(monthly_fee=1_000_000))
    np.testing.assert_allclose(imputer.statistics_, expected)


def test_prediction_failure_has_common_format(client, dataset_path, prediction_payload, monkeypatch):
    client.post("/model/train", json=CONFIG)
    def fail(_):
        raise ValueError("technical trace must not be returned")
    monkeypatch.setattr(client.app.state.model["model"], "predict", fail)
    response = client.post("/predict", json=prediction_payload)
    assert response.status_code == 500
    assert response.json() == {"code": "MODEL_PREDICTION_ERROR", "message": "Не удалось получить предсказание модели", "details": None}


def test_corrupt_model_does_not_prevent_retraining(dataset_path):
    path = dataset_path.parents[1] / "models" / "churn_model.joblib"
    path.parent.mkdir()
    path.write_bytes(b"broken model")
    with TestClient(create_app()) as client:
        assert client.get("/model/status").json()["fitted"] is False
        assert client.post("/model/train", json=CONFIG).status_code == 200


def test_unexpected_error_has_common_format(dataset_path):
    history = dataset_path.parent / "training_history.json"
    history.write_text("invalid json")
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.get("/model/metrics")
        assert response.status_code == 500
        assert response.json() == {"code": "INTERNAL_ERROR", "message": "Внутренняя ошибка сервиса", "details": None}
        assert client.get("/not-found").json()["code"] == "HTTP_ERROR"
