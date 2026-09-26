from datetime import datetime

import pytest


def test_full_churn_flow(client, dataset_path, churn_frame, training_config, prediction_payload):
    preview = client.get("/dataset/preview")
    assert preview.status_code == 200
    assert preview.json() == churn_frame.head(5).to_dict(orient="records")

    train = client.post("/model/train", json=training_config)
    assert train.status_code == 200, train.text
    metrics = train.json()
    assert set(metrics) == {"accuracy", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in metrics.values())

    status = client.get("/model/status")
    assert status.status_code == 200
    saved = status.json()
    assert saved["fitted"] is True
    assert datetime.fromisoformat(saved["last_fitted"]).tzinfo is not None
    assert saved["metrics"] == metrics
    assert saved["config"] == training_config
    assert set(saved["feature_names"]) == set(churn_frame.columns) - {"churn"}
    assert (dataset_path.parents[1] / "models" / "churn_model.joblib").is_file()

    prediction = client.post("/predict", json=prediction_payload)
    assert prediction.status_code == 200, prediction.text
    body = prediction.json()
    assert len(body["churn"]) == len(prediction_payload)
    assert set(body["churn"]) <= {0, 1}
    assert set(body["classes"]) == {str(i) for i in range(len(prediction_payload))}
    for index, label in enumerate(body["churn"]):
        probabilities = body["classes"][str(index)]
        assert set(probabilities) == {"0", "1"}
        assert all(0 <= value <= 1 for value in probabilities.values())
        assert sum(probabilities.values()) == pytest.approx(1, abs=0.011)
        assert probabilities[str(label)] == max(probabilities.values())

    repeated = client.post("/predict", json=prediction_payload)
    assert repeated.status_code == 200
    assert repeated.json() == body

    history = client.get("/model/metrics")
    assert history.status_code == 200
    entry = {"timestamp": saved["last_fitted"], **training_config, "metrics": metrics}
    assert history.json() == {"latest": entry, "history": [entry]}


def test_missing_model_returns_404(client, prediction_payload):
    response = client.post("/predict", json=prediction_payload)

    assert response.status_code == 404
    assert response.json() == {
        "code": "MODEL_NOT_FOUND",
        "message": "Обученная модель отсутствует",
        "details": None,
    }


@pytest.mark.parametrize("invalid_kind", ["missing_feature", "invalid_type", "extra_feature", "empty_list", "not_an_object"])
def test_invalid_prediction_request(client, prediction_payload, invalid_kind):
    if invalid_kind == "missing_feature":
        del prediction_payload[0]["monthly_fee"]
    elif invalid_kind == "invalid_type":
        prediction_payload[0]["monthly_fee"] = "not-a-number"
    elif invalid_kind == "extra_feature":
        prediction_payload[0]["unexpected"] = 1
    elif invalid_kind == "empty_list":
        prediction_payload = []
    else:
        prediction_payload = "invalid"

    response = client.post("/predict", json=prediction_payload)

    assert response.status_code == 422
    error = response.json()
    assert error["code"] == "VALIDATION_ERROR"
    assert error["message"] == "Некорректные входные данные"
    assert error["details"]
    if invalid_kind in ("missing_feature", "invalid_type"):
        assert any(item["loc"][-2:] == [0, "monthly_fee"] for item in error["details"])


@pytest.mark.parametrize("payload", [{}, {"model_type": "logistic_regression", "hyperparameters": []}])
def test_invalid_training_request(client, payload, isolated_workspace):
    response = client.post("/model/train", json=payload)

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert response.json()["details"]
    assert not (isolated_workspace / "models" / "churn_model.joblib").exists()
    assert not (isolated_workspace / "data" / "training_history.json").exists()
