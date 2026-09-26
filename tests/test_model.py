import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.exceptions import ModelNotFoundError
from src.model.churn_model import (
    build_model,
    load_churn_model,
    load_training_history,
    save_churn_model,
    save_to_history,
)
from src.preprocessing.churn_preprocessor import build_preprocessor, split_data


def test_train_predict_and_persist(churn_frame, feature_columns, training_config):
    model = build_model(build_preprocessor(*feature_columns), **training_config)
    expected_class = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
    }[training_config["model_type"]]
    assert isinstance(model.named_steps["model"], expected_class)
    for key, value in training_config["hyperparameters"].items():
        assert model.named_steps["model"].get_params()[key] == value

    X_train, X_test, y_train, y_test = split_data(churn_frame.drop(columns="churn"), churn_frame["churn"])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    assert predictions.shape == (len(X_test),)
    assert set(predictions) == {0, 1}
    # В синтетических данных failed_payments точно определяет целевой класс.
    assert accuracy_score(y_test, predictions) >= 0.75
    assert probabilities.shape == (len(X_test), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(model.classes_[probabilities.argmax(axis=1)], predictions)

    saved = {"model": model, "feature_names": list(X_train.columns), "config": training_config}
    assert save_churn_model(saved) is True
    loaded = load_churn_model()
    assert loaded["feature_names"] == saved["feature_names"]
    assert loaded["config"] == training_config
    np.testing.assert_array_equal(loaded["model"].predict(X_test), predictions)
    np.testing.assert_allclose(loaded["model"].predict_proba(X_test), probabilities)


def test_unknown_model_type(feature_columns):
    with pytest.raises(ValueError, match="Unknown model type: unsupported"):
        build_model(build_preprocessor(*feature_columns), "unsupported", {})


def test_predict_before_fit(churn_frame, feature_columns, training_config):
    model = build_model(build_preprocessor(*feature_columns), **training_config)

    with pytest.raises(NotFittedError):
        model.predict(churn_frame.drop(columns="churn"))


def test_load_without_saved_model():
    with pytest.raises(ModelNotFoundError, match="Trained churn model not found"):
        load_churn_model()


def test_training_history_preserves_previous_entries():
    assert load_training_history() == []
    first = {"model_type": "logistic_regression", "metrics": {"accuracy": 0.8}}
    second = {"model_type": "random_forest", "metrics": {"accuracy": 0.9}}

    save_to_history(first)
    save_to_history(second)

    assert load_training_history() == [first, second]
