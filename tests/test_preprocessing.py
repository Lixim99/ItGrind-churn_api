import numpy as np
import pandas as pd
import pytest

from src.preprocessing.churn_preprocessor import build_preprocessor, split_data


def test_split_is_stratified_disjoint_and_reproducible(churn_frame):
    X, y = churn_frame.drop(columns="churn"), churn_frame["churn"]
    first = split_data(X, y)
    X_train, X_test, y_train, y_test = first

    assert (len(X_train), len(X_test)) == (32, 8)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_test.index) == set(X.index)
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert y_train.value_counts().to_dict() == {0: 16, 1: 16}
    assert y_test.value_counts().to_dict() == {0: 4, 1: 4}
    for actual, repeated in zip(first, split_data(X, y)):
        if isinstance(actual, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, repeated)
        else:
            pd.testing.assert_series_equal(actual, repeated)


def test_preprocessor_scales_and_encodes(churn_frame, feature_columns):
    numeric, categorical = feature_columns
    X = churn_frame.drop(columns="churn")
    preprocessor = build_preprocessor(numeric, categorical)

    transformed = preprocessor.fit_transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert transformed.shape == (len(X), len(numeric) + sum(X[c].nunique() for c in categorical))
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed[:, :len(numeric)].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(transformed[:, :len(numeric)].std(axis=0), 1)
    encoded = transformed[:, len(numeric):]
    assert set(np.unique(encoded)) == {0, 1}
    np.testing.assert_array_equal(encoded.sum(axis=1), len(categorical))


def test_preprocessor_accepts_unknown_categories(churn_frame, feature_columns):
    numeric, categorical = feature_columns
    X = churn_frame.drop(columns="churn")
    preprocessor = build_preprocessor(numeric, categorical)
    preprocessor.fit(X)
    unseen = X.iloc[:1].copy()
    unseen[categorical[:3]] = "unseen-category"
    unseen["autopay_enabled"] = 2

    transformed = preprocessor.transform(unseen)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(transformed[:, len(numeric):], 0)


def test_scaler_uses_only_training_data(churn_frame, feature_columns):
    numeric, categorical = feature_columns
    X_train, X_test, _, _ = split_data(churn_frame.drop(columns="churn"), churn_frame["churn"])
    preprocessor = build_preprocessor(numeric, categorical)
    preprocessor.fit(X_train)
    X_test = X_test.copy()
    X_test[numeric] = 1_000_000
    preprocessor.transform(X_test)

    np.testing.assert_allclose(preprocessor.named_transformers_["num"]["scaler"].mean_, X_train[numeric].mean())


def test_preprocessor_rejects_missing_feature(churn_frame, feature_columns):
    preprocessor = build_preprocessor(*feature_columns)
    preprocessor.fit(churn_frame.drop(columns="churn"))

    with pytest.raises(ValueError, match="monthly_fee"):
        preprocessor.transform(churn_frame.drop(columns=["churn", "monthly_fee"]))
