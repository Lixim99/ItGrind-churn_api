from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exceptions import DataPreparationError


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])


def split_data(X, y):
    if set(y.dropna().unique()) != {0, 1} or y.isna().any():
        raise DataPreparationError(
            "Для обучения нужны оба класса churn: 0 и 1")

    try:
        result = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        if any(set(part.unique()) != {0, 1} for part in result[2:]):
            raise ValueError("Both splits must contain both churn classes")

        return result
    except ValueError as exc:
        raise DataPreparationError(
            "Недостаточно данных для стратифицированного разбиения train/test",
            {"class_counts": y.value_counts().to_dict()},
        ) from exc
