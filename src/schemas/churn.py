from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]

Region = Literal["europe", "asia", "america", "africa"]
Device = Literal["mobile", "desktop", "tablet"]
Payment = Literal["card", "paypal", "crypto"]
ModelType = Literal["logistic_regression", "logreg", "random_forest"]

FEATURE_EXAMPLE = {
    "monthly_fee": 19.99, "usage_hours": 21.48, "support_requests": 2,
    "account_age_months": 12, "failed_payments": 0, "region": "europe",
    "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1,
}


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [FEATURE_EXAMPLE]}
    )

    monthly_fee: NonNegativeFloat
    usage_hours: NonNegativeFloat
    support_requests: NonNegativeInt
    account_age_months: NonNegativeInt
    failed_payments: NonNegativeInt
    region: Region
    device_type: Device
    payment_method: Payment
    autopay_enabled: Literal[0, 1]


class DatasetRowChurn(FeatureVectorChurn):
    monthly_fee: NonNegativeFloat | None
    usage_hours: NonNegativeFloat | None
    support_requests: NonNegativeInt | None
    account_age_months: NonNegativeInt | None
    failed_payments: NonNegativeInt | None
    region: Region | None
    device_type: Device | None
    payment_method: Payment | None
    autopay_enabled: Literal[0, 1] | None
    churn: Literal[0, 1]


class PredictionResponseChurn(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"churn": [0], "classes": {"0": {"0": 0.8, "1": 0.2}}}
    ]})

    churn: list[int]
    classes: dict[int, dict[int, float]]


class TrainingConfigChurn(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"model_type": "logistic_regression", "hyperparameters": {"max_iter": 1000}}
    ]})

    model_type: ModelType
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
