import os
from pathlib import Path

DATASET_PATH = Path(os.getenv("CHURN_DATASET_PATH", "data/churn_dataset.csv"))
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", "models/churn_model.joblib"))
HISTORY_PATH = Path(os.getenv("CHURN_HISTORY_PATH",
                    "data/training_history.json"))

NUMERIC_FEATURES = [
    "monthly_fee", "usage_hours", "support_requests", "account_age_months",
    "failed_payments",
]

CATEGORICAL_FEATURES = ["region", "device_type",
                        "payment_method", "autopay_enabled"]

FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
