import os
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\tusha\Documents\AquaSense"
)

# Real historical dataset
INPUT_FILE = (
    PROJECT_ROOT
    / "ml_outputs"
    / "water_quality_hourly_regularized.csv"
)

# Training feature list
FEATURE_FILE = (
    PROJECT_ROOT
    / "ml_outputs"
    / "prepared_data"
    / "feature_list.csv"
)

# Models
MODEL_DIR = PROJECT_ROOT / "models"

# Output
OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_outputs"
    / "simulation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

PARAMETERS = [
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "do_saturation",
    "turbidity"
]

LAGS = [1, 3, 6, 12, 24]

CHANGE_PERIODS = [3, 6, 12, 24]

ROLLING_WINDOWS = [6, 12, 24]

# Thresholds from your walk-forward analysis
RISK_THRESHOLDS = {
    "1h": 0.25,
    "6h": 0.20,
    "12h": 0.15,
    "24h": 0.15
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AQUASENSE - ML SIMULATION ENGINE")
print("=" * 70)


# ============================================================
# LOAD FEATURE LIST
# ============================================================

print("\nLoading training feature list...")

feature_list = pd.read_csv(
    FEATURE_FILE
)

feature_columns = (
    feature_list["feature"]
    .tolist()
)

print(
    f"Original training features: "
    f"{len(feature_columns)}"
)


# ============================================================
# LOAD IMPUTER
# ============================================================

print("\nLoading median imputer...")

imputer = joblib.load(
    MODEL_DIR / "median_imputer.joblib"
)

print("Imputer loaded.")


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading XGBoost models...")

models = {}

for horizon in ["1h", "6h", "12h", "24h"]:

    model = XGBClassifier()

    model.load_model(
        MODEL_DIR / f"xgboost_{horizon}.json"
    )

    models[horizon] = model

    print(
        f"Loaded XGBoost {horizon}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading historical water-quality data...")

df = pd.read_csv(
    INPUT_FILE
)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    utc=True
)

df = df.sort_values(
    ["segment_id", "timestamp_utc"]
).reset_index(drop=True)

print(
    f"Rows loaded: {len(df):,}"
)

print(
    f"Segments: {df['segment_id'].nunique()}"
)


# ============================================================
# TIME FEATURES
# ============================================================

print("\nCreating time features...")

df["hour"] = (
    df["timestamp_utc"].dt.hour
)

df["day_of_week"] = (
    df["timestamp_utc"].dt.dayofweek
)

df["month"] = (
    df["timestamp_utc"].dt.month
)

df["day_of_year"] = (
    df["timestamp_utc"].dt.dayofyear
)

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)


# ============================================================
# LAG FEATURES
# ============================================================

print("Creating lag features...")

for parameter in PARAMETERS:

    for lag in LAGS:

        feature_name = (
            f"{parameter}_lag_{lag}h"
        )

        df[feature_name] = (
            df.groupby("segment_id")[parameter]
            .shift(lag)
        )


# ============================================================
# CHANGE FEATURES
# ============================================================

print("Creating change features...")

for parameter in PARAMETERS:

    for period in CHANGE_PERIODS:

        feature_name = (
            f"{parameter}_change_{period}h"
        )

        previous_value = (
            df.groupby("segment_id")[parameter]
            .shift(period)
        )

        df[feature_name] = (
            df[parameter] - previous_value
        )


# ============================================================
# ROLLING FEATURES
# ============================================================

print("Creating rolling features...")

for parameter in PARAMETERS:

    for window in ROLLING_WINDOWS:

        group = (
            df.groupby("segment_id")[parameter]
        )

        shifted = group.shift(1)

        rolling_mean = (
            shifted
            .groupby(df["segment_id"])
            .transform(
                lambda x:
                x.rolling(
                    window,
                    min_periods=window
                ).mean()
            )
        )

        rolling_max = (
            shifted
            .groupby(df["segment_id"])
            .transform(
                lambda x:
                x.rolling(
                    window,
                    min_periods=window
                ).max()
            )
        )

        rolling_std = (
            shifted
            .groupby(df["segment_id"])
            .transform(
                lambda x:
                x.rolling(
                    window,
                    min_periods=window
                ).std()
            )
        )

        df[
            f"{parameter}_rolling_mean_{window}h"
        ] = rolling_mean

        df[
            f"{parameter}_rolling_max_{window}h"
        ] = rolling_max

        df[
            f"{parameter}_rolling_std_{window}h"
        ] = rolling_std


# ============================================================
# TURBIDITY FEATURES
# ============================================================

print("Creating turbidity features...")

df["turbidity_change_1h"] = (
    df["turbidity"]
    -
    df.groupby("segment_id")["turbidity"]
    .shift(1)
)

previous_turbidity = (
    df.groupby("segment_id")["turbidity"]
    .shift(1)
)

df["turbidity_ratio_1h"] = (
    df["turbidity"]
    /
    previous_turbidity.replace(
        0,
        np.nan
    )
)


# ============================================================
# SELECT EXACT TRAINING FEATURES
# ============================================================

print("\nSelecting exact training features...")

X = df[feature_columns].copy()

print(
    f"Feature matrix before imputation: "
    f"{X.shape}"
)


# ============================================================
# APPLY SAVED IMPUTER
# ============================================================

print("\nApplying saved training imputer...")

X_processed = imputer.transform(
    X
)

print(
    f"Feature matrix after imputation: "
    f"{X_processed.shape}"
)


# ============================================================
# VERIFY FEATURE COUNT
# ============================================================

expected_features = 241

if X_processed.shape[1] != expected_features:

    raise ValueError(
        f"Expected {expected_features} "
        f"features, but got "
        f"{X_processed.shape[1]}"
    )

print(
    f"Feature count check: "
    f"{X_processed.shape[1]} PASSED"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating ML predictions...")

for horizon, model in models.items():

    probabilities = (
        model.predict_proba(
            X_processed
        )[:, 1]
    )

    df[
        f"risk_probability_{horizon}"
    ] = probabilities

    threshold = (
        RISK_THRESHOLDS[horizon]
    )

    df[
        f"risk_flag_{horizon}"
    ] = (
        probabilities >= threshold
    )


# ============================================================
# DETERMINE EARLIEST RISK HORIZON
# ============================================================

print(
    "\nDetermining earliest predicted "
    "risk horizon..."
)


def determine_risk_horizon(row):

    # Check shortest horizon first

    if row["risk_flag_1h"]:
        return "Within 1 hour"

    if row["risk_flag_6h"]:
        return "Within 6 hours"

    if row["risk_flag_12h"]:
        return "Within 12 hours"

    if row["risk_flag_24h"]:
        return "Within 24 hours"

    return "No significant risk"


df["predicted_risk_horizon"] = (
    df.apply(
        determine_risk_horizon,
        axis=1
    )
)


# ============================================================
# RISK LEVEL
# ============================================================

def determine_risk_level(row):

    probability = (
        row["risk_probability_12h"]
    )

    if probability >= 0.50:
        return "HIGH"

    elif probability >= 0.25:
        return "MEDIUM"

    else:
        return "LOW"


df["risk_level"] = (
    df.apply(
        determine_risk_level,
        axis=1
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    OUTPUT_DIR
    / "simulation_predictions.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY RECENT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LATEST SIMULATION RESULTS")
print("=" * 70)

display_columns = [
    "timestamp_utc",
    "ph",
    "turbidity",
    "do_concentration",
    "sp_conductance",
    "risk_probability_1h",
    "risk_probability_6h",
    "risk_probability_12h",
    "risk_probability_24h",
    "predicted_risk_horizon",
    "risk_level"
]

print(
    df[
        display_columns
    ].tail(20).to_string(
        index=False
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SIMULATION ENGINE COMPLETED")
print("=" * 70)

print("\nSaved prediction file:")

print(output_file)

print("\nNext step:")
print(
    "Use the prediction output to build "
    "the trend-analysis and AI report."
)