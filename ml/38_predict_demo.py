import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "demo",
    "water_quality_demo_48h.csv"
)

FEATURE_LIST_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "prepared_data",
    "feature_list.csv"
)

IMPUTER_FILE = os.path.join(
    BASE_DIR,
    "models",
    "median_imputer.joblib"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


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

RISK_THRESHOLDS = {
    "1h": 0.25,
    "6h": 0.20,
    "12h": 0.15,
    "24h": 0.15
}


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Make sure timestamp exists
    # --------------------------------------------------------

    if "timestamp" not in df.columns:

        if "timestamp_utc" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp_utc"],
                utc=True
            )

        else:
            raise ValueError(
                "CSV must contain either 'timestamp' "
                "or 'timestamp_utc'."
            )

    else:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # Treat uploaded data as one continuous segment
    # --------------------------------------------------------

    df["segment_id"] = 0

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["month"] = df["timestamp"].dt.month

    df["day_of_year"] = df["timestamp"].dt.dayofyear

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

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for parameter in PARAMETERS:

        for lag in LAGS:

            column_name = f"{parameter}_lag_{lag}h"

            df[column_name] = (
                df.groupby("segment_id")[parameter]
                .shift(lag)
            )

    # --------------------------------------------------------
    # Change features
    # --------------------------------------------------------

    for parameter in PARAMETERS:

        for period in CHANGE_PERIODS:

            column_name = f"{parameter}_change_{period}h"

            previous_value = (
                df.groupby("segment_id")[parameter]
                .shift(period)
            )

            df[column_name] = (
                df[parameter] - previous_value
            )

    # --------------------------------------------------------
    # Rolling features
    #
    # shift(1) is important:
    # the current measurement is NOT included.
    # --------------------------------------------------------

    for parameter in PARAMETERS:

        for window in ROLLING_WINDOWS:

            shifted = (
                df.groupby("segment_id")[parameter]
                .shift(1)
            )

            rolling_group = (
                shifted.groupby(df["segment_id"])
                .rolling(window=window, min_periods=1)
            )

            df[
                f"{parameter}_rolling_mean_{window}h"
            ] = (
                rolling_group.mean()
                .reset_index(level=0, drop=True)
            )

            df[
                f"{parameter}_rolling_max_{window}h"
            ] = (
                rolling_group.max()
                .reset_index(level=0, drop=True)
            )

            df[
                f"{parameter}_rolling_std_{window}h"
            ] = (
                rolling_group.std()
                .reset_index(level=0, drop=True)
            )

    # --------------------------------------------------------
    # Turbidity short-term features
    # --------------------------------------------------------

    df["turbidity_change_1h"] = (
        df["turbidity"]
        - df.groupby("segment_id")["turbidity"].shift(1)
    )

    previous_turbidity = (
        df.groupby("segment_id")["turbidity"]
        .shift(1)
    )

    df["turbidity_ratio_1h"] = np.where(
        previous_turbidity != 0,
        df["turbidity"] / previous_turbidity,
        np.nan
    )

    return df


# ============================================================
# LOAD INPUT DATA
# ============================================================

print("=" * 70)
print("AquaSense - AI Water Quality Prediction")
print("=" * 70)

print("\nLoading demo data...")

df = pd.read_csv(INPUT_FILE)

print(f"Input file: {INPUT_FILE}")
print(f"Rows loaded: {len(df)}")


# ============================================================
# CREATE FEATURES
# ============================================================

print("\nCreating model features...")

feature_df = create_features(df)

print(f"Total columns after feature engineering: {len(feature_df.columns)}")


# ============================================================
# LOAD TRAINING FEATURE LIST
# ============================================================

print("\nLoading training feature list...")

feature_list_df = pd.read_csv(FEATURE_LIST_FILE)

# Handle either possible column name
if "feature" in feature_list_df.columns:
    feature_list = feature_list_df["feature"].tolist()
elif "feature_name" in feature_list_df.columns:
    feature_list = feature_list_df["feature_name"].tolist()
else:
    feature_list = feature_list_df.iloc[:, 0].tolist()

print(f"Training feature-list entries: {len(feature_list)}")


# ============================================================
# REMOVE TARGET / LABEL COLUMNS
# ============================================================

TARGET_COLUMNS = [
    "high_turbidity",
    "risk_1h",
    "risk_6h",
    "risk_12h",
    "risk_24h"
]

feature_list = [
    feature
    for feature in feature_list
    if feature not in TARGET_COLUMNS
]

print(
    f"Actual input features after removing target columns: "
    f"{len(feature_list)}"
)


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

missing_features = [
    feature
    for feature in feature_list
    if feature not in feature_df.columns
]

if missing_features:

    print("\nERROR: Missing required input features:")

    for feature in missing_features:
        print("  -", feature)

    raise ValueError(
        f"{len(missing_features)} required input features are missing."
    )

if missing_features:

    print("\nERROR: Missing required features:")

    for feature in missing_features:
        print("  -", feature)

    raise ValueError(
        f"{len(missing_features)} required features are missing."
    )

# ============================================================
# RESTORE TRAINING FEATURE ORDER
# ============================================================

prediction_feature_list = feature_list.copy()

if "high_turbidity" not in prediction_feature_list:
    prediction_feature_list.append("high_turbidity")


# ============================================================
# CREATE UNKNOWN TARGET COLUMN
# ============================================================

# high_turbidity is part of the saved training preprocessing
# pipeline, but its value is unknown during a real prediction.
#
# Therefore we create it as NaN.
# The trained imputer will handle the missing value.

if "high_turbidity" not in feature_df.columns:
    feature_df["high_turbidity"] = np.nan


# ============================================================
# SELECT FEATURES IN EXACT TRAINING ORDER
# ============================================================

X = feature_df[prediction_feature_list].copy()

print(f"\nFeature matrix shape: {X.shape}")

# ============================================================
# LOAD IMPUTER
# ============================================================

print("\nLoading training imputer...")

imputer = joblib.load(IMPUTER_FILE)


# ============================================================
# MATCH EXACT TRAINING FEATURE ORDER
# ============================================================

expected_features = list(imputer.feature_names_in_)

print(
    f"Imputer expects {len(expected_features)} features."
)

# Check for features missing from the demo feature matrix
missing_for_imputer = [
    feature
    for feature in expected_features
    if feature not in X.columns
]

if missing_for_imputer:

    print("\nERROR: Features expected by the imputer are missing:")

    for feature in missing_for_imputer:
        print("  -", feature)

    raise ValueError(
        "Demo data does not contain all features required "
        "by the training preprocessing pipeline."
    )

# Reorder EXACTLY like training
X = X[expected_features]

print(
    f"Feature matrix reordered to training order: {X.shape}"
)


# ============================================================
# APPLY TRAINING IMPUTER
# ============================================================

X_processed = imputer.transform(X)

print(
    f"Processed feature matrix shape: {X_processed.shape}"
)

print(
    f"Processed feature matrix shape: {X_processed.shape}"
)


# ============================================================
# LOAD MODELS
# ============================================================

models = {}

for horizon in ["1h", "6h", "12h", "24h"]:

    model_file = os.path.join(
        MODEL_DIR,
        f"xgboost_{horizon}.json"
    )

    print(f"\nLoading {horizon} model...")

    model = XGBClassifier()

    model.load_model(model_file)

    models[horizon] = model


# ============================================================
# PREDICT ONLY THE LATEST ROW
# ============================================================

latest_index = len(feature_df) - 1

latest_features = X_processed[latest_index:latest_index + 1]

latest_timestamp = feature_df.iloc[
    latest_index
]["timestamp"]


# ============================================================
# CURRENT WATER QUALITY
# ============================================================

latest_row = feature_df.iloc[latest_index]


print("\n")
print("=" * 70)
print("LATEST OBSERVATION")
print("=" * 70)

print(
    f"Timestamp:          "
    f"{latest_timestamp}"
)

print(
    f"Water temperature:  "
    f"{latest_row['water_temperature']:.2f}"
)

print(
    f"Conductance:        "
    f"{latest_row['sp_conductance']:.4f}"
)

print(
    f"pH:                 "
    f"{latest_row['ph']:.2f}"
)

print(
    f"DO concentration:   "
    f"{latest_row['do_concentration']:.2f}"
)

print(
    f"DO saturation:      "
    f"{latest_row['do_saturation']:.2f}"
)

print(
    f"Turbidity:          "
    f"{latest_row['turbidity']:.2f} FNU"
)


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = {}

print("\n")
print("=" * 70)
print("AI FUTURE RISK PREDICTION")
print("=" * 70)

for horizon in ["1h", "6h", "12h", "24h"]:

    model = models[horizon]

    probability = model.predict_proba(
        latest_features
    )[0, 1]

    probabilities[horizon] = float(probability)

    print(
        f"{horizon:>3} future risk: "
        f"{probability * 100:6.2f}%"
    )


# ============================================================
# DETERMINE RISK HORIZON
# ============================================================

risk_horizon = "No actionable warning"

for horizon in ["1h", "6h", "12h", "24h"]:

    if probabilities[horizon] >= RISK_THRESHOLDS[horizon]:

        risk_horizon = horizon
        break


# ============================================================
# DETERMINE OVERALL RISK LEVEL
# ============================================================

risk_12h = probabilities["12h"]

if risk_12h >= 0.50:

    risk_level = "HIGH"

elif risk_12h >= 0.25:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("AQUASENSE AI ASSESSMENT")
print("=" * 70)

print(f"Risk level:        {risk_level}")

print(
    f"Risk horizon:      {risk_horizon}"
)

if risk_horizon == "No actionable warning":

    print(
        "Prediction:        No actionable high-turbidity "
        "risk detected."
    )

else:

    print(
        "Prediction:        Future high-turbidity risk "
        "detected."
    )

print("=" * 70)


# ============================================================
# SAVE RESULT
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "demo"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

result = pd.DataFrame([{
    "timestamp": latest_timestamp,
    "water_temperature": latest_row["water_temperature"],
    "sp_conductance": latest_row["sp_conductance"],
    "ph": latest_row["ph"],
    "do_concentration": latest_row["do_concentration"],
    "do_saturation": latest_row["do_saturation"],
    "turbidity": latest_row["turbidity"],
    "risk_1h": probabilities["1h"],
    "risk_6h": probabilities["6h"],
    "risk_12h": probabilities["12h"],
    "risk_24h": probabilities["24h"],
    "risk_horizon": risk_horizon,
    "risk_level": risk_level
}])

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "demo_prediction_result.csv"
)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nPrediction saved to:\n{OUTPUT_FILE}"
)