import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# ============================================================
# AQUASENSE - HISTORICAL EVENT SIMULATION
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

DATA_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "water_quality_hourly_regularized.csv"
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "simulation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# EVENT TO REPLAY
# ------------------------------------------------------------

EVENT_START = pd.Timestamp("2022-09-26 00:00:00", tz="UTC")
EVENT_END = pd.Timestamp("2022-09-29 23:00:00", tz="UTC")

# We need historical data before the event because our
# features use lags and rolling windows up to 24 hours.
CONTEXT_HOURS = 48

SIMULATION_START = EVENT_START - pd.Timedelta(hours=CONTEXT_HOURS)


# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

HORIZONS = ["1h", "6h", "12h", "24h"]

RISK_THRESHOLDS = {
    "1h": 0.25,
    "6h": 0.20,
    "12h": 0.15,
    "24h": 0.15
}


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def create_features(df):
    """
    Recreate the exact feature-engineering logic used during
    model training.

    Only lagged/rolling historical information is used.
    """

    df = df.copy()

    parameters = [
        "water_temperature",
        "sp_conductance",
        "ph",
        "do_concentration",
        "do_saturation",
        "turbidity"
    ]

    # --------------------------------------------------------
    # Time features
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

    LAGS = [1, 3, 6, 12, 24]

    for parameter in parameters:

        for lag in LAGS:

            df[f"{parameter}_lag_{lag}h"] = (
                df.groupby("segment_id")[parameter]
                .shift(lag)
            )


    # --------------------------------------------------------
    # Change features
    # --------------------------------------------------------

    CHANGE_PERIODS = [3, 6, 12, 24]

    for parameter in parameters:

        for period in CHANGE_PERIODS:

            previous = (
                df.groupby("segment_id")[parameter]
                .shift(period)
            )

            df[f"{parameter}_change_{period}h"] = (
                df[parameter] - previous
            )


    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    ROLLING_WINDOWS = [6, 12, 24]

    for parameter in parameters:

        # Shift first so the current measurement is not used
        # inside its own rolling historical window.
        shifted = (
            df.groupby("segment_id")[parameter]
            .shift(1)
        )

        for window in ROLLING_WINDOWS:

            df[f"{parameter}_rolling_mean_{window}h"] = (
                shifted.groupby(df["segment_id"])
                .rolling(window)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"{parameter}_rolling_max_{window}h"] = (
                shifted.groupby(df["segment_id"])
                .rolling(window)
                .max()
                .reset_index(level=0, drop=True)
            )

            df[f"{parameter}_rolling_std_{window}h"] = (
                shifted.groupby(df["segment_id"])
                .rolling(window)
                .std()
                .reset_index(level=0, drop=True)
            )


    # --------------------------------------------------------
    # Turbidity-specific features
    # --------------------------------------------------------

    turbidity_previous = (
        df.groupby("segment_id")["turbidity"]
        .shift(1)
    )

    df["turbidity_change_1h"] = (
        df["turbidity"] - turbidity_previous
    )

    df["turbidity_ratio_1h"] = (
        df["turbidity"] /
        turbidity_previous.replace(0, np.nan)
    )


    return df


def calculate_trend(df, parameter):

    """
    Calculate simple recent trend using the latest available
    value and the value 6 hours earlier.
    """

    if len(df) < 7:
        return 0.0

    latest = df.iloc[-1][parameter]
    previous = df.iloc[-7][parameter]

    if pd.isna(latest) or pd.isna(previous):
        return 0.0

    return float(latest - previous)


def trend_direction(value):

    if value > 0:
        return "INCREASING"

    if value < 0:
        return "DECREASING"

    return "STABLE"


def determine_risk_horizon(row):

    for horizon in HORIZONS:

        probability = row[f"risk_probability_{horizon}"]

        threshold = RISK_THRESHOLDS[horizon]

        if probability >= threshold:
            return horizon

    return "NO SIGNIFICANT RISK"


def determine_risk_level(probability):

    if probability >= 0.50:
        return "HIGH"

    if probability >= 0.25:
        return "MEDIUM"

    return "LOW"


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("AQUASENSE - HISTORICAL EVENT SIMULATION")
print("=" * 70)


# ------------------------------------------------------------
# LOAD FEATURE LIST
# ------------------------------------------------------------

print("\nLoading training feature list...")

feature_columns = pd.read_csv(
    FEATURE_LIST_FILE
)["feature"].tolist()

print("Training features:", len(feature_columns))


# ------------------------------------------------------------
# LOAD IMPUTER
# ------------------------------------------------------------

print("\nLoading saved training imputer...")

imputer = joblib.load(IMPUTER_FILE)

print("Imputer loaded.")


# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

print("\nLoading XGBoost models...")

models = {}

for horizon in HORIZONS:

    model_path = os.path.join(
        MODEL_DIR,
        f"xgboost_{horizon}.json"
    )

    model = XGBClassifier()

    model.load_model(model_path)

    models[horizon] = model

    print(f"Loaded XGBoost {horizon}")


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading historical data...")

df = pd.read_csv(DATA_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp_utc"],
    utc=True
)

df = df.sort_values(
    ["segment_id", "timestamp_utc"]
).reset_index(drop=True)

print("Total rows:", len(df))


# ------------------------------------------------------------
# SELECT EVENT WINDOW + CONTEXT
# ------------------------------------------------------------

print("\nSelecting historical event window...")

simulation_df = df[
    (df["timestamp"] >= SIMULATION_START) &
    (df["timestamp"] <= EVENT_END)
].copy()

print(
    "Simulation data:",
    simulation_df["timestamp"].min(),
    "to",
    simulation_df["timestamp"].max()
)

print(
    "Rows:",
    len(simulation_df)
)


# ------------------------------------------------------------
# CREATE FEATURES
# ------------------------------------------------------------

print("\nCreating historical features...")

simulation_features = create_features(
    simulation_df
)

print("Feature engineering completed.")


# ------------------------------------------------------------
# SELECT EXACT TRAINING FEATURES
# ------------------------------------------------------------

print("\nSelecting exact training features...")

X = simulation_features[
    feature_columns
].copy()

print(
    "Feature matrix:",
    X.shape
)


# ------------------------------------------------------------
# IMPUTE
# ------------------------------------------------------------

print("\nApplying saved imputer...")

X_processed = imputer.transform(X)

print(
    "Processed feature matrix:",
    X_processed.shape
)


# ------------------------------------------------------------
# GENERATE PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating historical predictions...")

for horizon in HORIZONS:

    model = models[horizon]

    probabilities = model.predict_proba(
        X_processed
    )[:, 1]

    simulation_features[
        f"risk_probability_{horizon}"
    ] = probabilities


# ------------------------------------------------------------
# RISK HORIZON
# ------------------------------------------------------------

print("\nDetermining risk horizon...")

simulation_features["predicted_risk_horizon"] = (
    simulation_features.apply(
        determine_risk_horizon,
        axis=1
    )
)


# ------------------------------------------------------------
# RISK LEVEL
# ------------------------------------------------------------

simulation_features["risk_level"] = (
    simulation_features[
        "risk_probability_12h"
    ].apply(
        determine_risk_level
    )
)

# ------------------------------------------------------------
# TREND ANALYSIS
# ------------------------------------------------------------

print("\nCalculating water-quality trends...")

parameters = [
    "turbidity",
    "ph",
    "do_concentration",
    "sp_conductance",
    "water_temperature"
]

for parameter in parameters:

    trend_column = f"{parameter}_trend_6h"

    previous_value = (
        simulation_features
        .groupby("segment_id")[parameter]
        .shift(6)
    )

    simulation_features[trend_column] = (
        simulation_features[parameter] - previous_value
    )

print("Trend analysis completed.")


# ------------------------------------------------------------
# ACTUAL EVENT LABEL
# ------------------------------------------------------------

# The research threshold is 319.81 FNU.

TURBIDITY_THRESHOLD = 319.81

simulation_features["actual_high_turbidity"] = (
    simulation_features["turbidity"]
    >= TURBIDITY_THRESHOLD
)


# ------------------------------------------------------------
# SAVE COMPLETE SIMULATION
# ------------------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    "historical_event_2022_simulation.csv"
)

simulation_features.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY EVENT REPLAY
# ============================================================

display_columns = [
    "timestamp",
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "turbidity",
    "risk_probability_1h",
    "risk_probability_6h",
    "risk_probability_12h",
    "risk_probability_24h",
    "predicted_risk_horizon",
    "risk_level",
    "actual_high_turbidity"
]

display_df = simulation_features[
    (simulation_features["timestamp"] >= EVENT_START) &
    (simulation_features["timestamp"] <= EVENT_END)
][display_columns].copy()


print("\n")
print("=" * 70)
print("HISTORICAL EVENT REPLAY")
print("=" * 70)

pd.set_option(
    "display.max_rows",
    100
)

pd.set_option(
    "display.max_columns",
    30
)

print(
    display_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# FIND FIRST WARNING
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("FIRST AI WARNING")
print("=" * 70)

warning_rows = display_df[
    display_df["predicted_risk_horizon"]
    != "NO SIGNIFICANT RISK"
]

if len(warning_rows) > 0:

    first_warning = warning_rows.iloc[0]

    print(
        "First warning:",
        first_warning["timestamp"]
    )

    print(
        "Predicted horizon:",
        first_warning["predicted_risk_horizon"]
    )

    print(
        "12h risk probability:",
        round(
            first_warning["risk_probability_12h"] * 100,
            2
        ),
        "%"
    )

    print(
        "Risk level:",
        first_warning["risk_level"]
    )

else:

    print(
        "No warning was generated in this replay window."
    )


# ------------------------------------------------------------
# FIND ACTUAL EVENT
# ------------------------------------------------------------

actual_event_rows = display_df[
    display_df["actual_high_turbidity"]
]

print("\n")
print("=" * 70)
print("ACTUAL HIGH-TURBIDITY EVENT")
print("=" * 70)

if len(actual_event_rows) > 0:

    first_event = actual_event_rows.iloc[0]

    print(
        "Event begins:",
        first_event["timestamp"]
    )

    print(
        "Turbidity:",
        round(
            first_event["turbidity"],
            2
        ),
        "FNU"
    )

else:

    print(
        "No threshold crossing found in this window."
    )


print("\n")
print("=" * 70)
print("SIMULATION COMPLETED")
print("=" * 70)

print("\nSaved:")
print(output_file)