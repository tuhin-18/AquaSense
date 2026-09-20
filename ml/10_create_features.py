import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\tusha\Documents\AquaSense")

INPUT_FILE = (
    PROJECT_ROOT
    / "ml_outputs"
    / "future_risk_labels.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml_outputs"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ml_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Water-quality parameters
PARAMETERS = [
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "do_saturation",
    "turbidity"
]

# Lag periods in hours
LAGS = [1, 3, 6, 12, 24]

# Change periods
CHANGE_PERIODS = [3, 6, 12, 24]

# Rolling-window periods
ROLLING_WINDOWS = [6, 12, 24]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 10 - CREATE TIME-SERIES FEATURES")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    utc=True
)

df = df.sort_values(
    ["segment_id", "timestamp_utc"]
).reset_index(drop=True)

print(f"Rows loaded: {len(df):,}")
print(f"Segments: {df['segment_id'].nunique()}")


# ============================================================
# CREATE TIME FEATURES
# ============================================================

print("\nCreating calendar/time features...")

df["hour"] = df["timestamp_utc"].dt.hour
df["day_of_week"] = df["timestamp_utc"].dt.dayofweek
df["month"] = df["timestamp_utc"].dt.month
df["day_of_year"] = df["timestamp_utc"].dt.dayofyear

# Cyclic representation of hour
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

# Cyclic representation of month
df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)


# ============================================================
# CREATE LAG FEATURES
# ============================================================

print("\nCreating lag features...")

for parameter in PARAMETERS:

    for lag in LAGS:

        feature_name = f"{parameter}_lag_{lag}h"

        df[feature_name] = (
            df.groupby("segment_id")[parameter]
            .shift(lag)
        )


# ============================================================
# CREATE CHANGE FEATURES
# ============================================================

print("\nCreating change/trend features...")

for parameter in PARAMETERS:

    for period in CHANGE_PERIODS:

        feature_name = f"{parameter}_change_{period}h"

        current_value = df[parameter]

        previous_value = (
            df.groupby("segment_id")[parameter]
            .shift(period)
        )

        df[feature_name] = (
            current_value - previous_value
        )


# ============================================================
# CREATE ROLLING FEATURES
# ============================================================

print("\nCreating rolling statistics...")

for parameter in PARAMETERS:

    for window in ROLLING_WINDOWS:

        group = df.groupby("segment_id")[parameter]

        # Shift by one hour first.
        # This ensures the rolling statistics
        # do not use future information.

        shifted = group.shift(1)

        rolling_mean = (
            shifted
            .groupby(df["segment_id"])
            .transform(
                lambda x: x.rolling(
                    window,
                    min_periods=window
                ).mean()
            )
        )

        rolling_max = (
            shifted
            .groupby(df["segment_id"])
            .transform(
                lambda x: x.rolling(
                    window,
                    min_periods=window
                ).max()
            )
        )

        rolling_std = (
            shifted
            .groupby(df["segment_id"])
            .transform(
                lambda x: x.rolling(
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
# CREATE TURBIDITY-SPECIFIC FEATURES
# ============================================================

print("\nCreating additional turbidity features...")

# Recent turbidity trend
df["turbidity_change_1h"] = (
    df["turbidity"]
    - df.groupby("segment_id")["turbidity"].shift(1)
)

# Ratio relative to previous value
previous_turbidity = (
    df.groupby("segment_id")["turbidity"]
    .shift(1)
)

df["turbidity_ratio_1h"] = (
    df["turbidity"] /
    previous_turbidity.replace(0, np.nan)
)


# ============================================================
# BASIC FEATURE SUMMARY
# ============================================================

print("\nFeature creation completed.")

print(f"Total columns: {len(df.columns):,}")

print("\nNumber of missing values in selected ML features:")

feature_columns = [
    column
    for column in df.columns
    if column not in [
        "timestamp_utc",
        "segment_id"
    ]
]

missing_summary = (
    df[feature_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(
    missing_summary.head(15)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print("\nSaved file:")
print(OUTPUT_FILE)

print(f"\nFinal dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")