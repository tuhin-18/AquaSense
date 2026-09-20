import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\tusha\Documents\AquaSense")

INPUT_FILE = PROJECT_ROOT / "ml_outputs" / "water_quality_hourly_regularized.csv"

OUTPUT_DIR = PROJECT_ROOT / "ml_outputs"
OUTPUT_FILE = OUTPUT_DIR / "future_risk_labels.csv"


# ============================================================
# SETTINGS
# ============================================================

# High turbidity threshold obtained from our earlier analysis
TURBIDITY_THRESHOLD = 319.81

# Prediction horizons in hours
HORIZONS = [1, 6, 12, 24]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 9 - CREATE FUTURE RISK LABELS")
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
# CREATE FUTURE-RISK LABELS
# ============================================================

print("\nCreating future-risk labels...")

for horizon in HORIZONS:

    future_columns = []

    # Look at the next 1, 2, ..., H hours
    for step in range(1, horizon + 1):

        future_turbidity = (
            df.groupby("segment_id")["turbidity"]
            .shift(-step)
        )

        future_columns.append(future_turbidity)

    future_values = pd.concat(
        future_columns,
        axis=1
    )

    # If ANY future value within the horizon
    # exceeds the threshold -> future high-risk event
    df[f"future_risk_{horizon}h"] = (
        future_values.gt(TURBIDITY_THRESHOLD)
        .any(axis=1)
        .astype(int)
    )

    # If there is no future observation available
    # for the complete horizon, remove the label.
    future_complete = future_values.notna().all(axis=1)

    df.loc[
        ~future_complete,
        f"future_risk_{horizon}h"
    ] = np.nan


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


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nFuture-risk label summary:")
print("-" * 60)

for horizon in HORIZONS:

    column = f"future_risk_{horizon}h"

    valid = df[column].dropna()

    positive = int((valid == 1).sum())
    negative = int((valid == 0).sum())

    print(f"\n{horizon}-hour horizon:")
    print(f"  Valid samples : {len(valid):,}")
    print(f"  Risk = 1      : {positive:,}")
    print(f"  Risk = 0      : {negative:,}")

    if len(valid) > 0:
        print(
            f"  Risk percentage: "
            f"{(positive / len(valid)) * 100:.2f}%"
        )


print("\n" + "=" * 60)
print("LABEL CREATION COMPLETE")
print("=" * 60)

print(f"\nSaved file:")
print(OUTPUT_FILE)