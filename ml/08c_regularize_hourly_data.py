import pandas as pd
from pathlib import Path


# ============================================================
# AQUASENSE - REGULARIZE HOURLY WATER-QUALITY DATA
# ============================================================

print("=" * 70)
print("AQUASENSE HOURLY TIME-SERIES REGULARIZATION")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(r"C:\Users\tusha\Documents\AquaSense")

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "SEAN_FQ_Q_2011-2024_KLGO_TA.csv"
)

OUTPUT_DIR = BASE_DIR / "ml_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)

print("Dataset loaded successfully.")


# ============================================================
# 3. PREPARE TIMESTAMP
# ============================================================

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    errors="coerce",
    utc=True
)

df["turbidity"] = pd.to_numeric(
    df["turbidity"],
    errors="coerce"
)

df = (
    df
    .sort_values("timestamp_utc")
    .reset_index(drop=True)
)


# ============================================================
# 4. KEEP RELEVANT VARIABLES
# ============================================================

columns = [
    "timestamp_utc",
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "do_saturation",
    "turbidity"
]

df = df[columns]


# ============================================================
# 5. CREATE CONTINUOUS SEGMENTS
# ============================================================
#
# A gap > 6 hours means a new monitoring segment.
#
# We DO NOT resample across these large gaps.
# ============================================================

time_diff = df["timestamp_utc"].diff()

df["segment_id"] = (
    (time_diff > pd.Timedelta(hours=6))
    .fillna(True)
    .cumsum()
)

print(
    f"\nMonitoring segments: "
    f"{df['segment_id'].nunique()}"
)


# ============================================================
# 6. REGULARIZE EACH SEGMENT TO HOURLY FREQUENCY
# ============================================================

print("\nRegularizing segments to hourly frequency...")

segments = []

for segment_id, segment in df.groupby("segment_id"):

    segment = (
        segment
        .set_index("timestamp_utc")
        .sort_index()
    )

    # Remove segment_id before resampling
    segment = segment.drop(
        columns=["segment_id"]
    )

    # Resample to hourly frequency.
    #
    # Mean is used if more than one observation
    # falls into the same hourly bin.
    hourly = (
        segment
        .resample("1h")
        .mean()
    )

    hourly["segment_id"] = segment_id

    segments.append(hourly)


# ============================================================
# 7. COMBINE SEGMENTS
# ============================================================

hourly_df = pd.concat(
    segments
)

hourly_df = (
    hourly_df
    .reset_index()
    .sort_values("timestamp_utc")
    .reset_index(drop=True)
)


# ============================================================
# 8. REPORT SIZE
# ============================================================

print("\n" + "=" * 70)
print("REGULARIZATION RESULT")
print("=" * 70)

print(
    f"Original observations : {len(df):,}"
)

print(
    f"Hourly observations   : {len(hourly_df):,}"
)

print(
    f"Monitoring segments   : "
    f"{hourly_df['segment_id'].nunique()}"
)


# ============================================================
# 9. CHECK TURBIDITY
# ============================================================

threshold = (
    hourly_df["turbidity"]
    .dropna()
    .quantile(0.99)
)

print(
    f"\n99th percentile turbidity "
    f"after regularization: {threshold:.2f} FNU"
)


hourly_df["high_turbidity"] = (
    hourly_df["turbidity"] > threshold
)


print(
    "High-turbidity hourly observations:",
    int(hourly_df["high_turbidity"].sum())
)


# ============================================================
# 10. SAVE REGULARIZED DATA
# ============================================================

output_path = (
    OUTPUT_DIR
    / "water_quality_hourly_regularized.csv"
)

hourly_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 11. SAVE TURBIDITY EVENTS
# ============================================================

event_df = hourly_df[
    hourly_df["high_turbidity"]
].copy()

event_output_path = (
    OUTPUT_DIR
    / "high_turbidity_hourly_observations.csv"
)

event_df.to_csv(
    event_output_path,
    index=False
)


# ============================================================
# 12. FINAL
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(output_path)
print(event_output_path)

print("\n" + "=" * 70)
print("HOURLY REGULARIZATION COMPLETE")
print("=" * 70)