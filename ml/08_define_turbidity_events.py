import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# AQUASENSE - TURBIDITY EVENT LABEL ANALYSIS
# ============================================================

print("=" * 70)
print("AQUASENSE TURBIDITY EVENT LABEL ANALYSIS")
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
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded successfully.")
print(f"Rows: {len(df):,}")


# ============================================================
# 3. PREPARE TIMESTAMP
# ============================================================

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    errors="coerce",
    utc=True
)

df = df.sort_values("timestamp_utc").reset_index(drop=True)


# ============================================================
# 4. CONVERT TURBIDITY TO NUMERIC
# ============================================================

df["turbidity"] = pd.to_numeric(
    df["turbidity"],
    errors="coerce"
)


# ============================================================
# 5. DEFINE CONTINUOUS MONITORING SEGMENTS
# ============================================================
#
# A gap greater than 6 hours starts a new monitoring segment.
# This prevents us from treating seasonal/off-season gaps
# as continuous time-series data.
# ============================================================

time_diff = df["timestamp_utc"].diff()

df["segment_id"] = (
    (time_diff > pd.Timedelta(hours=6))
    .fillna(True)
    .cumsum()
)


print("\nContinuous monitoring segments:")
print(f"Number of segments: {df['segment_id'].nunique()}")


# ============================================================
# 6. CALCULATE TURBIDITY THRESHOLD
# ============================================================

valid_turbidity = df["turbidity"].dropna()

q95 = valid_turbidity.quantile(0.95)
q99 = valid_turbidity.quantile(0.99)

print("\n" + "=" * 70)
print("TURBIDITY THRESHOLDS")
print("=" * 70)

print(f"95th percentile : {q95:.2f} FNU")
print(f"99th percentile : {q99:.2f} FNU")


# ============================================================
# 7. CREATE HIGH-TURBIDITY FLAG
# ============================================================

df["high_turbidity"] = (
    df["turbidity"] > q99
)


print("\nObservations above 99th percentile:")
print(
    df["high_turbidity"]
    .sum()
)


# ============================================================
# 8. IDENTIFY CONSECUTIVE HIGH-TURBIDITY RUNS
# ============================================================
#
# A new event begins when:
# - the previous observation was not high turbidity, OR
# - there is a time gap greater than 2 hours, OR
# - the monitoring segment changes.
#
# This keeps separate events separate.
# ============================================================

previous_high = (
    df["high_turbidity"]
    .shift(1)
    .fillna(False)
)

previous_segment = (
    df["segment_id"]
    .shift(1)
)

new_event = (
    df["high_turbidity"]
    & (
        (~previous_high)
        | (df["segment_id"] != previous_segment)
        | (time_diff > pd.Timedelta(hours=2))
    )
)

df["event_number"] = new_event.cumsum()

# Rows that are not high turbidity do not belong to an event
df.loc[
    ~df["high_turbidity"],
    "event_number"
] = np.nan


# ============================================================
# 9. SUMMARIZE EVENTS
# ============================================================

event_rows = df[df["high_turbidity"]].copy()

if len(event_rows) == 0:

    print("\nNo high-turbidity events found.")
    raise SystemExit


events = (
    event_rows
    .groupby("event_number")
    .agg(
        start_time=("timestamp_utc", "min"),
        end_time=("timestamp_utc", "max"),
        observations=("timestamp_utc", "count"),
        peak_turbidity=("turbidity", "max"),
        mean_turbidity=("turbidity", "mean"),
        segment_id=("segment_id", "first")
    )
    .reset_index(drop=True)
)


# Calculate duration

events["duration_hours"] = (
    (
        events["end_time"]
        - events["start_time"]
    )
    .dt.total_seconds()
    / 3600
)


# Reorder columns

events = events[
    [
        "start_time",
        "end_time",
        "duration_hours",
        "observations",
        "peak_turbidity",
        "mean_turbidity",
        "segment_id"
    ]
]


# ============================================================
# 10. CLASSIFY EVENT PERSISTENCE
# ============================================================

events["persistence"] = np.where(
    events["observations"] >= 3,
    "persistent",
    "short"
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EVENT SUMMARY")
print("=" * 70)

print(f"Total candidate events : {len(events)}")

print(
    f"Persistent events (>=3 observations) : "
    f"{(events['persistence'] == 'persistent').sum()}"
)

print(
    f"Short events (<3 observations)       : "
    f"{(events['persistence'] == 'short').sum()}"
)


print("\nTop 20 events by peak turbidity:")

top_events = (
    events
    .sort_values(
        "peak_turbidity",
        ascending=False
    )
    .head(20)
)

print(
    top_events.to_string(index=False)
)


# ============================================================
# 12. SAVE ALL EVENTS
# ============================================================

all_events_path = (
    OUTPUT_DIR
    / "turbidity_events_all.csv"
)

events.to_csv(
    all_events_path,
    index=False
)


# ============================================================
# 13. SAVE PERSISTENT EVENTS
# ============================================================

persistent_events = events[
    events["persistence"] == "persistent"
].copy()

persistent_path = (
    OUTPUT_DIR
    / "turbidity_persistent_events.csv"
)

persistent_events.to_csv(
    persistent_path,
    index=False
)


# ============================================================
# 14. SAVE DATA WITH EVENT LABELS
# ============================================================

labeled_data_path = (
    OUTPUT_DIR
    / "water_quality_event_labels.csv"
)

label_columns = [
    "timestamp_utc",
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "do_saturation",
    "turbidity",
    "segment_id",
    "high_turbidity",
    "event_number"
]

df[label_columns].to_csv(
    labeled_data_path,
    index=False
)


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"All events:")
print(all_events_path)

print(f"\nPersistent events:")
print(persistent_path)

print(f"\nDataset with event labels:")
print(labeled_data_path)


print("\n" + "=" * 70)
print("EVENT LABEL ANALYSIS COMPLETE")
print("=" * 70)