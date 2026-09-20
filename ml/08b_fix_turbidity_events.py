import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# AQUASENSE - CORRECTED TURBIDITY EVENT ANALYSIS
# ============================================================

print("=" * 70)
print("AQUASENSE CORRECTED TURBIDITY EVENT ANALYSIS")
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
print(f"Rows: {len(df):,}")


# ============================================================
# 3. PREPARE DATA
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
# 4. CREATE MONITORING SEGMENTS
# ============================================================

time_diff = df["timestamp_utc"].diff()

df["segment_id"] = (
    (time_diff > pd.Timedelta(hours=6))
    .fillna(True)
    .cumsum()
)

print(
    f"\nContinuous monitoring segments: "
    f"{df['segment_id'].nunique()}"
)


# ============================================================
# 5. CALCULATE THRESHOLD
# ============================================================

valid_turbidity = df["turbidity"].dropna()

threshold = valid_turbidity.quantile(0.99)

print("\n" + "=" * 70)
print("EVENT THRESHOLD")
print("=" * 70)

print(
    f"99th percentile turbidity: "
    f"{threshold:.2f} FNU"
)


# ============================================================
# 6. FLAG HIGH TURBIDITY
# ============================================================

df["high_turbidity"] = (
    df["turbidity"] > threshold
)

print(
    "\nHigh-turbidity observations:",
    int(df["high_turbidity"].sum())
)


# ============================================================
# 7. CREATE EVENT GROUPS
# ============================================================
#
# A new event starts when:
#
# 1. Current observation is high turbidity AND
# 2. Previous observation is not high turbidity
#
# OR
#
# 3. Monitoring segment changes
#
# OR
#
# 4. Gap between high-turbidity observations > 3 hours
#
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
        | (time_diff > pd.Timedelta(hours=3))
    )
)

df["event_id"] = new_event.cumsum()

df.loc[
    ~df["high_turbidity"],
    "event_id"
] = np.nan


# ============================================================
# 8. EXTRACT EVENT OBSERVATIONS
# ============================================================

event_data = df[
    df["high_turbidity"]
].copy()


# ============================================================
# 9. SUMMARIZE EVENTS
# ============================================================

events = (
    event_data
    .groupby("event_id")
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


# ============================================================
# 10. CALCULATE EVENT DURATION
# ============================================================

events["duration_hours"] = (
    (
        events["end_time"]
        - events["start_time"]
    )
    .dt.total_seconds()
    / 3600
)


# ============================================================
# 11. CLASSIFY PERSISTENCE
# ============================================================

events["persistence"] = np.where(
    events["observations"] >= 3,
    "persistent",
    "short"
)


# ============================================================
# 12. REORDER COLUMNS
# ============================================================

events = events[
    [
        "start_time",
        "end_time",
        "duration_hours",
        "observations",
        "peak_turbidity",
        "mean_turbidity",
        "persistence",
        "segment_id"
    ]
]


# ============================================================
# 13. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED EVENT SUMMARY")
print("=" * 70)

total_events = len(events)

persistent_events = (
    events["persistence"] == "persistent"
).sum()

short_events = (
    events["persistence"] == "short"
).sum()

print(f"Total events              : {total_events}")
print(
    f"Persistent events (>=3)  : "
    f"{persistent_events}"
)
print(
    f"Short events (<3)        : "
    f"{short_events}"
)


# ============================================================
# 14. DISPLAY PERSISTENT EVENTS
# ============================================================

print("\n" + "=" * 70)
print("PERSISTENT HIGH-TURBIDITY EVENTS")
print("=" * 70)

persistent = (
    events[
        events["persistence"] == "persistent"
    ]
    .sort_values(
        "peak_turbidity",
        ascending=False
    )
)

print(
    persistent
    .to_string(index=False)
)


# ============================================================
# 15. SAVE ALL EVENTS
# ============================================================

all_events_path = (
    OUTPUT_DIR
    / "turbidity_events_corrected.csv"
)

events.to_csv(
    all_events_path,
    index=False
)


# ============================================================
# 16. SAVE PERSISTENT EVENTS
# ============================================================

persistent_path = (
    OUTPUT_DIR
    / "turbidity_persistent_events_corrected.csv"
)

persistent.to_csv(
    persistent_path,
    index=False
)


# ============================================================
# 17. SAVE LABELED DATA
# ============================================================

labeled_path = (
    OUTPUT_DIR
    / "water_quality_event_labels_corrected.csv"
)

columns_to_save = [
    "timestamp_utc",
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "do_saturation",
    "turbidity",
    "segment_id",
    "high_turbidity",
    "event_id"
]

df[columns_to_save].to_csv(
    labeled_path,
    index=False
)


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(all_events_path)
print(persistent_path)
print(labeled_path)

print("\n" + "=" * 70)
print("CORRECTED EVENT ANALYSIS COMPLETE")
print("=" * 70)