import os
import numpy as np
import pandas as pd


# ============================================================
# STEP 21 - EVENT-LEVEL EARLY WARNING ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 21 - EVENT-LEVEL EARLY WARNING ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

DATA_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "water_quality_hourly_regularized.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "event_level_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

TURBIDITY_THRESHOLD = 319.81

# Minimum consecutive high-turbidity observations
MIN_EVENT_HOURS = 3

# Maximum gap allowed between high-turbidity observations
# Since data are hourly, 2 hours allows one missing hour.
MAX_GAP_HOURS = 2


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading hourly data...")

df = pd.read_csv(DATA_FILE)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    utc=True
)

df = df.sort_values(
    ["segment_id", "timestamp_utc"]
).reset_index(drop=True)

print("Rows loaded:", len(df))


# ------------------------------------------------------------
# CREATE HIGH-TURBIDITY FLAG
# ------------------------------------------------------------

df["high_turbidity"] = (
    df["turbidity"] >= TURBIDITY_THRESHOLD
)

print("\nTurbidity threshold:", TURBIDITY_THRESHOLD)
print(
    "High-turbidity observations:",
    int(df["high_turbidity"].sum())
)


# ------------------------------------------------------------
# IDENTIFY CANDIDATE EVENT GROUPS
# ------------------------------------------------------------

events = []

event_id = 0


for segment_id, group in df.groupby("segment_id"):

    group = group.sort_values("timestamp_utc").copy()

    high = group[
        group["high_turbidity"]
    ].copy()

    if len(high) == 0:
        continue

    # Calculate time gap between high-turbidity observations
    high["time_gap_hours"] = (
        high["timestamp_utc"]
        .diff()
        .dt.total_seconds()
        / 3600
    )

    # Start a new event when:
    # 1. first observation
    # 2. gap is larger than allowed
    new_event = (
        high["time_gap_hours"].isna()
        |
        (high["time_gap_hours"] > MAX_GAP_HOURS)
    )

    high["local_event_id"] = new_event.cumsum()

    for local_id, event_group in high.groupby(
        "local_event_id"
    ):

        event_group = event_group.sort_values(
            "timestamp_utc"
        )

        duration_hours = (
            (
                event_group["timestamp_utc"].iloc[-1]
                -
                event_group["timestamp_utc"].iloc[0]
            ).total_seconds()
            / 3600
        ) + 1

        # Require at least MIN_EVENT_HOURS observations
        if len(event_group) < MIN_EVENT_HOURS:
            continue

        event_id += 1

        events.append(
            {
                "event_id": event_id,
                "segment_id": segment_id,
                "start_time": event_group[
                    "timestamp_utc"
                ].iloc[0],
                "end_time": event_group[
                    "timestamp_utc"
                ].iloc[-1],
                "duration_hours": duration_hours,
                "high_turbidity_observations": len(
                    event_group
                ),
                "maximum_turbidity": event_group[
                    "turbidity"
                ].max(),
                "mean_turbidity": event_group[
                    "turbidity"
                ].mean()
            }
        )


events_df = pd.DataFrame(events)


# ------------------------------------------------------------
# SAVE EVENTS
# ------------------------------------------------------------

events_file = os.path.join(
    OUTPUT_DIR,
    "high_turbidity_events.csv"
)

events_df.to_csv(
    events_file,
    index=False
)


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HIGH-TURBIDITY EVENT SUMMARY")
print("=" * 70)

print(
    "Minimum event duration:",
    MIN_EVENT_HOURS,
    "hours"
)

print(
    "Detected events:",
    len(events_df)
)

if len(events_df) > 0:

    print(
        "\nAverage event duration:",
        round(
            events_df["duration_hours"].mean(),
            2
        ),
        "hours"
    )

    print(
        "Maximum event duration:",
        round(
            events_df["duration_hours"].max(),
            2
        ),
        "hours"
    )

    print(
        "Maximum turbidity across events:",
        round(
            events_df["maximum_turbidity"].max(),
            2
        )
    )

    print("\nDetected events:")
    print(
        events_df[
            [
                "event_id",
                "segment_id",
                "start_time",
                "end_time",
                "duration_hours",
                "maximum_turbidity"
            ]
        ].to_string(index=False)
    )


print("\nFile saved:")
print(events_file)

print("\n" + "=" * 70)
print("STEP 21 COMPLETED")
print("=" * 70)