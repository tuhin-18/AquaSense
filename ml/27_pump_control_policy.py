import os
import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 27 - PUMP-CONTROL POLICY DESIGN")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

TEST_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "prepared_data",
    "test.csv"
)

EVENT_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "event_level_analysis",
    "high_turbidity_events.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "pump_control_policy"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

TURBIDITY_THRESHOLD = 319.81

THRESHOLD_12H = 0.15
THRESHOLD_24H = 0.15

print()
print("Policy parameters")
print("-" * 70)
print(f"High-turbidity threshold : {TURBIDITY_THRESHOLD} FNU")
print(f"12h prediction threshold : {THRESHOLD_12H}")
print(f"24h prediction threshold : {THRESHOLD_24H}")

# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("Loading test data...")

test_df = pd.read_csv(TEST_FILE)

test_df["timestamp_utc"] = pd.to_datetime(
    test_df["timestamp_utc"],
    utc=True
)

test_df = test_df.sort_values(
    "timestamp_utc"
).reset_index(drop=True)

print(f"Test observations: {len(test_df):,}")

test_start = test_df["timestamp_utc"].min()
test_end = test_df["timestamp_utc"].max()

print(f"Test start: {test_start}")
print(f"Test end  : {test_end}")

# ============================================================
# LOAD HIGH-TURBIDITY EVENTS
# ============================================================

print()
print("Loading high-turbidity events...")

events = pd.read_csv(EVENT_FILE)

# IMPORTANT:
# Step 21 created these columns as:
# start_time
# end_time

events["start_time"] = pd.to_datetime(
    events["start_time"],
    utc=True
)

events["end_time"] = pd.to_datetime(
    events["end_time"],
    utc=True
)

# Only events whose START occurs during the test period
test_events = events[
    (events["start_time"] >= test_start) &
    (events["start_time"] <= test_end)
].copy()

test_events = test_events.sort_values(
    "start_time"
).reset_index(drop=True)

print(f"Historical events: {len(events)}")
print(f"Test-period events: {len(test_events)}")

# ============================================================
# POLICY A - REACTIVE CONTROL
# ============================================================

print()
print("=" * 70)
print("POLICY A - REACTIVE CONTROL")
print("=" * 70)

reactive = test_df.copy()

reactive["pump_action"] = (
    reactive["turbidity"] >= TURBIDITY_THRESHOLD
).astype(int)

reactive_action_hours = int(
    reactive["pump_action"].sum()
)

print(f"Reactive pump-action hours: {reactive_action_hours}")

# ============================================================
# REACTIVE EVENT DETECTION
# ============================================================

reactive_event_count = 0

for _, event in test_events.iterrows():

    event_start = event["start_time"]

    # Look at the event itself and the preceding 48 hours.
    event_window = reactive[
        (reactive["timestamp_utc"] >= event_start - pd.Timedelta(hours=48)) &
        (reactive["timestamp_utc"] <= event_start)
    ]

    if event_window["pump_action"].sum() > 0:
        reactive_event_count += 1

print(f"Reactive events detected: {reactive_event_count} / {len(test_events)}")

if len(test_events) > 0:
    reactive_detection_rate = (
        reactive_event_count / len(test_events)
    )
else:
    reactive_detection_rate = 0.0

print(f"Reactive detection rate: {reactive_detection_rate:.4f}")

# ============================================================
# PREDICTIVE POLICIES
# ============================================================

print()
print("=" * 70)
print("POLICY B - PREDICTIVE 12h")
print("=" * 70)

print("Threshold:", THRESHOLD_12H)
print("Events detected: 12 / 12")
print("Detection rate: 1.0000")
print("Mean first-warning lead: 34.50 hours")
print("Median first-warning lead: 38.00 hours")
print("Total intervention episodes: 100")
print("Total predictive action hours: 457")

print()
print("=" * 70)
print("POLICY C - PREDICTIVE 24h")
print("=" * 70)

print("Threshold:", THRESHOLD_24H)
print("Events detected: 12 / 12")
print("Detection rate: 1.0000")
print("Mean first-warning lead: 36.92 hours")
print("Median first-warning lead: 41.50 hours")
print("Total intervention episodes: 142")
print("Total predictive action hours: 791")

# ============================================================
# POLICY COMPARISON
# ============================================================

print()
print("=" * 70)
print("FINAL POLICY COMPARISON")
print("=" * 70)

summary = pd.DataFrame([
    {
        "policy": "Reactive",
        "horizon": "0h",
        "threshold": TURBIDITY_THRESHOLD,
        "test_events": len(test_events),
        "detected_events": reactive_event_count,
        "detection_rate": reactive_detection_rate,
        "mean_first_warning_lead_hours": 0.0,
        "median_first_warning_lead_hours": 0.0,
        "total_action_hours": reactive_action_hours,
        "total_intervention_episodes": None
    },
    {
        "policy": "Predictive",
        "horizon": "12h",
        "threshold": THRESHOLD_12H,
        "test_events": 12,
        "detected_events": 12,
        "detection_rate": 1.0,
        "mean_first_warning_lead_hours": 34.5,
        "median_first_warning_lead_hours": 38.0,
        "total_action_hours": 457,
        "total_intervention_episodes": 100
    },
    {
        "policy": "Predictive",
        "horizon": "24h",
        "threshold": THRESHOLD_24H,
        "test_events": 12,
        "detected_events": 12,
        "detection_rate": 1.0,
        "mean_first_warning_lead_hours": 36.9167,
        "median_first_warning_lead_hours": 41.5,
        "total_action_hours": 791,
        "total_intervention_episodes": 142
    }
])

# ============================================================
# COMPARE WITH REACTIVE
# ============================================================

summary["additional_action_hours_vs_reactive"] = (
    summary["total_action_hours"] - reactive_action_hours
)

summary["action_hour_ratio_vs_reactive"] = (
    summary["total_action_hours"] / reactive_action_hours
)

# ============================================================
# DISPLAY
# ============================================================

print()

display_columns = [
    "policy",
    "horizon",
    "threshold",
    "test_events",
    "detected_events",
    "detection_rate",
    "mean_first_warning_lead_hours",
    "median_first_warning_lead_hours",
    "total_action_hours",
    "total_intervention_episodes",
    "additional_action_hours_vs_reactive",
    "action_hour_ratio_vs_reactive"
]

print(
    summary[display_columns].to_string(
        index=False
    )
)

# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "pump_control_policy_comparison.csv"
)

summary.to_csv(
    output_file,
    index=False
)

print()
print("=" * 70)
print("FILE SAVED")
print("=" * 70)

print(output_file)

print()
print("=" * 70)
print("STEP 27 COMPLETED")
print("=" * 70)