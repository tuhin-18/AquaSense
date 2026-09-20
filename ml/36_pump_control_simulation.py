import os
import pandas as pd


# ============================================================
# AQUASENSE - PREVENTIVE PUMP CONTROL SIMULATION
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "simulation",
    "historical_event_2022_simulation.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "simulation"
)


# ============================================================
# SETTINGS
# ============================================================

TURBIDITY_THRESHOLD = 319.81

# Preventive policy:
# stop/hold pump when the 12-hour model probability
# reaches 25%.

PREDICTIVE_THRESHOLD = 0.25


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AQUASENSE - PREVENTIVE PUMP CONTROL SIMULATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = df.sort_values("timestamp").reset_index(drop=True)

print("\nRows loaded:", len(df))


# ============================================================
# REACTIVE POLICY
# ============================================================

# Reactive control:
# Pump continues running until turbidity actually
# crosses the high-risk threshold.

df["reactive_pump_action"] = (
    df["turbidity"] >= TURBIDITY_THRESHOLD
)


# ============================================================
# PREDICTIVE POLICY
# ============================================================

# Predictive control:
# Pump is stopped/held before the event if the AI
# predicts elevated risk within the next 12 hours.

df["predictive_pump_action"] = (
    df["risk_probability_12h"] >= PREDICTIVE_THRESHOLD
)


# ============================================================
# FIND FIRST ACTION TIMES
# ============================================================

reactive_rows = df[
    df["reactive_pump_action"]
]

predictive_rows = df[
    df["predictive_pump_action"]
]


print("\n")
print("=" * 70)
print("PUMP CONTROL RESULTS")
print("=" * 70)


# ------------------------------------------------------------
# REACTIVE
# ------------------------------------------------------------

if len(reactive_rows) > 0:

    reactive_start = (
        reactive_rows["timestamp"].iloc[0]
    )

    print(
        "\nReactive pump action starts:",
        reactive_start
    )

else:

    reactive_start = None

    print(
        "\nReactive policy did not activate."
    )


# ------------------------------------------------------------
# PREDICTIVE
# ------------------------------------------------------------

if len(predictive_rows) > 0:

    predictive_start = (
        predictive_rows["timestamp"].iloc[0]
    )

    print(
        "Predictive pump action starts:",
        predictive_start
    )

else:

    predictive_start = None

    print(
        "Predictive policy did not activate."
    )


# ============================================================
# LEAD TIME
# ============================================================

if (
    predictive_start is not None
    and reactive_start is not None
):

    lead_time = (
        reactive_start - predictive_start
    )

    lead_hours = (
        lead_time.total_seconds() / 3600
    )

    print(
        "\nPreventive lead time:",
        round(lead_hours, 2),
        "hours"
    )

else:

    lead_hours = None


# ============================================================
# ACTION HOURS
# ============================================================

reactive_hours = int(
    df["reactive_pump_action"].sum()
)

predictive_hours = int(
    df["predictive_pump_action"].sum()
)


print(
    "\nReactive pump-action hours:",
    reactive_hours
)

print(
    "Predictive pump-action hours:",
    predictive_hours
)


# ============================================================
# EVENT DETECTION
# ============================================================

actual_event_exists = (
    df["turbidity"] >= TURBIDITY_THRESHOLD
).any()

predictive_detected = (
    df["risk_probability_12h"] >= PREDICTIVE_THRESHOLD
).any()


print(
    "\nActual high-turbidity event:",
    "YES" if actual_event_exists else "NO"
)

print(
    "Predictive warning generated:",
    "YES" if predictive_detected else "NO"
)


# ============================================================
# DISPLAY KEY TIMELINE
# ============================================================

print("\n")
print("=" * 70)
print("KEY PUMP-CONTROL TIMELINE")
print("=" * 70)

if predictive_start is not None:

    timeline_start = (
        predictive_start - pd.Timedelta(hours=2)
    )

    timeline_end = (
        predictive_start + pd.Timedelta(hours=10)
    )

    timeline = df[
        (df["timestamp"] >= timeline_start) &
        (df["timestamp"] <= timeline_end)
    ][
        [
            "timestamp",
            "turbidity",
            "risk_probability_12h",
            "predictive_pump_action",
            "reactive_pump_action"
        ]
    ]

    print(
        timeline.to_string(index=False)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "pump_control_2022_simulation.csv"
)

df.to_csv(
    output_file,
    index=False
)


summary = pd.DataFrame([
    {
        "reactive_action_start": reactive_start,
        "predictive_action_start": predictive_start,
        "preventive_lead_hours": lead_hours,
        "reactive_action_hours": reactive_hours,
        "predictive_action_hours": predictive_hours,
        "actual_event": actual_event_exists,
        "predictive_warning": predictive_detected
    }
])

summary_file = os.path.join(
    OUTPUT_DIR,
    "pump_control_2022_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


print("\n")
print("=" * 70)
print("SIMULATION COMPLETED")
print("=" * 70)

print("\nDetailed results:")
print(output_file)

print("\nSummary:")
print(summary_file)