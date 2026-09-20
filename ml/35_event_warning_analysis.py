import os
import pandas as pd


# ============================================================
# AQUASENSE - EVENT WARNING ANALYSIS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "simulation",
    "historical_event_2022_simulation.csv"
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

# Strong/actionable warning definition.
#
# We use the 12-hour model because our system is intended
# to provide enough advance notice for preventive action.

ACTIONABLE_THRESHOLD = 0.25

EVENT_TURBIDITY_THRESHOLD = 319.81


# ============================================================
# LOAD SIMULATION
# ============================================================

print("=" * 70)
print("AQUASENSE - EVENT WARNING ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

print("\nRows loaded:", len(df))


# ============================================================
# FIND ACTUAL EVENT
# ============================================================

event_rows = df[
    df["turbidity"] >= EVENT_TURBIDITY_THRESHOLD
]

if len(event_rows) == 0:

    print("\nNo high-turbidity event found.")

    raise SystemExit


event_start = event_rows["timestamp"].iloc[0]


# ============================================================
# FIND ACTIONABLE WARNING
# ============================================================

warning_rows = df[
    (df["timestamp"] < event_start) &
    (df["risk_probability_12h"] >= ACTIONABLE_THRESHOLD)
]


print("\n")
print("=" * 70)
print("EVENT INFORMATION")
print("=" * 70)

print(
    "Actual event start:",
    event_start
)

print(
    "Turbidity at event start:",
    round(
        event_rows["turbidity"].iloc[0],
        2
    ),
    "FNU"
)


# ============================================================
# WARNING RESULT
# ============================================================

if len(warning_rows) == 0:

    print("\nNo actionable warning was generated.")

else:

    first_warning = warning_rows.iloc[0]

    warning_time = first_warning["timestamp"]

    lead_time = event_start - warning_time

    print("\n")
    print("=" * 70)
    print("ACTIONABLE AI WARNING")
    print("=" * 70)

    print(
        "Warning time:",
        warning_time
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
        "6h risk probability:",
        round(
            first_warning["risk_probability_6h"] * 100,
            2
        ),
        "%"
    )

    print(
        "24h risk probability:",
        round(
            first_warning["risk_probability_24h"] * 100,
            2
        ),
        "%"
    )

    print(
        "Warning lead time:",
        lead_time
    )

    print(
        "Lead time in hours:",
        round(
            lead_time.total_seconds() / 3600,
            2
        )
    )


# ============================================================
# SAVE RESULT
# ============================================================

result = {
    "event_start": event_start,
    "warning_time": (
        warning_rows["timestamp"].iloc[0]
        if len(warning_rows) > 0
        else pd.NaT
    ),
    "event_turbidity": event_rows["turbidity"].iloc[0],
    "warning_probability_12h": (
        warning_rows["risk_probability_12h"].iloc[0]
        if len(warning_rows) > 0
        else None
    ),
    "warning_probability_6h": (
        warning_rows["risk_probability_6h"].iloc[0]
        if len(warning_rows) > 0
        else None
    ),
    "warning_probability_24h": (
        warning_rows["risk_probability_24h"].iloc[0]
        if len(warning_rows) > 0
        else None
    ),
    "lead_time_hours": (
        (event_start - warning_rows["timestamp"].iloc[0])
        .total_seconds() / 3600
        if len(warning_rows) > 0
        else None
    )
}

result_df = pd.DataFrame([result])

output_file = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "simulation",
    "event_warning_2022.csv"
)

result_df.to_csv(
    output_file,
    index=False
)


print("\n")
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print("\nSaved:")
print(output_file)