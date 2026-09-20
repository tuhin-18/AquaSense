import pandas as pd
import os

# ============================================================
# AQUASENSE - INVESTIGATE EXTREME OBSERVATIONS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "SEAN_FQ_Q_2011-2024_KLGO_TA.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "ml_outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


PARAMETERS = [
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "do_saturation",
    "turbidity"
]


print("=" * 70)
print("AQUASENSE EXTREME OBSERVATION INVESTIGATION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    errors="coerce",
    utc=True
)

df = df.dropna(
    subset=["timestamp_utc"]
).copy()

df = df.sort_values(
    "timestamp_utc"
).reset_index(drop=True)

print("Dataset loaded successfully.")


# ============================================================
# DISPLAY EXTREME VALUES
# ============================================================

print("\n" + "=" * 70)
print("EXTREME OBSERVATIONS")
print("=" * 70)


extreme_records = []

for parameter in PARAMETERS:

    print("\n" + "-" * 70)
    print(parameter)
    print("-" * 70)

    # Highest 5
    highest = (
        df[
            [
                "timestamp_utc",
                parameter
            ]
        ]
        .dropna()
        .nlargest(5, parameter)
    )

    print("\nHighest 5:")
    print(
        highest.to_string(index=False)
    )

    # Lowest 5
    lowest = (
        df[
            [
                "timestamp_utc",
                parameter
            ]
        ]
        .dropna()
        .nsmallest(5, parameter)
    )

    print("\nLowest 5:")
    print(
        lowest.to_string(index=False)
    )


# ============================================================
# TURBIDITY EVENTS
# ============================================================

print("\n" + "=" * 70)
print("TURBIDITY ABOVE 300")
print("=" * 70)

turbidity_high = df[
    df["turbidity"] > 300
][
    [
        "timestamp_utc",
        "turbidity",
        "ph",
        "sp_conductance",
        "do_concentration",
        "do_saturation",
        "water_temperature"
    ]
]

print(
    f"Observations with turbidity > 300: "
    f"{len(turbidity_high)}"
)

print("\nFirst 20:")
print(
    turbidity_high.head(20).to_string(
        index=False
    )
)


# ============================================================
# TURBIDITY ABOVE 500
# ============================================================

print("\n" + "=" * 70)
print("TURBIDITY ABOVE 500")
print("=" * 70)

turbidity_very_high = df[
    df["turbidity"] > 500
][
    [
        "timestamp_utc",
        "turbidity",
        "ph",
        "sp_conductance",
        "do_concentration",
        "do_saturation",
        "water_temperature"
    ]
]

print(
    f"Observations with turbidity > 500: "
    f"{len(turbidity_very_high)}"
)

print("\nObservations:")
print(
    turbidity_very_high.to_string(
        index=False
    )
)


# ============================================================
# SUSPICIOUS CONDUCTANCE
# ============================================================

print("\n" + "=" * 70)
print("SPECIFIC CONDUCTANCE ABOVE 0.15")
print("=" * 70)

conductance_high = df[
    df["sp_conductance"] > 0.15
][
    [
        "timestamp_utc",
        "sp_conductance",
        "ph",
        "turbidity",
        "do_concentration",
        "do_saturation",
        "water_temperature"
    ]
]

print(
    f"Observations above 0.15: "
    f"{len(conductance_high)}"
)

print(
    conductance_high.to_string(
        index=False
    )
)


# ============================================================
# DO SATURATION = 0
# ============================================================

print("\n" + "=" * 70)
print("DISSOLVED OXYGEN SATURATION = 0")
print("=" * 70)

do_zero = df[
    df["do_saturation"] == 0
][
    [
        "timestamp_utc",
        "do_saturation",
        "do_concentration",
        "ph",
        "turbidity",
        "sp_conductance",
        "water_temperature"
    ]
]

print(
    f"Observations with DO saturation = 0: "
    f"{len(do_zero)}"
)

print(
    do_zero.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

extreme_output = os.path.join(
    OUTPUT_DIR,
    "extreme_observations.csv"
)

turbidity_output = os.path.join(
    OUTPUT_DIR,
    "high_turbidity_observations.csv"
)

conductance_output = os.path.join(
    OUTPUT_DIR,
    "high_conductance_observations.csv"
)

do_zero_output = os.path.join(
    OUTPUT_DIR,
    "zero_do_saturation_observations.csv"
)

turbidity_high.to_csv(
    turbidity_output,
    index=False
)

conductance_high.to_csv(
    conductance_output,
    index=False
)

do_zero.to_csv(
    do_zero_output,
    index=False
)


print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"\nHigh turbidity:\n{turbidity_output}"
)

print(
    f"\nHigh conductance:\n{conductance_output}"
)

print(
    f"\nZero DO saturation:\n{do_zero_output}"
)

print("\n" + "=" * 70)
print("EXTREME INVESTIGATION COMPLETE")
print("=" * 70)