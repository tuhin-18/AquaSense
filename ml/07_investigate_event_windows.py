import pandas as pd
from pathlib import Path

# ============================================================
# AQUASENSE EVENT WINDOW INVESTIGATION
# ============================================================

print("=" * 70)
print("AQUASENSE EVENT WINDOW INVESTIGATION")
print("=" * 70)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\Users\tusha\Documents\AquaSense")

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "SEAN_FQ_Q_2011-2024_KLGO_TA.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH, low_memory=False)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    errors="coerce",
    utc=True
)

df = df.sort_values("timestamp_utc").reset_index(drop=True)

print("Dataset loaded successfully.")

# ------------------------------------------------------------
# Columns
# ------------------------------------------------------------

columns = [
    "timestamp_utc",
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "do_saturation",
    "turbidity"
]

# ------------------------------------------------------------
# Function to display a time window
# ------------------------------------------------------------

def show_window(center_time, hours_before=6, hours_after=6):

    center_time = pd.Timestamp(center_time)

    start_time = center_time - pd.Timedelta(hours=hours_before)
    end_time = center_time + pd.Timedelta(hours=hours_after)

    window = df[
        (df["timestamp_utc"] >= start_time) &
        (df["timestamp_utc"] <= end_time)
    ][columns].copy()

    print("\n" + "=" * 70)
    print(f"EVENT WINDOW")
    print(f"Center : {center_time}")
    print(f"Window : {start_time} -> {end_time}")
    print("=" * 70)

    print(window.to_string(index=False))

    return window


# ============================================================
# 1. CONDUCTANCE EXTREME
# ============================================================

print("\n")
print("#" * 70)
print("1. SPECIFIC CONDUCTANCE EXTREME")
print("#" * 70)

conductance_window = show_window(
    "2011-04-30 01:00:00+00:00",
    hours_before=6,
    hours_after=6
)

conductance_window.to_csv(
    OUTPUT_DIR / "conductance_extreme_window.csv",
    index=False
)


# ============================================================
# 2. ZERO DO SATURATION
# ============================================================

print("\n")
print("#" * 70)
print("2. ZERO DO SATURATION")
print("#" * 70)

do_window = show_window(
    "2018-04-23 23:00:33+00:00",
    hours_before=6,
    hours_after=6
)

do_window.to_csv(
    OUTPUT_DIR / "zero_do_saturation_window.csv",
    index=False
)


# ============================================================
# 3. MAJOR TURBIDITY EVENT - 2013
# ============================================================

print("\n")
print("#" * 70)
print("3. MAJOR TURBIDITY EVENT - 2013")
print("#" * 70)

turbidity_2013 = show_window(
    "2013-09-08 16:00:52+00:00",
    hours_before=12,
    hours_after=12
)

turbidity_2013.to_csv(
    OUTPUT_DIR / "turbidity_event_2013_window.csv",
    index=False
)


# ============================================================
# 4. MAJOR TURBIDITY EVENT - 2019
# ============================================================

print("\n")
print("#" * 70)
print("4. MAJOR TURBIDITY EVENT - 2019")
print("#" * 70)

turbidity_2019 = show_window(
    "2019-09-21 07:00:54+00:00",
    hours_before=12,
    hours_after=12
)

turbidity_2019.to_csv(
    OUTPUT_DIR / "turbidity_event_2019_window.csv",
    index=False
)


# ============================================================
# 5. MAJOR TURBIDITY EVENT - 2022
# ============================================================

print("\n")
print("#" * 70)
print("5. MAJOR TURBIDITY EVENT - 2022")
print("#" * 70)

turbidity_2022 = show_window(
    "2022-09-27 00:00:00+00:00",
    hours_before=12,
    hours_after=12
)

turbidity_2022.to_csv(
    OUTPUT_DIR / "turbidity_event_2022_window.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("EVENT WINDOW INVESTIGATION COMPLETE")
print("=" * 70)

print("\nOutput files saved in:")
print(OUTPUT_DIR)