import os
import pandas as pd


# ============================================================
# AQUASENSE - CREATE 48-HOUR PREDICTION DEMO INPUT
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "water_quality_hourly_regularized.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "demo"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "water_quality_demo_48h.csv"
)


# ============================================================
# DEMO TIME WINDOW
# ============================================================

DEMO_END = pd.Timestamp(
    "2022-09-26 14:00:00",
    tz="UTC"
)

DEMO_START = DEMO_END - pd.Timedelta(hours=48)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AQUASENSE - CREATE 48-HOUR PREDICTION DEMO INPUT")
print("=" * 70)

print("\nLoading regularized water-quality dataset...")

df = pd.read_csv(INPUT_FILE)


# ============================================================
# CHECK TIMESTAMP COLUMN
# ============================================================

print("\nAvailable columns:")
print(df.columns.tolist())

if "timestamp_utc" in df.columns:
    timestamp_column = "timestamp_utc"
elif "timestamp" in df.columns:
    timestamp_column = "timestamp"
else:
    raise ValueError(
        "No timestamp column found. Expected 'timestamp_utc' or 'timestamp'."
    )


print("\nUsing timestamp column:", timestamp_column)


# ============================================================
# CONVERT TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df[timestamp_column],
    utc=True
)

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# SELECT ONLY THE HISTORICAL INPUT WINDOW
# ============================================================

demo = df[
    (df["timestamp"] >= DEMO_START) &
    (df["timestamp"] <= DEMO_END)
].copy()


# ============================================================
# SELECT SENSOR COLUMNS
# ============================================================

columns = [
    "timestamp",
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "do_saturation",
    "turbidity"
]

missing_columns = [
    column for column in columns
    if column not in demo.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

demo = demo[columns]


# ============================================================
# CHECK NUMBER OF ROWS
# ============================================================

if len(demo) < 24:
    raise ValueError(
        f"Only {len(demo)} rows found. "
        "At least 24 hourly observations are required."
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

demo.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\nDemo input created successfully.")

print("\nTime range:")
print("Start :", demo["timestamp"].min())
print("End   :", demo["timestamp"].max())

print("\nNumber of rows:", len(demo))

print("\nColumns:")
for column in demo.columns:
    print(" -", column)

print("\nLast 10 observations:")
print(demo.tail(10).to_string(index=False))

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)