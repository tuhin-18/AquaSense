import pandas as pd
import os

# ============================================================
# AQUASENSE - TIMESTAMP / TIME-SERIES ANALYSIS
# ============================================================

# Project folder
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "SEAN_FQ_Q_2011-2024_KLGO_TA.csv"
)

# Output folder
OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "ml_outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("AQUASENSE TIMESTAMP ANALYSIS")
print("=" * 70)

print("\nDataset path:")
print(DATASET_PATH)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)

print("Dataset loaded successfully.")

# ------------------------------------------------------------
# Check timestamp column
# ------------------------------------------------------------

TIMESTAMP_COLUMN = "timestamp_utc"

if TIMESTAMP_COLUMN not in df.columns:
    print("\nERROR: timestamp_utc column was not found.")
    print("\nAvailable columns:")
    print(df.columns.tolist())
    exit()

# ------------------------------------------------------------
# Convert timestamp
# ------------------------------------------------------------

print("\nConverting timestamp column...")

df[TIMESTAMP_COLUMN] = pd.to_datetime(
    df[TIMESTAMP_COLUMN],
    errors="coerce",
    utc=True
)

# Count invalid timestamps
invalid_timestamps = df[TIMESTAMP_COLUMN].isna().sum()

print(f"Invalid timestamps : {invalid_timestamps}")

# Remove invalid timestamps for time analysis
df = df.dropna(subset=[TIMESTAMP_COLUMN]).copy()

# ------------------------------------------------------------
# Sort chronologically
# ------------------------------------------------------------

df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

print("\nDataset sorted chronologically.")

# ------------------------------------------------------------
# Basic time information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TIME RANGE")
print("=" * 70)

print(f"Earliest timestamp : {df[TIMESTAMP_COLUMN].min()}")
print(f"Latest timestamp   : {df[TIMESTAMP_COLUMN].max()}")

# ------------------------------------------------------------
# Duplicate timestamps
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE TIMESTAMPS")
print("=" * 70)

duplicate_count = df[TIMESTAMP_COLUMN].duplicated().sum()

print(f"Duplicate timestamps : {duplicate_count}")

# ------------------------------------------------------------
# Calculate time difference between consecutive observations
# ------------------------------------------------------------

df["time_difference"] = df[TIMESTAMP_COLUMN].diff()

# Remove first row because it has no previous timestamp
time_differences = df["time_difference"].dropna()

# ------------------------------------------------------------
# Most common intervals
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MOST COMMON TIME INTERVALS")
print("=" * 70)

interval_counts = time_differences.value_counts().head(15)

print(interval_counts)

# ------------------------------------------------------------
# Specific gap counts
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GAP ANALYSIS")
print("=" * 70)

gap_1h = (time_differences > pd.Timedelta(hours=1)).sum()
gap_6h = (time_differences > pd.Timedelta(hours=6)).sum()
gap_12h = (time_differences > pd.Timedelta(hours=12)).sum()
gap_24h = (time_differences > pd.Timedelta(hours=24)).sum()

print(f"Gaps greater than 1 hour  : {gap_1h}")
print(f"Gaps greater than 6 hours : {gap_6h}")
print(f"Gaps greater than 12 hours: {gap_12h}")
print(f"Gaps greater than 24 hours: {gap_24h}")

# ------------------------------------------------------------
# Largest gap
# ------------------------------------------------------------

largest_gap = time_differences.max()

print("\n" + "=" * 70)
print("LARGEST TIME GAP")
print("=" * 70)

print(f"Largest gap : {largest_gap}")

# ------------------------------------------------------------
# Top 20 largest gaps
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 LARGEST GAPS")
print("=" * 70)

# Get indices of largest gaps
largest_gap_indices = (
    df["time_difference"]
    .nlargest(20)
    .index
)

gap_records = []

for index in largest_gap_indices:

    current_timestamp = df.loc[index, TIMESTAMP_COLUMN]
    previous_timestamp = df.loc[index - 1, TIMESTAMP_COLUMN]
    difference = df.loc[index, "time_difference"]

    gap_records.append({
        "previous_timestamp": previous_timestamp,
        "current_timestamp": current_timestamp,
        "gap": difference
    })

largest_gaps_df = pd.DataFrame(gap_records)

print(largest_gaps_df.to_string(index=False))

# ------------------------------------------------------------
# Save gap analysis
# ------------------------------------------------------------

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "timestamp_gap_analysis.csv"
)

largest_gaps_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUT FILE")
print("=" * 70)

print(f"Saved to:")
print(OUTPUT_FILE)

# ------------------------------------------------------------
# Finish
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TIMESTAMP ANALYSIS COMPLETE")
print("=" * 70)