import pandas as pd
import os

# ============================================================
# AQUASENSE - CONTINUOUS MONITORING PERIOD ANALYSIS
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
print("AQUASENSE CONTINUOUS PERIOD ANALYSIS")
print("=" * 70)

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
# Convert timestamp
# ------------------------------------------------------------

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    errors="coerce",
    utc=True
)

# Remove invalid timestamps
df = df.dropna(subset=["timestamp_utc"]).copy()

# Sort chronologically
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# ------------------------------------------------------------
# Calculate difference between observations
# ------------------------------------------------------------

df["time_difference"] = df["timestamp_utc"].diff()

# ------------------------------------------------------------
# Define a new continuous segment
#
# We consider a gap greater than 6 hours as a break.
#
# Small gaps are still part of the same monitoring period.
# Major gaps are treated as separate periods.
# ------------------------------------------------------------

SEGMENT_GAP = pd.Timedelta(hours=6)

df["new_segment"] = (
    df["time_difference"] > SEGMENT_GAP
)

# First observation starts Segment 1
df.loc[0, "new_segment"] = True

# Assign segment number
df["segment_id"] = df["new_segment"].cumsum()

# ------------------------------------------------------------
# Analyze each segment
# ------------------------------------------------------------

segments = []

for segment_id, group in df.groupby("segment_id"):

    start_time = group["timestamp_utc"].min()
    end_time = group["timestamp_utc"].max()

    row_count = len(group)

    duration = end_time - start_time

    segments.append({
        "segment_id": int(segment_id),
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "observations": row_count
    })

segments_df = pd.DataFrame(segments)

# ------------------------------------------------------------
# Sort by duration
# ------------------------------------------------------------

segments_df = segments_df.sort_values(
    "duration",
    ascending=False
).reset_index(drop=True)

# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMBER OF CONTINUOUS SEGMENTS")
print("=" * 70)

print(
    f"Continuous segments: {len(segments_df)}"
)

# ------------------------------------------------------------
# Longest segments
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 LONGEST CONTINUOUS SEGMENTS")
print("=" * 70)

print(
    segments_df.head(20).to_string(index=False)
)

# ------------------------------------------------------------
# Calculate duration in days
# ------------------------------------------------------------

segments_df["duration_days"] = (
    segments_df["duration"].dt.total_seconds() / 86400
)

# ------------------------------------------------------------
# Count segments by length
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SEGMENT LENGTH SUMMARY")
print("=" * 70)

less_than_1_day = (
    segments_df["duration_days"] < 1
).sum()

one_to_7_days = (
    (segments_df["duration_days"] >= 1) &
    (segments_df["duration_days"] < 7)
).sum()

one_to_4_weeks = (
    (segments_df["duration_days"] >= 7) &
    (segments_df["duration_days"] < 28)
).sum()

one_to_3_months = (
    (segments_df["duration_days"] >= 28) &
    (segments_df["duration_days"] < 90)
).sum()

three_to_6_months = (
    (segments_df["duration_days"] >= 90) &
    (segments_df["duration_days"] < 180)
).sum()

six_months_or_more = (
    segments_df["duration_days"] >= 180
).sum()

print(f"Less than 1 day       : {less_than_1_day}")
print(f"1 day to 7 days       : {one_to_7_days}")
print(f"7 days to 4 weeks     : {one_to_4_weeks}")
print(f"4 weeks to 3 months   : {one_to_3_months}")
print(f"3 to 6 months         : {three_to_6_months}")
print(f"6 months or more      : {six_months_or_more}")

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "continuous_monitoring_periods.csv"
)

segments_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUT FILE")
print("=" * 70)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)

print("\n" + "=" * 70)
print("CONTINUOUS PERIOD ANALYSIS COMPLETE")
print("=" * 70)