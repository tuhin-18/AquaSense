import pandas as pd
import os

# ============================================================
# AQUASENSE - WATER QUALITY PARAMETER ANALYSIS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


# ============================================================
# PARAMETERS WE WILL USE FOR THE RESEARCH
# ============================================================

PARAMETERS = [
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "do_saturation",
    "turbidity"
]


print("=" * 70)
print("AQUASENSE WATER QUALITY PARAMETER ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)

print("Dataset loaded successfully.")


# ============================================================
# CONVERT TIMESTAMP
# ============================================================

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


# ============================================================
# CHECK THAT REQUIRED PARAMETERS EXIST
# ============================================================

print("\n" + "=" * 70)
print("CHECKING PARAMETERS")
print("=" * 70)

for parameter in PARAMETERS:

    if parameter in df.columns:
        print(f"[OK] {parameter}")
    else:
        print(f"[MISSING] {parameter}")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Total observations : {len(df)}")

print(
    f"Start time         : "
    f"{df['timestamp_utc'].min()}"
)

print(
    f"End time           : "
    f"{df['timestamp_utc'].max()}"
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_rows = []

for parameter in PARAMETERS:

    missing_count = df[parameter].isna().sum()

    total_count = len(df)

    missing_percentage = (
        missing_count / total_count
    ) * 100

    missing_rows.append({
        "parameter": parameter,
        "missing_values": missing_count,
        "missing_percentage": missing_percentage
    })

    print(
        f"{parameter:20s} : "
        f"{missing_count:6d} "
        f"({missing_percentage:.2f}%)"
    )


missing_df = pd.DataFrame(missing_rows)

missing_output = os.path.join(
    OUTPUT_DIR,
    "water_quality_missing_values.csv"
)

missing_df.to_csv(
    missing_output,
    index=False
)


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

summary = df[PARAMETERS].describe(
    percentiles=[
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99
    ]
)

print(
    summary.to_string()
)


summary_output = os.path.join(
    OUTPUT_DIR,
    "water_quality_summary.csv"
)

summary.to_csv(
    summary_output
)


# ============================================================
# MEDIAN AND STANDARD DEVIATION
# ============================================================

print("\n" + "=" * 70)
print("CENTRAL TENDENCY AND VARIABILITY")
print("=" * 70)

for parameter in PARAMETERS:

    median = df[parameter].median()

    mean = df[parameter].mean()

    std = df[parameter].std()

    print(
        f"\n{parameter}"
    )

    print(
        f"  Mean   : {mean:.4f}"
    )

    print(
        f"  Median : {median:.4f}"
    )

    print(
        f"  Std    : {std:.4f}"
    )


# ============================================================
# EXTREME VALUES
# ============================================================

print("\n" + "=" * 70)
print("EXTREME VALUES")
print("=" * 70)

for parameter in PARAMETERS:

    print(f"\n{parameter}")

    print(
        f"  Minimum : "
        f"{df[parameter].min()}"
    )

    print(
        f"  Maximum : "
        f"{df[parameter].max()}"
    )


# ============================================================
# 1% AND 99% QUANTILES
# ============================================================

print("\n" + "=" * 70)
print("1% - 99% RANGE")
print("=" * 70)

for parameter in PARAMETERS:

    q01 = df[parameter].quantile(0.01)

    q99 = df[parameter].quantile(0.99)

    print(
        f"{parameter:20s} : "
        f"1% = {q01:.4f}, "
        f"99% = {q99:.4f}"
    )


# ============================================================
# POSSIBLE EXTREME OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("EXTREME OBSERVATION COUNTS")
print("=" * 70)

extreme_rows = []

for parameter in PARAMETERS:

    q01 = df[parameter].quantile(0.01)

    q99 = df[parameter].quantile(0.99)

    below_01 = (
        df[parameter] < q01
    ).sum()

    above_99 = (
        df[parameter] > q99
    ).sum()

    extreme_rows.append({
        "parameter": parameter,
        "below_1_percent": below_01,
        "above_99_percent": above_99
    })

    print(
        f"\n{parameter}"
    )

    print(
        f"  Below 1% quantile : {below_01}"
    )

    print(
        f"  Above 99% quantile: {above_99}"
    )


extreme_df = pd.DataFrame(
    extreme_rows
)

extreme_output = os.path.join(
    OUTPUT_DIR,
    "water_quality_extreme_counts.csv"
)

extreme_df.to_csv(
    extreme_output,
    index=False
)


# ============================================================
# SAVE CLEAN NUMERIC DATA FOR LATER ANALYSIS
# ============================================================

selected_columns = [
    "timestamp_utc"
] + PARAMETERS

analysis_df = df[
    selected_columns
].copy()

analysis_output = os.path.join(
    OUTPUT_DIR,
    "water_quality_analysis_data.csv"
)

analysis_df.to_csv(
    analysis_output,
    index=False
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"\nMissing values:\n"
    f"{missing_output}"
)

print(
    f"\nSummary statistics:\n"
    f"{summary_output}"
)

print(
    f"\nExtreme value counts:\n"
    f"{extreme_output}"
)

print(
    f"\nAnalysis dataset:\n"
    f"{analysis_output}"
)


print("\n" + "=" * 70)
print("WATER QUALITY ANALYSIS COMPLETE")
print("=" * 70)