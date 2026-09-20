import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

DATA_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "water_quality_hourly_regularized.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "temporal_shift_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TURBIDITY_THRESHOLD = 319.81


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 18 - TEMPORAL DISTRIBUTION SHIFT ANALYSIS")
print("=" * 70)

print("\nLoading hourly dataset...")

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["timestamp_utc"]
)

df = df.sort_values("timestamp_utc").reset_index(drop=True)

print(f"Total rows: {len(df)}")


# ============================================================
# ADD YEAR
# ============================================================

df["year"] = df["timestamp_utc"].dt.year


# ============================================================
# DEFINE CHRONOLOGICAL PERIODS
# ============================================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)


df["period"] = "test"

df.loc[
    :train_end - 1,
    "period"
] = "train"

df.loc[
    train_end:validation_end - 1,
    "period"
] = "validation"


# ============================================================
# PERIOD SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PERIOD SUMMARY")
print("=" * 70)

for period in ["train", "validation", "test"]:

    subset = df[
        df["period"] == period
    ]

    turbidity = subset["turbidity"].dropna()

    high_risk = (
        turbidity > TURBIDITY_THRESHOLD
    ).sum()

    print(f"\n{period.upper()}")

    print(f"Rows                 : {len(subset)}")
    print(f"Start                : {subset['timestamp_utc'].min()}")
    print(f"End                  : {subset['timestamp_utc'].max()}")

    print(
        f"Mean turbidity       : "
        f"{turbidity.mean():.2f}"
    )

    print(
        f"Median turbidity     : "
        f"{turbidity.median():.2f}"
    )

    print(
        f"95th percentile      : "
        f"{turbidity.quantile(0.95):.2f}"
    )

    print(
        f"99th percentile      : "
        f"{turbidity.quantile(0.99):.2f}"
    )

    print(
        f"Maximum turbidity    : "
        f"{turbidity.max():.2f}"
    )

    print(
        f"High-risk observations: "
        f"{high_risk}"
    )


# ============================================================
# YEARLY TURBIDITY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("YEARLY TURBIDITY SUMMARY")
print("=" * 70)

yearly = (
    df
    .groupby("year")["turbidity"]
    .agg(
        observations="count",
        mean="mean",
        median="median",
        p95=lambda x: x.quantile(0.95),
        p99=lambda x: x.quantile(0.99),
        maximum="max"
    )
    .reset_index()
)

yearly["high_risk_count"] = (
    df
    .groupby("year")["turbidity"]
    .apply(
        lambda x: (x > TURBIDITY_THRESHOLD).sum()
    )
    .values
)

yearly["high_risk_rate_percent"] = (
    yearly["high_risk_count"]
    / yearly["observations"]
    * 100
)


print(
    yearly.to_string(index=False)
)


# ============================================================
# SAVE YEARLY SUMMARY
# ============================================================

yearly_file = os.path.join(
    OUTPUT_DIR,
    "yearly_turbidity_summary.csv"
)

yearly.to_csv(
    yearly_file,
    index=False
)


# ============================================================
# PERIOD + YEAR SUMMARY
# ============================================================

period_yearly = (
    df
    .groupby(
        ["period", "year"]
    )["turbidity"]
    .agg(
        observations="count",
        mean="mean",
        median="median",
        p95=lambda x: x.quantile(0.95),
        p99=lambda x: x.quantile(0.99),
        maximum="max"
    )
    .reset_index()
)

period_yearly["high_risk_count"] = (
    df
    .groupby(
        ["period", "year"]
    )["turbidity"]
    .apply(
        lambda x: (x > TURBIDITY_THRESHOLD).sum()
    )
    .values
)


period_yearly["high_risk_rate_percent"] = (
    period_yearly["high_risk_count"]
    / period_yearly["observations"]
    * 100
)


# ============================================================
# SAVE PERIOD/YEAR SUMMARY
# ============================================================

period_yearly_file = os.path.join(
    OUTPUT_DIR,
    "period_yearly_turbidity_summary.csv"
)

period_yearly.to_csv(
    period_yearly_file,
    index=False
)


# ============================================================
# HIGH-TURBIDITY EVENTS BY YEAR
# ============================================================

high_turbidity = df[
    df["turbidity"] > TURBIDITY_THRESHOLD
].copy()


high_turbidity_file = os.path.join(
    OUTPUT_DIR,
    "high_turbidity_by_year.csv"
)

high_turbidity[
    [
        "timestamp_utc",
        "turbidity",
        "year",
        "period"
    ]
].to_csv(
    high_turbidity_file,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)

print("Files saved:")

print(yearly_file)
print(period_yearly_file)
print(high_turbidity_file)

print("\nStep 18 completed successfully.")

print("=" * 70)