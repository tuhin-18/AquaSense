import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\tusha\Documents\AquaSense")

INPUT_FILE = (
    PROJECT_ROOT
    / "ml_outputs"
    / "ml_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_outputs"
    / "prepared_data"
)


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "future_risk_1h",
    "future_risk_6h",
    "future_risk_12h",
    "future_risk_24h"
]


# ============================================================
# IDENTIFICATION COLUMNS
# ============================================================

ID_COLUMNS = [
    "timestamp_utc",
    "segment_id"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 11 - PREPARE ML DATASET")
print("=" * 60)

print("\nLoading feature dataset...")

df = pd.read_csv(INPUT_FILE)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    utc=True
)

df = df.sort_values(
    "timestamp_utc"
).reset_index(drop=True)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")

print(
    f"Date range: "
    f"{df['timestamp_utc'].min()} "
    f"to "
    f"{df['timestamp_utc'].max()}"
)


# ============================================================
# CHECK TARGETS
# ============================================================

print("\nChecking target columns...")

for target in TARGETS:

    if target not in df.columns:
        raise ValueError(
            f"Missing target column: {target}"
        )

print("All target columns found.")


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

excluded_columns = (
    ID_COLUMNS
    + TARGETS
)

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]

# Keep only numeric ML features
numeric_features = []

for column in feature_columns:

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):
        numeric_features.append(column)

feature_columns = numeric_features

print(
    f"\nNumeric ML features: "
    f"{len(feature_columns)}"
)


# ============================================================
# REMOVE COMPLETELY EMPTY FEATURES
# ============================================================

completely_empty = [
    column
    for column in feature_columns
    if df[column].isna().all()
]

if completely_empty:

    print(
        "\nRemoving completely empty features:"
    )

    for column in completely_empty:
        print(f"  - {column}")

    feature_columns = [
        column
        for column in feature_columns
        if column not in completely_empty
    ]


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

print("\nCreating chronological split...")

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train_df = df.iloc[
    :train_end
].copy()

validation_df = df.iloc[
    train_end:validation_end
].copy()

test_df = df.iloc[
    validation_end:
].copy()


# ============================================================
# DISPLAY SPLIT INFORMATION
# ============================================================

print("\nDataset split:")
print("-" * 60)

print(
    f"Training   : {len(train_df):,} rows "
    f"({len(train_df) / n * 100:.1f}%)"
)

print(
    f"Validation : {len(validation_df):,} rows "
    f"({len(validation_df) / n * 100:.1f}%)"
)

print(
    f"Test       : {len(test_df):,} rows "
    f"({len(test_df) / n * 100:.1f}%)"
)


print("\nDate ranges:")
print("-" * 60)

print(
    f"Training:"
    f"\n  {train_df['timestamp_utc'].min()}"
    f"\n  {train_df['timestamp_utc'].max()}"
)

print(
    f"\nValidation:"
    f"\n  {validation_df['timestamp_utc'].min()}"
    f"\n  {validation_df['timestamp_utc'].max()}"
)

print(
    f"\nTest:"
    f"\n  {test_df['timestamp_utc'].min()}"
    f"\n  {test_df['timestamp_utc'].max()}"
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print("-" * 60)

for target in TARGETS:

    print(f"\n{target}")

    for name, subset in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Test", test_df)
    ]:

        valid = subset[target].dropna()

        positive = int(
            (valid == 1).sum()
        )

        negative = int(
            (valid == 0).sum()
        )

        total = positive + negative

        if total > 0:
            percentage = (
                positive / total * 100
            )
        else:
            percentage = 0

        print(
            f"  {name:<11}: "
            f"Risk=1 {positive:>5,} | "
            f"Risk=0 {negative:>6,} | "
            f"Risk % {percentage:>5.2f}%"
        )


# ============================================================
# FEATURE MISSINGNESS
# ============================================================

print("\nFeature missingness:")
print("-" * 60)

missing_percentage = (
    train_df[feature_columns]
    .isna()
    .mean()
    * 100
)

missing_percentage = (
    missing_percentage
    .sort_values(ascending=False)
)

print(
    missing_percentage.head(20)
    .to_string()
)


# ============================================================
# SAVE DATASETS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Save complete split datasets.
# We are NOT imputing yet.

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

validation_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# Save feature list
feature_list = pd.DataFrame({
    "feature": feature_columns
})

feature_list.to_csv(
    OUTPUT_DIR / "feature_list.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ML DATASET PREPARATION COMPLETE")
print("=" * 60)

print("\nFiles saved:")

print(
    OUTPUT_DIR / "train.csv"
)

print(
    OUTPUT_DIR / "validation.csv"
)

print(
    OUTPUT_DIR / "test.csv"
)

print(
    OUTPUT_DIR / "feature_list.csv"
)

print(
    f"\nNumber of ML features: "
    f"{len(feature_columns)}"
)

print("\nIMPORTANT:")
print(
    "Missing values have NOT been filled yet."
)

print(
    "Preprocessing will be fitted using "
    "training data only."
)