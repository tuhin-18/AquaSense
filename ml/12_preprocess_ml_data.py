import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.impute import SimpleImputer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\tusha\Documents\AquaSense"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "ml_outputs"
    / "prepared_data"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_outputs"
    / "processed_data"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = INPUT_DIR / "train.csv"
VALIDATION_FILE = INPUT_DIR / "validation.csv"
TEST_FILE = INPUT_DIR / "test.csv"
FEATURE_FILE = INPUT_DIR / "feature_list.csv"


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
# LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 12 - PREPROCESS ML DATA")
print("=" * 60)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

feature_list = pd.read_csv(FEATURE_FILE)

feature_columns = feature_list["feature"].tolist()

print(
    f"Training rows   : {len(train_df):,}"
)

print(
    f"Validation rows : {len(validation_df):,}"
)

print(
    f"Test rows       : {len(test_df):,}"
)

print(
    f"Features        : {len(feature_columns):,}"
)


# ============================================================
# CREATE FEATURE MATRICES
# ============================================================

print("\nPreparing feature matrices...")

X_train = train_df[feature_columns].copy()

X_validation = validation_df[feature_columns].copy()

X_test = test_df[feature_columns].copy()


# ============================================================
# MEDIAN IMPUTATION
# ============================================================

print(
    "\nFitting median imputer on TRAINING data only..."
)

imputer = SimpleImputer(
    strategy="median",
    add_indicator=True
)

X_train_processed = imputer.fit_transform(
    X_train
)

X_validation_processed = imputer.transform(
    X_validation
)

X_test_processed = imputer.transform(
    X_test
)


# ============================================================
# CHECK FEATURE RESULTS
# ============================================================

print("\nAfter imputation:")

print(
    f"Training shape   : "
    f"{X_train_processed.shape}"
)

print(
    f"Validation shape : "
    f"{X_validation_processed.shape}"
)

print(
    f"Test shape       : "
    f"{X_test_processed.shape}"
)

print("\nRemaining missing values:")

print(
    f"Training   : "
    f"{np.isnan(X_train_processed).sum():,}"
)

print(
    f"Validation : "
    f"{np.isnan(X_validation_processed).sum():,}"
)

print(
    f"Test       : "
    f"{np.isnan(X_test_processed).sum():,}"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE PROCESSED FEATURE NAMES
# ============================================================

processed_feature_names = list(
    feature_columns
)

indicator_features = []

if hasattr(imputer, "indicator_"):

    indicator_indices = (
        imputer.indicator_.features_
    )

    for index in indicator_indices:

        indicator_features.append(
            f"{feature_columns[index]}_missing"
        )

processed_feature_names.extend(
    indicator_features
)

pd.DataFrame({
    "feature": processed_feature_names
}).to_csv(
    OUTPUT_DIR / "processed_feature_list.csv",
    index=False
)


# ============================================================
# SAVE TARGETS WITH ALIGNED FEATURES
# ============================================================

print(
    "\nPreparing target-specific datasets..."
)

for target in TARGETS:

    print("\n" + "-" * 60)
    print(f"TARGET: {target}")
    print("-" * 60)

    # --------------------------------------------------------
    # Create masks for VALID target values
    # --------------------------------------------------------

    train_mask = train_df[target].notna().to_numpy()

    validation_mask = (
        validation_df[target].notna().to_numpy()
    )

    test_mask = test_df[target].notna().to_numpy()


    # --------------------------------------------------------
    # Apply EXACT SAME mask to X and y
    # --------------------------------------------------------

    X_train_target = (
        X_train_processed[train_mask]
    )

    X_validation_target = (
        X_validation_processed[validation_mask]
    )

    X_test_target = (
        X_test_processed[test_mask]
    )


    y_train_target = (
        train_df.loc[train_mask, target]
        .to_numpy()
        .astype(int)
    )

    y_validation_target = (
        validation_df.loc[
            validation_mask,
            target
        ]
        .to_numpy()
        .astype(int)
    )

    y_test_target = (
        test_df.loc[
            test_mask,
            target
        ]
        .to_numpy()
        .astype(int)
    )


    # --------------------------------------------------------
    # Verify alignment
    # --------------------------------------------------------

    assert (
        X_train_target.shape[0]
        == y_train_target.shape[0]
    )

    assert (
        X_validation_target.shape[0]
        == y_validation_target.shape[0]
    )

    assert (
        X_test_target.shape[0]
        == y_test_target.shape[0]
    )


    # --------------------------------------------------------
    # Print alignment information
    # --------------------------------------------------------

    print(
        f"Training valid rows   : "
        f"{len(y_train_target):,}"
    )

    print(
        f"Validation valid rows : "
        f"{len(y_validation_target):,}"
    )

    print(
        f"Test valid rows       : "
        f"{len(y_test_target):,}"
    )

    print(
        f"Training positives    : "
        f"{np.sum(y_train_target == 1):,}"
    )

    print(
        f"Validation positives  : "
        f"{np.sum(y_validation_target == 1):,}"
    )

    print(
        f"Test positives        : "
        f"{np.sum(y_test_target == 1):,}"
    )

    print(
        "X/y alignment         : PASSED"
    )


    # --------------------------------------------------------
    # Save target-specific feature matrices
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR
        / f"X_train_future_risk_{target.split('_')[-1]}.npy",
        X_train_target
    )

    np.save(
        OUTPUT_DIR
        / f"X_validation_future_risk_{target.split('_')[-1]}.npy",
        X_validation_target
    )

    np.save(
        OUTPUT_DIR
        / f"X_test_future_risk_{target.split('_')[-1]}.npy",
        X_test_target
    )


    # --------------------------------------------------------
    # Save target arrays
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR
        / f"y_train_{target}.npy",
        y_train_target
    )

    np.save(
        OUTPUT_DIR
        / f"y_validation_{target}.npy",
        y_validation_target
    )

    np.save(
        OUTPUT_DIR
        / f"y_test_{target}.npy",
        y_test_target
    )


# ============================================================
# SAVE IMPUTER
# ============================================================

joblib.dump(
    imputer,
    MODEL_DIR / "median_imputer.joblib"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(
    "\nOriginal features:"
    f" {len(feature_columns)}"
)

print(
    "Missing indicators:"
    f" {len(indicator_features)}"
)

print(
    "Final feature count:"
    f" {len(processed_feature_names)}"
)

print("\nRemaining NaN values:")

print(
    f"Training   : "
    f"{np.isnan(X_train_processed).sum():,}"
)

print(
    f"Validation : "
    f"{np.isnan(X_validation_processed).sum():,}"
)

print(
    f"Test       : "
    f"{np.isnan(X_test_processed).sum():,}"
)

print("\nSaved processed data to:")
print(OUTPUT_DIR)

print("\nSaved imputer to:")
print(
    MODEL_DIR / "median_imputer.joblib"
)

print("\nIMPORTANT:")
print(
    "Each future-risk target now has its own "
    "aligned X and y datasets."
)