import os
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

print("=" * 70)
print("STEP 30 - MISSINGNESS SENSITIVITY ANALYSIS")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

PREPARED_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "prepared_data"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "missingness_sensitivity"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

HORIZONS = ["12h", "24h"]

# Same XGBoost settings used previously
XGB_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "auc",
    "n_jobs": -1,
    "random_state": 42
}

# ============================================================
# LOAD RAW PREPARED DATA
# ============================================================

print()
print("Loading prepared chronological datasets...")

train_df = pd.read_csv(
    os.path.join(
        PREPARED_DIR,
        "train.csv"
    )
)

validation_df = pd.read_csv(
    os.path.join(
        PREPARED_DIR,
        "validation.csv"
    )
)

test_df = pd.read_csv(
    os.path.join(
        PREPARED_DIR,
        "test.csv"
    )
)

print()
print("Train shape      :", train_df.shape)
print("Validation shape :", validation_df.shape)
print("Test shape       :", test_df.shape)

# ============================================================
# IDENTIFY ORIGINAL ML FEATURES
# ============================================================

feature_list_file = os.path.join(
    PREPARED_DIR,
    "feature_list.csv"
)

feature_df = pd.read_csv(
    feature_list_file
)

original_features = feature_df["feature"].tolist()

print()
print("Original feature count:", len(original_features))

# Safety check
missing_features = [
    feature
    for feature in original_features
    if feature not in train_df.columns
]

if missing_features:

    raise ValueError(
        "Some expected features are missing from train.csv:\n"
        + "\n".join(missing_features)
    )

# ============================================================
# FUNCTION: EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
    horizon
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Use 0.5 here only as a common comparison threshold.
    # ROC-AUC and PR-AUC are threshold-independent.
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    results = {
        "horizon": horizon,
        "model": model_name,
        "features_used": X_test.shape[1],
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities
        ),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp
    }

    return results


# ============================================================
# RUN EXPERIMENT
# ============================================================

all_results = []

for horizon in HORIZONS:

    print()
    print("=" * 70)
    print(f"HORIZON: {horizon}")
    print("=" * 70)

    target = f"future_risk_{horizon}"

    # --------------------------------------------------------
    # SELECT VALID ROWS
    # --------------------------------------------------------

    train_valid = train_df[
        train_df[target].notna()
    ].copy()

    validation_valid = validation_df[
        validation_df[target].notna()
    ].copy()

    test_valid = test_df[
        test_df[target].notna()
    ].copy()

    print()
    print("Valid rows:")
    print("Train      :", len(train_valid))
    print("Validation :", len(validation_valid))
    print("Test       :", len(test_valid))

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y_train = (
        train_valid[target]
        .astype(int)
        .values
    )

    y_validation = (
        validation_valid[target]
        .astype(int)
        .values
    )

    y_test = (
        test_valid[target]
        .astype(int)
        .values
    )

    # ========================================================
    # MODEL A
    # ORIGINAL + MISSINGNESS INDICATORS
    # ========================================================

    print()
    print("MODEL A")
    print("Original features + missingness indicators")

    X_train_all = train_valid[
        original_features
    ].copy()

    X_validation_all = validation_valid[
        original_features
    ].copy()

    X_test_all = test_valid[
        original_features
    ].copy()

    # --------------------------------------------------------
    # MEDIAN IMPUTATION + INDICATORS
    # --------------------------------------------------------

    imputer_with_indicator = SimpleImputer(
        strategy="median",
        add_indicator=True
    )

    X_train_A = imputer_with_indicator.fit_transform(
        X_train_all
    )

    X_validation_A = imputer_with_indicator.transform(
        X_validation_all
    )

    X_test_A = imputer_with_indicator.transform(
        X_test_all
    )

    print(
        "Model A feature count:",
        X_train_A.shape[1]
    )

    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    negative_count = np.sum(
        y_train == 0
    )

    positive_count = np.sum(
        y_train == 1
    )

    scale_pos_weight = (
        negative_count /
        positive_count
    )

    print(
        "scale_pos_weight:",
        scale_pos_weight
    )

    # --------------------------------------------------------
    # TRAIN MODEL A
    # --------------------------------------------------------

    model_A = xgb.XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight
    )

    model_A.fit(
        X_train_A,
        y_train,
        eval_set=[
            (
                X_validation_A,
                y_validation
            )
        ],
        verbose=False
    )

    result_A = evaluate_model(
        model_A,
        X_test_A,
        y_test,
        "with_missing_indicators",
        horizon
    )

    all_results.append(
        result_A
    )

    print()
    print("Model A test results:")

    for key, value in result_A.items():

        if key not in [
            "horizon",
            "model"
        ]:

            print(
                f"{key}: {value}"
            )

    # ========================================================
    # MODEL B
    # ORIGINAL FEATURES ONLY
    # ========================================================

    print()
    print("MODEL B")
    print("Original features only")

    imputer_without_indicator = SimpleImputer(
        strategy="median",
        add_indicator=False
    )

    X_train_B = imputer_without_indicator.fit_transform(
        X_train_all
    )

    X_validation_B = imputer_without_indicator.transform(
        X_validation_all
    )

    X_test_B = imputer_without_indicator.transform(
        X_test_all
    )

    print(
        "Model B feature count:",
        X_train_B.shape[1]
    )

    # --------------------------------------------------------
    # TRAIN MODEL B
    # --------------------------------------------------------

    model_B = xgb.XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight
    )

    model_B.fit(
        X_train_B,
        y_train,
        eval_set=[
            (
                X_validation_B,
                y_validation
            )
        ],
        verbose=False
    )

    result_B = evaluate_model(
        model_B,
        X_test_B,
        y_test,
        "without_missing_indicators",
        horizon
    )

    all_results.append(
        result_B
    )

    print()
    print("Model B test results:")

    for key, value in result_B.items():

        if key not in [
            "horizon",
            "model"
        ]:

            print(
                f"{key}: {value}"
            )

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_file = os.path.join(
    OUTPUT_DIR,
    "missingness_sensitivity_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)

comparison_columns = [
    "horizon",
    "model",
    "features_used",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "false_positive",
    "false_negative",
    "true_positive"
]

print(
    results_df[
        comparison_columns
    ].to_string(index=False)
)

# ============================================================
# CALCULATE DIFFERENCES
# ============================================================

comparison_rows = []

for horizon in HORIZONS:

    with_indicators = results_df[
        (results_df["horizon"] == horizon) &
        (
            results_df["model"] ==
            "with_missing_indicators"
        )
    ].iloc[0]

    without_indicators = results_df[
        (results_df["horizon"] == horizon) &
        (
            results_df["model"] ==
            "without_missing_indicators"
        )
    ].iloc[0]

    comparison_rows.append({
        "horizon": horizon,

        "f1_with_indicators":
            with_indicators["f1"],

        "f1_without_indicators":
            without_indicators["f1"],

        "f1_difference":
            with_indicators["f1"]
            -
            without_indicators["f1"],

        "roc_auc_with_indicators":
            with_indicators["roc_auc"],

        "roc_auc_without_indicators":
            without_indicators["roc_auc"],

        "roc_auc_difference":
            with_indicators["roc_auc"]
            -
            without_indicators["roc_auc"],

        "pr_auc_with_indicators":
            with_indicators["pr_auc"],

        "pr_auc_without_indicators":
            without_indicators["pr_auc"],

        "pr_auc_difference":
            with_indicators["pr_auc"]
            -
            without_indicators["pr_auc"]
    })

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_file = os.path.join(
    OUTPUT_DIR,
    "missingness_model_comparison.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)

print()
print("Performance differences:")
print(
    comparison_df.to_string(
        index=False
    )
)

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 30 COMPLETED")
print("=" * 70)

print()
print("Results saved:")
print(results_file)
print(comparison_file)