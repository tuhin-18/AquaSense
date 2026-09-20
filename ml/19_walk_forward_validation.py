import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "processed_data"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "walk_forward_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

HORIZONS = [
    "1h",
    "6h",
    "12h",
    "24h"
]

# Four chronological validation folds
N_SPLITS = 4

# Use 6-hour gap between training and validation.
# This prevents the closest observations around the
# split from being used directly on both sides.
GAP = 6

# Validation window = 15% of available observations
# for each horizon.
VALIDATION_FRACTION = 0.15

RANDOM_STATE = 42


# ============================================================
# XGBOOST SETTINGS
# ============================================================

def create_model(y_train):

    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)

    if positive_count == 0:
        raise ValueError(
            "Training fold contains no positive samples."
        )

    scale_pos_weight = (
        negative_count / positive_count
    )

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    return model


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 19 - WALK-FORWARD XGBOOST VALIDATION")
print("=" * 70)

print("\nEach fold trains only on earlier observations.")
print("A 6-hour gap is kept between training and validation.")


all_results = []


# ============================================================
# PROCESS EACH HORIZON
# ============================================================

for horizon in HORIZONS:

    print("\n" + "=" * 70)
    print(f"HORIZON: {horizon}")
    print("=" * 70)


    # --------------------------------------------------------
    # Load complete processed feature matrix
    # --------------------------------------------------------

    X_path = os.path.join(
        PROCESSED_DIR,
        f"X_train_future_risk_{horizon}.npy"
    )

    y_path = os.path.join(
        PROCESSED_DIR,
        f"y_train_future_risk_{horizon}.npy"
    )


    X = np.load(X_path)
    y = np.load(y_path)


    print(f"Total available samples: {len(y)}")
    print(f"Total positive samples : {int(y.sum())}")


    # --------------------------------------------------------
    # Determine validation size
    # --------------------------------------------------------

    validation_size = int(
        len(y) * VALIDATION_FRACTION
    )

    minimum_training_size = (
        validation_size + GAP
    )

    if len(y) <= (
        minimum_training_size * N_SPLITS
    ):
        raise ValueError(
            "Dataset is too small for the requested "
            "walk-forward configuration."
        )


    # --------------------------------------------------------
    # Create chronological folds manually
    #
    # This gives us:
    #
    # Fold 1:
    # earlier training -> validation
    #
    # Fold 2:
    # larger training  -> later validation
    #
    # etc.
    # --------------------------------------------------------

    fold_results = []


    for fold in range(N_SPLITS):

        # End of validation period
        validation_end = (
            len(y)
            - (N_SPLITS - 1 - fold) * validation_size
        )

        validation_start = (
            validation_end
            - validation_size
        )

        train_end = (
            validation_start
            - GAP
        )


        # Safety check
        if train_end <= 0:
            print(
                f"\nFold {fold + 1}: skipped "
                f"(insufficient training data)"
            )
            continue


        X_train = X[:train_end]
        y_train = y[:train_end]

        X_val = X[validation_start:validation_end]
        y_val = y[validation_start:validation_end]


        print("\n" + "-" * 70)
        print(f"Fold {fold + 1}")
        print("-" * 70)

        print(
            f"Training samples   : {len(y_train)}"
        )

        print(
            f"Training positives  : {int(y_train.sum())}"
        )

        print(
            f"Validation samples : {len(y_val)}"
        )

        print(
            f"Validation positives: {int(y_val.sum())}"
        )

        print(
            f"Gap samples        : {GAP}"
        )


        # ----------------------------------------------------
        # Train new XGBoost model
        # ----------------------------------------------------

        model = create_model(y_train)

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val)
            ],
            verbose=False
        )


        # ----------------------------------------------------
        # Probability prediction
        # ----------------------------------------------------

        y_probability = (
            model.predict_proba(X_val)[:, 1]
        )


        # ----------------------------------------------------
        # Use 0.5 only for fold-level baseline metrics
        #
        # We are NOT selecting the final pump threshold here.
        # This is only to compare model behavior across folds.
        # ----------------------------------------------------

        y_pred = (
            y_probability >= 0.50
        ).astype(int)


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_val,
            y_pred
        )

        precision = precision_score(
            y_val,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            y_pred,
            zero_division=0
        )


        # ROC-AUC requires both classes
        if len(np.unique(y_val)) == 2:

            roc_auc = roc_auc_score(
                y_val,
                y_probability
            )

            pr_auc = average_precision_score(
                y_val,
                y_probability
            )

        else:

            roc_auc = np.nan
            pr_auc = np.nan


        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            y_pred,
            labels=[0, 1]
        ).ravel()


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        result = {

            "horizon": horizon,

            "fold": fold + 1,

            "train_samples": len(y_train),

            "train_positives": int(
                y_train.sum()
            ),

            "validation_samples": len(y_val),

            "validation_positives": int(
                y_val.sum()
            ),

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "roc_auc": roc_auc,

            "pr_auc": pr_auc,

            "true_negatives": tn,

            "false_positives": fp,

            "false_negatives": fn,

            "true_positives": tp,

            "predicted_positive": int(
                y_pred.sum()
            )
        }


        fold_results.append(result)
        all_results.append(result)


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"Accuracy      : {accuracy:.4f}"
        )

        print(
            f"Precision     : {precision:.4f}"
        )

        print(
            f"Recall        : {recall:.4f}"
        )

        print(
            f"F1-score      : {f1:.4f}"
        )

        print(
            f"ROC-AUC       : {roc_auc:.4f}"
        )

        print(
            f"PR-AUC        : {pr_auc:.4f}"
        )

        print(
            f"False positives: {fp}"
        )

        print(
            f"False negatives: {fn}"
        )


    # ========================================================
    # HORIZON SUMMARY
    # ========================================================

    fold_df = pd.DataFrame(
        fold_results
    )


    if len(fold_df) > 0:

        print("\n" + "-" * 70)
        print(f"{horizon} WALK-FORWARD SUMMARY")
        print("-" * 70)


        metric_columns = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc"
        ]


        for metric in metric_columns:

            mean_value = (
                fold_df[metric].mean()
            )

            std_value = (
                fold_df[metric].std()
            )

            print(
                f"{metric.upper():12s}: "
                f"{mean_value:.4f} "
                f"+/- {std_value:.4f}"
            )


# ============================================================
# SAVE ALL FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_file = os.path.join(
    OUTPUT_DIR,
    "walk_forward_fold_results.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# CREATE SUMMARY
# ============================================================

summary_rows = []


for horizon in HORIZONS:

    subset = results_df[
        results_df["horizon"] == horizon
    ]


    if len(subset) == 0:
        continue


    summary_rows.append({

        "horizon": horizon,

        "mean_accuracy":
            subset["accuracy"].mean(),

        "std_accuracy":
            subset["accuracy"].std(),

        "mean_precision":
            subset["precision"].mean(),

        "std_precision":
            subset["precision"].std(),

        "mean_recall":
            subset["recall"].mean(),

        "std_recall":
            subset["recall"].std(),

        "mean_f1":
            subset["f1"].mean(),

        "std_f1":
            subset["f1"].std(),

        "mean_roc_auc":
            subset["roc_auc"].mean(),

        "std_roc_auc":
            subset["roc_auc"].std(),

        "mean_pr_auc":
            subset["pr_auc"].mean(),

        "std_pr_auc":
            subset["pr_auc"].std()
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_file = os.path.join(
    OUTPUT_DIR,
    "walk_forward_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("FINAL WALK-FORWARD SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 70)

print("Files saved:")

print(results_file)

print(summary_file)

print("\nStep 19 completed successfully.")

print("=" * 70)