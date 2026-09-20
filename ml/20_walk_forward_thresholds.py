import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
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
    "walk_forward_thresholds"
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

N_SPLITS = 4

GAP = 6

VALIDATION_FRACTION = 0.15

RANDOM_STATE = 42

THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.05
)


# ============================================================
# MODEL
# ============================================================

def create_model(y_train):

    positive_count = np.sum(
        y_train == 1
    )

    negative_count = np.sum(
        y_train == 0
    )

    scale_pos_weight = (
        negative_count / positive_count
    )

    return XGBClassifier(
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


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 20 - WALK-FORWARD THRESHOLD STABILITY ANALYSIS")
print("=" * 70)


all_results = []


# ============================================================
# EACH HORIZON
# ============================================================

for horizon in HORIZONS:

    print("\n" + "=" * 70)
    print(f"HORIZON: {horizon}")
    print("=" * 70)


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


    validation_size = int(
        len(y) * VALIDATION_FRACTION
    )


    for fold in range(N_SPLITS):

        validation_end = (
            len(y)
            - (N_SPLITS - 1 - fold)
            * validation_size
        )

        validation_start = (
            validation_end
            - validation_size
        )

        train_end = (
            validation_start
            - GAP
        )


        X_train = X[:train_end]
        y_train = y[:train_end]

        X_val = X[
            validation_start:validation_end
        ]

        y_val = y[
            validation_start:validation_end
        ]


        print("\n" + "-" * 70)
        print(
            f"Fold {fold + 1}"
        )
        print("-" * 70)

        print(
            f"Training samples   : "
            f"{len(y_train)}"
        )

        print(
            f"Training positives  : "
            f"{int(y_train.sum())}"
        )

        print(
            f"Validation samples : "
            f"{len(y_val)}"
        )

        print(
            f"Validation positives: "
            f"{int(y_val.sum())}"
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model = create_model(
            y_train
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val)
            ],
            verbose=False
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        y_probability = (
            model.predict_proba(
                X_val
            )[:, 1]
        )


        # ----------------------------------------------------
        # TEST EVERY THRESHOLD
        # ----------------------------------------------------

        for threshold in THRESHOLDS:

            y_pred = (
                y_probability >= threshold
            ).astype(int)


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


            tn, fp, fn, tp = (
                confusion_matrix(
                    y_val,
                    y_pred,
                    labels=[0, 1]
                ).ravel()
            )


            all_results.append({

                "horizon": horizon,

                "fold": fold + 1,

                "threshold": round(
                    float(threshold),
                    2
                ),

                "precision": precision,

                "recall": recall,

                "f1": f1,

                "true_negatives": tn,

                "false_positives": fp,

                "false_negatives": fn,

                "true_positives": tp,

                "predicted_positive":
                    int(y_pred.sum()),

                "actual_positive":
                    int(y_val.sum())
            })


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_file = os.path.join(
    OUTPUT_DIR,
    "walk_forward_threshold_results.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# BEST THRESHOLD PER FOLD
# ============================================================

best_per_fold = (
    results_df
    .sort_values(
        [
            "horizon",
            "fold",
            "f1"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )
    .groupby(
        [
            "horizon",
            "fold"
        ],
        as_index=False
    )
    .first()
)


best_fold_file = os.path.join(
    OUTPUT_DIR,
    "best_threshold_per_fold.csv"
)


best_per_fold.to_csv(
    best_fold_file,
    index=False
)


# ============================================================
# SUMMARY ACROSS FOLDS
# ============================================================

threshold_summary = (
    results_df
    .groupby(
        [
            "horizon",
            "threshold"
        ]
    )
    .agg(
        mean_precision=(
            "precision",
            "mean"
        ),

        mean_recall=(
            "recall",
            "mean"
        ),

        mean_f1=(
            "f1",
            "mean"
        ),

        std_f1=(
            "f1",
            "std"
        ),

        mean_false_positives=(
            "false_positives",
            "mean"
        ),

        mean_false_negatives=(
            "false_negatives",
            "mean"
        ),

        mean_predicted_positive=(
            "predicted_positive",
            "mean"
        )
    )
    .reset_index()
)


summary_file = os.path.join(
    OUTPUT_DIR,
    "threshold_stability_summary.csv"
)


threshold_summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# BEST ROBUST THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("BEST THRESHOLD PER WALK-FORWARD FOLD")
print("=" * 70)


for horizon in HORIZONS:

    subset = best_per_fold[
        best_per_fold["horizon"]
        == horizon
    ]

    print(
        f"\n{horizon}"
    )

    print(
        subset[
            [
                "fold",
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_positives",
                "false_negatives"
            ]
        ].to_string(
            index=False
        )
    )


print("\n" + "=" * 70)
print("BEST AVERAGE-F1 THRESHOLD")
print("=" * 70)


for horizon in HORIZONS:

    subset = threshold_summary[
        threshold_summary["horizon"]
        == horizon
    ]

    best = subset.loc[
        subset["mean_f1"].idxmax()
    ]


    print(
        f"\n{horizon}"
    )

    print(
        f"Threshold              : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Mean Precision         : "
        f"{best['mean_precision']:.4f}"
    )

    print(
        f"Mean Recall            : "
        f"{best['mean_recall']:.4f}"
    )

    print(
        f"Mean F1                : "
        f"{best['mean_f1']:.4f}"
    )

    print(
        f"F1 Std                 : "
        f"{best['std_f1']:.4f}"
    )

    print(
        f"Mean False Positives   : "
        f"{best['mean_false_positives']:.2f}"
    )

    print(
        f"Mean False Negatives   : "
        f"{best['mean_false_negatives']:.2f}"
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)

print("Files saved:")

print(results_file)
print(best_fold_file)
print(summary_file)

print("\nStep 20 completed successfully.")

print("=" * 70)