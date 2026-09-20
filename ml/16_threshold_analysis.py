import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from joblib import load
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "processed_data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "threshold_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

HORIZONS = ["1h", "6h", "12h", "24h"]

THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.05
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 16 - DECISION THRESHOLD ANALYSIS")
print("=" * 70)

print("\nLoading validation and test data...")


results = []


# ============================================================
# PROCESS EACH HORIZON
# ============================================================

for horizon in HORIZONS:

    print("\n" + "-" * 70)
    print(f"Analyzing {horizon} horizon")
    print("-" * 70)

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    X_val_path = os.path.join(
        PROCESSED_DIR,
        f"X_validation_future_risk_{horizon}.npy"
    )

    y_val_path = os.path.join(
        PROCESSED_DIR,
        f"y_validation_future_risk_{horizon}.npy"
    )

    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)

    print(f"Validation samples: {len(y_val)}")
    print(f"Validation positives: {int(y_val.sum())}")


    # --------------------------------------------------------
    # Load XGBoost model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"xgboost_{horizon}.json"
    )

    model = XGBClassifier()
    model.load_model(model_path)


    # --------------------------------------------------------
    # Generate probability predictions
    # --------------------------------------------------------

    y_probability = model.predict_proba(X_val)[:, 1]


    print(
        f"Probability range: "
        f"{y_probability.min():.4f} - "
        f"{y_probability.max():.4f}"
    )


    # --------------------------------------------------------
    # Test thresholds
    # --------------------------------------------------------

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


        tn, fp, fn, tp = confusion_matrix(
            y_val,
            y_pred,
            labels=[0, 1]
        ).ravel()


        results.append({
            "horizon": horizon,
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "predicted_risk_events": int(y_pred.sum())
        })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = os.path.join(
    OUTPUT_DIR,
    "threshold_analysis_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY BEST F1 OPERATING POINT
# ============================================================

print("\n" + "=" * 70)
print("BEST VALIDATION F1 OPERATING POINTS")
print("=" * 70)

for horizon in HORIZONS:

    horizon_df = results_df[
        results_df["horizon"] == horizon
    ]

    best_row = horizon_df.loc[
        horizon_df["f1"].idxmax()
    ]

    print(f"\n{horizon}")

    print(
        f"Threshold              : "
        f"{best_row['threshold']:.2f}"
    )

    print(
        f"Precision              : "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall                 : "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1-score               : "
        f"{best_row['f1']:.4f}"
    )

    print(
        f"False positives        : "
        f"{int(best_row['false_positives'])}"
    )

    print(
        f"False negatives        : "
        f"{int(best_row['false_negatives'])}"
    )

    print(
        f"Predicted risk events  : "
        f"{int(best_row['predicted_risk_events'])}"
    )


print("\n" + "=" * 70)

print(
    "Threshold analysis completed successfully."
)

print(
    f"Results saved to:\n{output_file}"
)

print("=" * 70)