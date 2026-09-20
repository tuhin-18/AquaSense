import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

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
    "final_test_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HORIZONS AND VALIDATION-SELECTED THRESHOLDS
# ============================================================

THRESHOLDS = {
    "1h": 0.85,
    "6h": 0.75,
    "12h": 0.80,
    "24h": 0.75
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 17 - FINAL XGBOOST TEST EVALUATION")
print("=" * 70)

print("\nImportant:")
print("Thresholds were selected using validation data only.")
print("The test set is being used only for final evaluation.")


results = []


# ============================================================
# EVALUATE EACH HORIZON
# ============================================================

for horizon, threshold in THRESHOLDS.items():

    print("\n" + "-" * 70)
    print(f"Evaluating {horizon} horizon")
    print("-" * 70)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test_path = os.path.join(
        PROCESSED_DIR,
        f"X_test_future_risk_{horizon}.npy"
    )

    y_test_path = os.path.join(
        PROCESSED_DIR,
        f"y_test_future_risk_{horizon}.npy"
    )

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    print(f"Test samples      : {len(y_test)}")
    print(f"Actual positives   : {int(y_test.sum())}")
    print(f"Selected threshold : {threshold:.2f}")


    # --------------------------------------------------------
    # Load trained XGBoost model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"xgboost_{horizon}.json"
    )

    model = XGBClassifier()
    model.load_model(model_path)


    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    y_probability = model.predict_proba(X_test)[:, 1]


    # --------------------------------------------------------
    # Apply validation-selected threshold
    # --------------------------------------------------------

    y_pred = (
        y_probability >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "horizon": horizon,
        "threshold": threshold,
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
        "actual_positive_events": int(y_test.sum()),
        "predicted_risk_events": int(y_pred.sum())
    })


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(f"\nAccuracy             : {accuracy:.4f}")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall               : {recall:.4f}")
    print(f"F1-score             : {f1:.4f}")
    print(f"ROC-AUC              : {roc_auc:.4f}")
    print(f"PR-AUC               : {pr_auc:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(f"True Negatives       : {tn}")
    print(f"False Positives      : {fp}")
    print(f"False Negatives      : {fn}")
    print(f"True Positives       : {tp}")

    print(f"\nActual risk events   : {int(y_test.sum())}")
    print(f"Predicted risk events: {int(y_pred.sum())}")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = os.path.join(
    OUTPUT_DIR,
    "final_xgboost_test_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

summary_columns = [
    "horizon",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc"
]

print(
    results_df[summary_columns].to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("Step 17 completed successfully.")
print("=" * 70)

print(f"\nResults saved to:")
print(output_file)