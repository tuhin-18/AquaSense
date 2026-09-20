import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

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
    "random_forest"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

HORIZONS = [
    "1h",
    "6h",
    "12h",
    "24h"
]


# ============================================================
# START
# ============================================================

print("=" * 60)
print("STEP 14 - RANDOM FOREST")
print("=" * 60)

results = []


# ============================================================
# TRAIN ONE RANDOM FOREST FOR EACH HORIZON
# ============================================================

for horizon in HORIZONS:

    target_name = f"future_risk_{horizon}"

    print("\n" + "=" * 60)
    print(f"TARGET: {target_name}")
    print("=" * 60)


    # --------------------------------------------------------
    # Load horizon-specific feature matrices
    # --------------------------------------------------------

    print("\nLoading processed data...")

    X_train = np.load(
        os.path.join(
            PROCESSED_DIR,
            f"X_train_future_risk_{horizon}.npy"
        )
    )

    X_validation = np.load(
        os.path.join(
            PROCESSED_DIR,
            f"X_validation_future_risk_{horizon}.npy"
        )
    )

    X_test = np.load(
        os.path.join(
            PROCESSED_DIR,
            f"X_test_future_risk_{horizon}.npy"
        )
    )


    # --------------------------------------------------------
    # Load matching targets
    # --------------------------------------------------------

    y_train = np.load(
        os.path.join(
            PROCESSED_DIR,
            f"y_train_future_risk_{horizon}.npy"
        )
    )

    y_validation = np.load(
        os.path.join(
            PROCESSED_DIR,
            f"y_validation_future_risk_{horizon}.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            PROCESSED_DIR,
            f"y_test_future_risk_{horizon}.npy"
        )
    )


    # --------------------------------------------------------
    # Display dimensions
    # --------------------------------------------------------

    print(
        f"Training features   : {X_train.shape}"
    )

    print(
        f"Validation features : {X_validation.shape}"
    )

    print(
        f"Test features       : {X_test.shape}"
    )

    print(
        f"Training target     : {y_train.shape}"
    )

    print(
        f"Validation target   : {y_validation.shape}"
    )

    print(
        f"Test target         : {y_test.shape}"
    )


    # --------------------------------------------------------
    # Verify alignment
    # --------------------------------------------------------

    assert X_train.shape[0] == y_train.shape[0]
    assert X_validation.shape[0] == y_validation.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    print("\nX/y alignment check: PASSED")


    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nTarget distribution:")

    print(
        f"Training positives   : "
        f"{np.sum(y_train == 1):,}"
    )

    print(
        f"Training negatives   : "
        f"{np.sum(y_train == 0):,}"
    )

    print(
        f"Validation positives : "
        f"{np.sum(y_validation == 1):,}"
    )

    print(
        f"Test positives       : "
        f"{np.sum(y_test == 1):,}"
    )


    # ========================================================
    # RANDOM FOREST MODEL
    # ========================================================

    print("\nCreating Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print("Training Random Forest...")
    print("This may take several minutes...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")


    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    print("\nGenerating validation predictions...")

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    validation_prediction = (
        validation_probability >= 0.5
    ).astype(int)


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    validation_accuracy = accuracy_score(
        y_validation,
        validation_prediction
    )

    validation_precision = precision_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )

    validation_recall = recall_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )

    validation_f1 = f1_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )

    validation_roc_auc = roc_auc_score(
        y_validation,
        validation_probability
    )

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_probability
    )

    validation_cm = confusion_matrix(
        y_validation,
        validation_prediction
    )


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    print("Generating test predictions...")

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    test_prediction = (
        test_probability >= 0.5
    ).astype(int)


    # ========================================================
    # TEST METRICS
    # ========================================================

    test_accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    test_precision = precision_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probability
    )

    test_pr_auc = average_precision_score(
        y_test,
        test_probability
    )

    test_cm = confusion_matrix(
        y_test,
        test_prediction
    )


    # ========================================================
    # DISPLAY VALIDATION RESULTS
    # ========================================================

    print("\n" + "-" * 60)
    print("VALIDATION RESULTS")
    print("-" * 60)

    print(
        f"Accuracy  : {validation_accuracy:.4f}"
    )

    print(
        f"Precision : {validation_precision:.4f}"
    )

    print(
        f"Recall    : {validation_recall:.4f}"
    )

    print(
        f"F1 Score  : {validation_f1:.4f}"
    )

    print(
        f"ROC-AUC   : {validation_roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {validation_pr_auc:.4f}"
    )

    print("\nValidation Confusion Matrix:")

    print(validation_cm)


    # ========================================================
    # DISPLAY TEST RESULTS
    # ========================================================

    print("\n" + "-" * 60)
    print("TEST RESULTS")
    print("-" * 60)

    print(
        f"Accuracy  : {test_accuracy:.4f}"
    )

    print(
        f"Precision : {test_precision:.4f}"
    )

    print(
        f"Recall    : {test_recall:.4f}"
    )

    print(
        f"F1 Score  : {test_f1:.4f}"
    )

    print(
        f"ROC-AUC   : {test_roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {test_pr_auc:.4f}"
    )

    print("\nTest Confusion Matrix:")

    print(test_cm)


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "horizon": horizon,

        "validation_accuracy":
            validation_accuracy,

        "validation_precision":
            validation_precision,

        "validation_recall":
            validation_recall,

        "validation_f1":
            validation_f1,

        "validation_roc_auc":
            validation_roc_auc,

        "validation_pr_auc":
            validation_pr_auc,

        "test_accuracy":
            test_accuracy,

        "test_precision":
            test_precision,

        "test_recall":
            test_recall,

        "test_f1":
            test_f1,

        "test_roc_auc":
            test_roc_auc,

        "test_pr_auc":
            test_pr_auc
    })


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_file = os.path.join(
        MODEL_DIR,
        f"random_forest_{horizon}.joblib"
    )

    import joblib

    joblib.dump(
        model,
        model_file
    )

    print(
        f"\nModel saved to:\n{model_file}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = os.path.join(
    OUTPUT_DIR,
    "random_forest_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST TRAINING COMPLETED")
print("=" * 60)

print("\nRandom Forest Results:")

print(
    results_df.to_string(index=False)
)

print("\nResults saved to:")

print(output_file)
