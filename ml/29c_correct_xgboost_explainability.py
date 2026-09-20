import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 29C - CORRECT XGBOOST EXPLAINABILITY")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

IMPUTER_FILE = os.path.join(
    MODEL_DIR,
    "median_imputer.joblib"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "xgboost_explainability_corrected"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

HORIZONS = ["12h", "24h"]
TOP_N = 20

# ============================================================
# LOAD IMPUTER FEATURE NAMES
# ============================================================

print()
print("Loading saved imputer:")

imputer = joblib.load(IMPUTER_FILE)

feature_names = imputer.get_feature_names_out()

print(f"Original features : {len(imputer.feature_names_in_)}")
print(f"Model features    : {len(feature_names)}")

if len(feature_names) != 241:
    raise ValueError(
        f"Expected 241 model features, found {len(feature_names)}"
    )

# ============================================================
# EXPLAIN EACH MODEL
# ============================================================

all_importance = []

for horizon in HORIZONS:

    print()
    print("=" * 70)
    print(f"MODEL: XGBOOST {horizon}")
    print("=" * 70)

    model_file = os.path.join(
        MODEL_DIR,
        f"xgboost_{horizon}.json"
    )

    print()
    print("Loading:")
    print(model_file)

    model = xgb.XGBClassifier()
    model.load_model(model_file)

    booster = model.get_booster()

    # --------------------------------------------------------
    # GET GAIN IMPORTANCE
    # --------------------------------------------------------

    importance = booster.get_score(
        importance_type="gain"
    )

    rows = []

    for feature_key, gain in importance.items():

        # XGBoost may return f0, f1, f2...
        if feature_key.startswith("f"):

            try:
                feature_index = int(feature_key[1:])

                if feature_index >= len(feature_names):
                    print(
                        f"WARNING: feature index {feature_index} "
                        f"is outside the 241-feature range."
                    )

                    real_name = feature_key

                else:
                    real_name = feature_names[feature_index]

            except ValueError:

                real_name = feature_key

        else:

            # Already a named feature
            real_name = feature_key

        rows.append({
            "feature_index": feature_key,
            "feature": real_name,
            "gain": float(gain),
            "horizon": horizon
        })

    importance_df = pd.DataFrame(rows)

    if importance_df.empty:

        print("No feature importance values found.")
        continue

    # --------------------------------------------------------
    # NORMALIZE GAIN
    # --------------------------------------------------------

    total_gain = importance_df["gain"].sum()

    importance_df["gain_percentage"] = (
        importance_df["gain"] /
        total_gain *
        100
    )

    importance_df = importance_df.sort_values(
        "gain_percentage",
        ascending=False
    ).reset_index(drop=True)

    importance_df["rank"] = (
        np.arange(len(importance_df)) + 1
    )

    # --------------------------------------------------------
    # DISPLAY TOP FEATURES
    # --------------------------------------------------------

    print()
    print(f"TOP {TOP_N} FEATURES")
    print("-" * 70)

    display_columns = [
        "rank",
        "feature",
        "gain_percentage"
    ]

    print(
        importance_df[
            display_columns
        ]
        .head(TOP_N)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE COMPLETE TABLE
    # --------------------------------------------------------

    complete_file = os.path.join(
        OUTPUT_DIR,
        f"xgboost_{horizon}_feature_importance.csv"
    )

    importance_df.to_csv(
        complete_file,
        index=False
    )

    # --------------------------------------------------------
    # SAVE TOP 20
    # --------------------------------------------------------

    top_file = os.path.join(
        OUTPUT_DIR,
        f"xgboost_{horizon}_top_features.csv"
    )

    importance_df.head(TOP_N).to_csv(
        top_file,
        index=False
    )

    print()
    print("Saved:")
    print(complete_file)
    print(top_file)

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    top_plot = importance_df.head(TOP_N).copy()

    top_plot = top_plot.iloc[::-1]

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_plot["feature"],
        top_plot["gain_percentage"]
    )

    plt.xlabel("Gain importance (%)")
    plt.ylabel("Feature")
    plt.title(
        f"XGBoost {horizon} - Top {TOP_N} Features"
    )

    plt.tight_layout()

    plot_file = os.path.join(
        PLOT_DIR,
        f"xgboost_{horizon}_top_features.png"
    )

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Plot saved:")
    print(plot_file)

    all_importance.append(
        importance_df
    )

# ============================================================
# CROSS-HORIZON COMPARISON
# ============================================================

print()
print("=" * 70)
print("CROSS-HORIZON COMPARISON")
print("=" * 70)

if all_importance:

    combined = pd.concat(
        all_importance,
        ignore_index=True
    )

    combined_file = os.path.join(
        OUTPUT_DIR,
        "combined_xgboost_feature_importance.csv"
    )

    combined.to_csv(
        combined_file,
        index=False
    )

    # --------------------------------------------------------
    # PIVOT 12h VS 24h
    # --------------------------------------------------------

    pivot = combined.pivot_table(
        index="feature",
        columns="horizon",
        values="gain_percentage",
        fill_value=0
    ).reset_index()

    if "12h" in pivot.columns:

        pivot["rank_12h"] = (
            pivot["12h"]
            .rank(
                ascending=False,
                method="min"
            )
        )

    if "24h" in pivot.columns:

        pivot["rank_24h"] = (
            pivot["24h"]
            .rank(
                ascending=False,
                method="min"
            )
        )

    pivot_file = os.path.join(
        OUTPUT_DIR,
        "feature_importance_12h_vs_24h.csv"
    )

    pivot.to_csv(
        pivot_file,
        index=False
    )

    print()
    print("Combined file:")
    print(combined_file)

    print()
    print("12h vs 24h comparison:")
    print(pivot_file)

    # --------------------------------------------------------
    # COMMON TOP 20
    # --------------------------------------------------------

    top12 = set(
        combined[
            combined["horizon"] == "12h"
        ]
        .sort_values(
            "gain_percentage",
            ascending=False
        )
        .head(TOP_N)["feature"]
    )

    top24 = set(
        combined[
            combined["horizon"] == "24h"
        ]
        .sort_values(
            "gain_percentage",
            ascending=False
        )
        .head(TOP_N)["feature"]
    )

    common = sorted(
        top12.intersection(top24)
    )

    print()
    print("COMMON TOP-20 FEATURES")
    print("-" * 70)

    if common:

        for feature in common:
            print(feature)

    else:

        print("No common features.")

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 29C COMPLETED")
print("=" * 70)

print()
print("Corrected output directory:")
print(OUTPUT_DIR)