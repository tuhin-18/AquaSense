import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 29 - XGBOOST EXPLAINABILITY")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "processed_data"
)

FEATURE_LIST_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "prepared_data",
    "feature_list.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "xgboost_explainability"
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
# LOAD FEATURE NAMES
# ============================================================

print()
print("Loading feature names...")

feature_df = pd.read_csv(FEATURE_LIST_FILE)

print("Feature-list columns:")
print(feature_df.columns.tolist())

# The feature list generated in Step 11 contains the ML feature names.
# Detect the appropriate column automatically.

possible_columns = [
    "feature",
    "feature_name",
    "features",
    "Feature",
    "Feature_Name"
]

feature_column = None

for column in possible_columns:
    if column in feature_df.columns:
        feature_column = column
        break

if feature_column is None:

    if len(feature_df.columns) == 1:
        feature_column = feature_df.columns[0]
    else:
        raise ValueError(
            "Could not identify the feature-name column in feature_list.csv"
        )

feature_names = feature_df[feature_column].astype(str).tolist()

print(f"Features found: {len(feature_names)}")

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
    print("Loading model:")
    print(model_file)

    model = xgb.XGBClassifier()

    model.load_model(model_file)

    # --------------------------------------------------------
    # GET FEATURE IMPORTANCE USING GAIN
    # --------------------------------------------------------

    importance = model.get_booster().get_score(
        importance_type="gain"
    )

    # XGBoost may use feature indices such as f0, f1...
    # Convert those indices into our real feature names.

    rows = []

    for feature_key, gain in importance.items():

        if feature_key.startswith("f"):

            try:
                feature_index = int(feature_key[1:])

                if feature_index < len(feature_names):
                    real_name = feature_names[feature_index]
                else:
                    real_name = feature_key

            except ValueError:
                real_name = feature_key

        else:
            real_name = feature_key

        rows.append({
            "feature": real_name,
            "gain": float(gain),
            "horizon": horizon
        })

    importance_df = pd.DataFrame(rows)

    if len(importance_df) == 0:

        print("WARNING: No feature importance values returned.")

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
    print(f"TOP {TOP_N} FEATURES BY GAIN")
    print("-" * 70)

    print(
        importance_df[
            [
                "rank",
                "feature",
                "gain_percentage"
            ]
        ]
        .head(TOP_N)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE COMPLETE IMPORTANCE TABLE
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
    # SAVE TOP FEATURES
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
    # CREATE PLOT
    # --------------------------------------------------------

    top_plot = importance_df.head(TOP_N).copy()

    # Reverse order so the most important feature appears
    # at the top of the horizontal chart.

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

    # --------------------------------------------------------
    # KEEP FOR CROSS-HORIZON COMPARISON
    # --------------------------------------------------------

    all_importance.append(
        importance_df
    )

# ============================================================
# COMBINE IMPORTANCE RESULTS
# ============================================================

print()
print("=" * 70)
print("CROSS-HORIZON FEATURE COMPARISON")
print("=" * 70)

if len(all_importance) > 0:

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

    print()
    print("Combined file saved:")
    print(combined_file)

    # --------------------------------------------------------
    # TOP FEATURES APPEARING IN BOTH MODELS
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

    print("Cross-horizon comparison saved:")
    print(pivot_file)

    # --------------------------------------------------------
    # COMMON TOP FEATURES
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
    print("COMMON TOP FEATURES")
    print("-" * 70)

    if len(common) > 0:

        for feature in common:
            print(feature)

    else:

        print("No common features in the top 20.")

# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("STEP 29 COMPLETED")
print("=" * 70)

print()
print("Output directory:")
print(OUTPUT_DIR)