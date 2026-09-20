import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 31 - FINAL RESEARCH VISUALIZATIONS")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

OUTPUT_BASE = os.path.join(
    BASE_DIR,
    "ml_outputs"
)

FIGURE_DIR = os.path.join(
    OUTPUT_BASE,
    "research_figures"
)

os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_figure(filename):

    path = os.path.join(
        FIGURE_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


def check_columns(df, required_columns, filename):

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print()
        print("ERROR: Missing columns in:")
        print(filename)

        print()
        print("Missing:")
        print(missing)

        print()
        print("Available columns:")
        print(df.columns.tolist())

        raise ValueError(
            "Required columns are missing."
        )


# ============================================================
# FIGURE 1
# ANNUAL TURBIDITY DISTRIBUTION
# ============================================================

print()
print("Creating Figure 1 - Turbidity distribution...")

yearly_file = os.path.join(
    OUTPUT_BASE,
    "temporal_shift_analysis",
    "yearly_turbidity_summary.csv"
)

yearly = pd.read_csv(
    yearly_file
)

check_columns(
    yearly,
    [
        "year",
        "mean",
        "p95",
        "p99"
    ],
    yearly_file
)

plt.figure(figsize=(11, 6))

plt.plot(
    yearly["year"],
    yearly["mean"],
    marker="o",
    label="Mean turbidity"
)

plt.plot(
    yearly["year"],
    yearly["p95"],
    marker="o",
    label="95th percentile"
)

plt.plot(
    yearly["year"],
    yearly["p99"],
    marker="o",
    label="99th percentile"
)

plt.xlabel("Year")

plt.ylabel(
    "Turbidity (FNU)"
)

plt.title(
    "Annual Turbidity Distribution Across the Monitoring Period"
)

plt.legend()

save_figure(
    "figure_1_annual_turbidity_distribution.png"
)


# ============================================================
# FIGURE 2
# EXAMPLE HIGH-TURBIDITY EVENT
# ============================================================

print()
print("Creating Figure 2 - Example high-turbidity event...")

hourly_file = os.path.join(
    OUTPUT_BASE,
    "water_quality_hourly_regularized.csv"
)

hourly = pd.read_csv(
    hourly_file,
    parse_dates=["timestamp_utc"]
)

event_start = pd.Timestamp(
    "2022-09-26 00:00:00",
    tz="UTC"
)

event_end = pd.Timestamp(
    "2022-09-28 12:00:00",
    tz="UTC"
)

event_data = hourly[
    (hourly["timestamp_utc"] >= event_start) &
    (hourly["timestamp_utc"] <= event_end)
].copy()

plt.figure(figsize=(12, 6))

plt.plot(
    event_data["timestamp_utc"],
    event_data["turbidity"],
    marker="o",
    markersize=3
)

plt.axhline(
    319.81,
    linestyle="--",
    label="High-risk threshold (99th percentile)"
)

plt.xlabel("Time")

plt.ylabel(
    "Turbidity (FNU)"
)

plt.title(
    "Example High-Turbidity Event: September 2022"
)

plt.legend()

plt.xticks(
    rotation=30
)

save_figure(
    "figure_2_example_high_turbidity_event.png"
)


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

print()
print("Loading model results...")

logistic_file = os.path.join(
    OUTPUT_BASE,
    "baseline",
    "logistic_regression_results.csv"
)

rf_file = os.path.join(
    OUTPUT_BASE,
    "random_forest",
    "random_forest_results.csv"
)

xgb_file = os.path.join(
    OUTPUT_BASE,
    "xgboost",
    "xgboost_results.csv"
)

logistic = pd.read_csv(
    logistic_file
)

random_forest = pd.read_csv(
    rf_file
)

xgboost_results = pd.read_csv(
    xgb_file
)

required_model_columns = [
    "horizon",
    "test_roc_auc",
    "test_pr_auc",
    "test_f1",
    "test_precision",
    "test_recall"
]

check_columns(
    logistic,
    required_model_columns,
    logistic_file
)

check_columns(
    random_forest,
    required_model_columns,
    rf_file
)

check_columns(
    xgboost_results,
    required_model_columns,
    xgb_file
)

models = {
    "Logistic Regression": logistic,
    "Random Forest": random_forest,
    "XGBoost": xgboost_results
}


# ============================================================
# FIGURE 3
# TEST ROC-AUC
# ============================================================

print()
print("Creating Figure 3 - Test ROC-AUC comparison...")

plt.figure(figsize=(10, 6))

for model_name, dataframe in models.items():

    plt.plot(
        dataframe["horizon"],
        dataframe["test_roc_auc"],
        marker="o",
        label=model_name
    )

plt.xlabel(
    "Prediction horizon"
)

plt.ylabel(
    "Test ROC-AUC"
)

plt.title(
    "Out-of-Sample ROC-AUC Across Prediction Horizons"
)

plt.legend()

plt.ylim(
    0.5,
    1.0
)

save_figure(
    "figure_3_test_roc_auc_comparison.png"
)


# ============================================================
# FIGURE 4
# TEST PR-AUC
# ============================================================

print()
print("Creating Figure 4 - Test PR-AUC comparison...")

plt.figure(figsize=(10, 6))

for model_name, dataframe in models.items():

    plt.plot(
        dataframe["horizon"],
        dataframe["test_pr_auc"],
        marker="o",
        label=model_name
    )

plt.xlabel(
    "Prediction horizon"
)

plt.ylabel(
    "Test PR-AUC"
)

plt.title(
    "Out-of-Sample Precision-Recall AUC Across Prediction Horizons"
)

plt.legend()

save_figure(
    "figure_4_test_pr_auc_comparison.png"
)


# ============================================================
# FIGURE 5
# TEST F1 SCORE
# ============================================================

print()
print("Creating Figure 5 - Test F1 comparison...")

plt.figure(figsize=(10, 6))

for model_name, dataframe in models.items():

    plt.plot(
        dataframe["horizon"],
        dataframe["test_f1"],
        marker="o",
        label=model_name
    )

plt.xlabel(
    "Prediction horizon"
)

plt.ylabel(
    "Test F1-score"
)

plt.title(
    "Out-of-Sample F1-score Across Prediction Horizons"
)

plt.legend()

plt.ylim(
    0,
    1
)

save_figure(
    "figure_5_test_f1_comparison.png"
)


# ============================================================
# FIGURE 6
# WALK-FORWARD ROC-AUC
# ============================================================

print()
print("Creating Figure 6 - Walk-forward ROC-AUC...")

walk_file = os.path.join(
    OUTPUT_BASE,
    "walk_forward_validation",
    "walk_forward_fold_results.csv"
)

walk = pd.read_csv(
    walk_file
)

check_columns(
    walk,
    [
        "horizon",
        "fold",
        "roc_auc"
    ],
    walk_file
)

plt.figure(figsize=(10, 6))

for horizon in [
    "1h",
    "6h",
    "12h",
    "24h"
]:

    subset = walk[
        walk["horizon"] == horizon
    ]

    plt.plot(
        subset["fold"],
        subset["roc_auc"],
        marker="o",
        label=horizon
    )

plt.xlabel(
    "Walk-forward fold"
)

plt.ylabel(
    "ROC-AUC"
)

plt.title(
    "Walk-Forward ROC-AUC Across Temporal Folds"
)

plt.legend()

save_figure(
    "figure_6_walk_forward_roc_auc.png"
)


# ============================================================
# FIGURE 7
# EVENT WARNING DETECTION
# ============================================================

print()
print("Creating Figure 7 - Event warning detection...")

warning_file = os.path.join(
    OUTPUT_BASE,
    "strict_event_warning",
    "strict_event_warning_summary.csv"
)

warning = pd.read_csv(
    warning_file
)

print()
print("Warning file columns:")
print(warning.columns.tolist())

# ------------------------------------------------------------
# Verify required columns
# ------------------------------------------------------------

check_columns(
    warning,
    [
        "horizon",
        "warning_window_hours",
        "test_events",
        "detected_events",
        "detection_rate"
    ],
    warning_file
)

# ------------------------------------------------------------
# Keep 24-hour and 48-hour warning windows
# ------------------------------------------------------------

warning_plot = warning[
    warning["warning_window_hours"].isin(
        [24, 48]
    )
].copy()

# Convert detection rate to percentage.
#
# The CSV already contains detection_rate.
# Example:
# 0.9167 -> 91.67%
#
# We calculate it again from detected_events/test_events
# to make the figure completely transparent.

warning_plot["detection_percentage"] = (
    warning_plot["detected_events"]
    /
    warning_plot["test_events"]
    *
    100
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for horizon in warning_plot["horizon"].unique():

    subset = warning_plot[
        warning_plot["horizon"] == horizon
    ]

    plt.plot(
        subset["warning_window_hours"],
        subset["detection_percentage"],
        marker="o",
        label=horizon
    )

plt.xlabel(
    "Warning window (hours)"
)

plt.ylabel(
    "Events detected (%)"
)

plt.title(
    "Out-of-Sample High-Turbidity Event Detection"
)

plt.xticks(
    [24, 48]
)

plt.ylim(
    0,
    110
)

plt.legend()

save_figure(
    "figure_7_event_warning_detection.png"
)

# ============================================================
# FIGURE 8
# FIRST-WARNING LEAD TIME
# ============================================================

print()
print("Creating Figure 8 - Warning lead time...")

event_policy_file = os.path.join(
    OUTPUT_BASE,
    "event_level_intervention_analysis",
    "event_level_results.csv"
)

event_policy = pd.read_csv(
    event_policy_file
)

print()
print("Event-level file columns:")
print(event_policy.columns.tolist())

# ------------------------------------------------------------
# Detect correct lead-time column
# ------------------------------------------------------------

if "first_warning_lead_hours" in event_policy.columns:

    lead_column = "first_warning_lead_hours"

elif "mean_first_warning_lead_hours" in event_policy.columns:

    lead_column = "mean_first_warning_lead_hours"

else:

    raise ValueError(
        "Could not find a first-warning lead-time column."
    )


labels = []
mean_leads = []
median_leads = []

for horizon in [
    "12h",
    "24h"
]:

    subset = event_policy[
        event_policy["horizon"] == horizon
    ]

    if len(subset) == 0:
        continue

    labels.append(
        horizon
    )

    if "first_warning_lead_hours" in subset.columns:

        mean_leads.append(
            subset["first_warning_lead_hours"].mean()
        )

        median_leads.append(
            subset["first_warning_lead_hours"].median()
        )

    else:

        # Summary-style file
        mean_leads.append(
            subset[lead_column].mean()
        )

        if "median_first_warning_lead_hours" in subset.columns:

            median_leads.append(
                subset[
                    "median_first_warning_lead_hours"
                ].mean()
            )

        else:

            median_leads.append(
                subset[lead_column].median()
            )


plt.figure(figsize=(9, 6))

x = np.arange(
    len(labels)
)

width = 0.35

plt.bar(
    x - width / 2,
    mean_leads,
    width,
    label="Mean"
)

plt.bar(
    x + width / 2,
    median_leads,
    width,
    label="Median"
)

plt.xticks(
    x,
    labels
)

plt.xlabel(
    "Prediction horizon"
)

plt.ylabel(
    "First-warning lead time (hours)"
)

plt.title(
    "First-Warning Lead Time for Detected Events"
)

plt.legend()

save_figure(
    "figure_8_warning_lead_time.png"
)


# ============================================================
# FIGURE 9
# REACTIVE VS PREDICTIVE ACTION HOURS
# ============================================================

print()
print("Creating Figure 9 - Pump action comparison...")

policy_file = os.path.join(
    OUTPUT_BASE,
    "final_policy_evaluation",
    "final_predictive_vs_reactive_results.csv"
)

policy = pd.read_csv(
    policy_file
)

print()
print("Policy file columns:")
print(policy.columns.tolist())

# ------------------------------------------------------------
# Verify required columns
# ------------------------------------------------------------

check_columns(
    policy,
    [
        "policy",
        "total_action_hours"
    ],
    policy_file
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    policy["policy"],
    policy["total_action_hours"]
)

plt.ylabel(
    "Pump-action hours"
)

plt.xlabel(
    "Control strategy"
)

plt.title(
    "Simulated Pump-Action Hours: Reactive vs Predictive Control"
)

plt.xticks(
    rotation=20
)

save_figure(
    "figure_9_pump_action_comparison.png"
)
# ============================================================
# FIGURE 10
# XGBOOST FEATURE IMPORTANCE
# ============================================================

print()
print("Creating Figure 10 - XGBoost feature importance...")

importance_dir = os.path.join(
    OUTPUT_BASE,
    "xgboost_explainability_corrected"
)

importance_file = os.path.join(
    importance_dir,
    "feature_importance_12h_vs_24h.csv"
)

importance = pd.read_csv(
    importance_file
)

print()
print("Feature importance columns:")
print(importance.columns.tolist())

# ------------------------------------------------------------
# 12-HOUR FEATURE IMPORTANCE
# ------------------------------------------------------------

if (
    "feature" in importance.columns
    and
    "gain_percentage_12h" in importance.columns
):

    importance_12h = (
        importance[
            [
                "feature",
                "gain_percentage_12h"
            ]
        ]
        .dropna()
        .sort_values(
            "gain_percentage_12h",
            ascending=False
        )
        .head(10)
    )

    importance_12h = (
        importance_12h
        .iloc[::-1]
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        importance_12h["feature"],
        importance_12h["gain_percentage_12h"]
    )

    plt.xlabel(
        "Gain importance (%)"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "XGBoost 12-Hour Model - Top 10 Features"
    )

    save_figure(
        "figure_10_xgboost_12h_feature_importance.png"
    )


# ------------------------------------------------------------
# 24-HOUR FEATURE IMPORTANCE
# ------------------------------------------------------------

if (
    "feature" in importance.columns
    and
    "gain_percentage_24h" in importance.columns
):

    importance_24h = (
        importance[
            [
                "feature",
                "gain_percentage_24h"
            ]
        ]
        .dropna()
        .sort_values(
            "gain_percentage_24h",
            ascending=False
        )
        .head(10)
    )

    importance_24h = (
        importance_24h
        .iloc[::-1]
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        importance_24h["feature"],
        importance_24h["gain_percentage_24h"]
    )

    plt.xlabel(
        "Gain importance (%)"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "XGBoost 24-Hour Model - Top 10 Features"
    )

    save_figure(
        "figure_10_xgboost_24h_feature_importance.png"
    )


# ============================================================
# FIGURE 11
# MISSINGNESS SENSITIVITY
# ============================================================

print()
print("Creating Figure 11 - Missingness sensitivity...")

sensitivity_file = os.path.join(
    OUTPUT_BASE,
    "missingness_sensitivity",
    "missingness_model_comparison.csv"
)

sensitivity = pd.read_csv(
    sensitivity_file
)

print()
print("Missingness file columns:")
print(sensitivity.columns.tolist())

# ------------------------------------------------------------
# Detect columns
# ------------------------------------------------------------

required_sensitivity = [
    "horizon",
    "pr_auc_with_indicators",
    "pr_auc_without_indicators"
]

check_columns(
    sensitivity,
    required_sensitivity,
    sensitivity_file
)

plt.figure(figsize=(9, 6))

x = np.arange(
    len(sensitivity["horizon"])
)

width = 0.35

plt.bar(
    x - width / 2,
    sensitivity["pr_auc_with_indicators"],
    width,
    label="With missingness indicators"
)

plt.bar(
    x + width / 2,
    sensitivity["pr_auc_without_indicators"],
    width,
    label="Without missingness indicators"
)

plt.xticks(
    x,
    sensitivity["horizon"]
)

plt.xlabel(
    "Prediction horizon"
)

plt.ylabel(
    "PR-AUC"
)

plt.title(
    "Missingness Sensitivity: Precision-Recall AUC"
)

plt.legend()

save_figure(
    "figure_11_missingness_sensitivity.png"
)


# ============================================================
# FIGURE INDEX
# ============================================================

print()
print("Creating figure index...")

figure_index = pd.DataFrame({

    "figure": [
        "Figure 1",
        "Figure 2",
        "Figure 3",
        "Figure 4",
        "Figure 5",
        "Figure 6",
        "Figure 7",
        "Figure 8",
        "Figure 9",
        "Figure 10 - 12h",
        "Figure 10 - 24h",
        "Figure 11"
    ],

    "filename": [
        "figure_1_annual_turbidity_distribution.png",
        "figure_2_example_high_turbidity_event.png",
        "figure_3_test_roc_auc_comparison.png",
        "figure_4_test_pr_auc_comparison.png",
        "figure_5_test_f1_comparison.png",
        "figure_6_walk_forward_roc_auc.png",
        "figure_7_event_warning_detection.png",
        "figure_8_warning_lead_time.png",
        "figure_9_pump_action_comparison.png",
        "figure_10_xgboost_12h_feature_importance.png",
        "figure_10_xgboost_24h_feature_importance.png",
        "figure_11_missingness_sensitivity.png"
    ]
})

index_file = os.path.join(
    FIGURE_DIR,
    "research_figure_index.csv"
)

figure_index.to_csv(
    index_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 31 COMPLETED")
print("=" * 70)

print()
print("Figures saved to:")

print(FIGURE_DIR)

print()
print("Figure index:")

print(index_file)

print()
print("Total planned figures:")

print(len(figure_index))