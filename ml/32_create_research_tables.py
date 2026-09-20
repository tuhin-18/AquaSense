import os
import pandas as pd

# ============================================================
# STEP 32 - FINAL RESEARCH RESULTS TABLES
# ============================================================

print("=" * 70)
print("STEP 32 - FINAL RESEARCH RESULTS TABLES")
print("=" * 70)

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"
OUTPUT_BASE = os.path.join(BASE_DIR, "ml_outputs")

TABLE_DIR = os.path.join(
    OUTPUT_BASE,
    "research_tables"
)

os.makedirs(TABLE_DIR, exist_ok=True)


def save_table(df, filename):
    path = os.path.join(TABLE_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def check_columns(df, required, filename):
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"\nMissing columns in {filename}: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


# ============================================================
# TABLE 1
# DATASET SUMMARY
# ============================================================

print()
print("Creating Table 1 - Dataset summary...")

yearly_file = os.path.join(
    OUTPUT_BASE,
    "temporal_shift_analysis",
    "yearly_turbidity_summary.csv"
)

yearly = pd.read_csv(yearly_file)

check_columns(
    yearly,
    [
        "year",
        "observations",
        "mean",
        "median",
        "p95",
        "p99",
        "maximum",
        "high_risk_count",
        "high_risk_rate_percent"
    ],
    yearly_file
)

table1 = pd.DataFrame({
    "Metric": [
        "Monitoring years",
        "Number of yearly records",
        "Total observations",
        "Mean turbidity (FNU)",
        "Median turbidity (FNU)",
        "99th percentile turbidity (FNU)",
        "Maximum turbidity (FNU)",
        "Total high-risk observations",
    ],
    "Value": [
        f"{int(yearly['year'].min())}-{int(yearly['year'].max())}",
        len(yearly),
        int(yearly["observations"].sum()),
        round(
            yearly["mean"].mean(),
            2
        ),
        round(
            yearly["median"].mean(),
            2
        ),
        round(
            yearly["p99"].mean(),
            2
        ),
        round(
            yearly["maximum"].max(),
            2
        ),
        int(
            yearly["high_risk_count"].sum()
        )
    ]
})

save_table(
    table1,
    "table_1_dataset_summary.csv"
)


# ============================================================
# TABLE 2
# MODEL PERFORMANCE
# ============================================================

print()
print("Creating Table 2 - Model performance...")

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

logistic = pd.read_csv(logistic_file)
rf = pd.read_csv(rf_file)
xgb = pd.read_csv(xgb_file)

required_model_columns = [
    "horizon",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "test_pr_auc"
]

check_columns(
    logistic,
    required_model_columns,
    logistic_file
)

check_columns(
    rf,
    required_model_columns,
    rf_file
)

check_columns(
    xgb,
    required_model_columns,
    xgb_file
)


def make_model_table(df, model_name):
    result = df[
        [
            "horizon",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc"
        ]
    ].copy()

    result.insert(
        1,
        "model",
        model_name
    )

    return result


table2 = pd.concat(
    [
        make_model_table(logistic, "Logistic Regression"),
        make_model_table(rf, "Random Forest"),
        make_model_table(xgb, "XGBoost")
    ],
    ignore_index=True
)

table2 = table2.rename(
    columns={
        "test_precision": "precision",
        "test_recall": "recall",
        "test_f1": "f1",
        "test_roc_auc": "roc_auc",
        "test_pr_auc": "pr_auc"
    }
)

numeric_columns = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc"
]

table2[numeric_columns] = table2[numeric_columns].round(4)

save_table(
    table2,
    "table_2_model_performance.csv"
)


# ============================================================
# TABLE 3
# WALK-FORWARD VALIDATION
# ============================================================

print()
print("Creating Table 3 - Walk-forward validation...")

walk_file = os.path.join(
    OUTPUT_BASE,
    "walk_forward_validation",
    "walk_forward_summary.csv"
)

if os.path.exists(walk_file):

    walk = pd.read_csv(walk_file)

    print("Walk-forward columns:")
    print(walk.columns.tolist())

    save_table(
        walk,
        "table_3_walk_forward_validation.csv"
    )

else:

    # Fall back to fold-level results
    walk_file = os.path.join(
        OUTPUT_BASE,
        "walk_forward_validation",
        "walk_forward_fold_results.csv"
    )

    walk = pd.read_csv(walk_file)

    print("Walk-forward file columns:")
    print(walk.columns.tolist())

    save_table(
        walk,
        "table_3_walk_forward_validation.csv"
    )


# ============================================================
# TABLE 4
# EVENT WARNING PERFORMANCE
# ============================================================

print()
print("Creating Table 4 - Event warning performance...")

warning_file = os.path.join(
    OUTPUT_BASE,
    "strict_event_warning",
    "strict_event_warning_summary.csv"
)

warning = pd.read_csv(warning_file)

print("Warning columns:")
print(warning.columns.tolist())

check_columns(
    warning,
    [
        "horizon",
        "threshold",
        "warning_window_hours",
        "test_events",
        "detected_events",
        "missed_events",
        "detection_rate",
        "mean_lead_time_hours",
        "median_lead_time_hours",
        "minimum_lead_time_hours",
        "maximum_lead_time_hours",
        "total_test_warnings"
    ],
    warning_file
)

table4 = warning[
    [
        "horizon",
        "threshold",
        "warning_window_hours",
        "test_events",
        "detected_events",
        "missed_events",
        "detection_rate",
        "mean_lead_time_hours",
        "median_lead_time_hours",
        "minimum_lead_time_hours",
        "maximum_lead_time_hours",
        "total_test_warnings"
    ]
].copy()

table4 = table4.rename(
    columns={
        "detection_rate": "detection_rate_percent"
    }
)

table4["detection_rate_percent"] = (
    table4["detection_rate_percent"] * 100
).round(2)

table4 = table4.round(2)

save_table(
    table4,
    "table_4_event_warning_performance.csv"
)


# ============================================================
# TABLE 5
# PREDICTIVE VS REACTIVE PUMP POLICY
# ============================================================

print()
print("Creating Table 5 - Pump policy comparison...")

policy_file = os.path.join(
    OUTPUT_BASE,
    "final_policy_evaluation",
    "final_predictive_vs_reactive_results.csv"
)

policy = pd.read_csv(policy_file)

print("Policy columns:")
print(policy.columns.tolist())

check_columns(
    policy,
    [
        "policy",
        "horizon",
        "test_events",
        "detected_events",
        "missed_events",
        "event_detection_rate",
        "mean_first_warning_lead_hours",
        "median_first_warning_lead_hours",
        "total_action_hours",
        "total_intervention_episodes",
        "additional_action_hours_vs_reactive",
        "action_hour_ratio_vs_reactive"
    ],
    policy_file
)

table5 = policy[
    [
        "policy",
        "horizon",
        "test_events",
        "detected_events",
        "missed_events",
        "event_detection_rate",
        "mean_first_warning_lead_hours",
        "median_first_warning_lead_hours",
        "total_action_hours",
        "total_intervention_episodes",
        "additional_action_hours_vs_reactive",
        "action_hour_ratio_vs_reactive"
    ]
].copy()

table5 = table5.rename(
    columns={
        "event_detection_rate": "event_detection_rate_percent"
    }
)

table5["event_detection_rate_percent"] = (
    table5["event_detection_rate_percent"] * 100
).round(2)

table5 = table5.round(2)

save_table(
    table5,
    "table_5_pump_policy_comparison.csv"
)


# ============================================================
# TABLE 6
# XGBOOST FEATURE IMPORTANCE
# ============================================================

print()
print("Creating Table 6 - XGBoost feature importance...")

importance_file = os.path.join(
    OUTPUT_BASE,
    "xgboost_explainability_corrected",
    "feature_importance_12h_vs_24h.csv"
)

importance = pd.read_csv(importance_file)

print("Feature importance columns:")
print(importance.columns.tolist())

check_columns(
    importance,
    [
        "feature",
        "12h",
        "24h",
        "rank_12h",
        "rank_24h"
    ],
    importance_file
)

table6 = importance[
    [
        "feature",
        "12h",
        "24h",
        "rank_12h",
        "rank_24h"
    ]
].copy()

table6 = table6.sort_values(
    "rank_12h"
)

table6["12h"] = table6["12h"].round(4)
table6["24h"] = table6["24h"].round(4)

save_table(
    table6,
    "table_6_feature_importance.csv"
)


# ============================================================
# TABLE 7
# MISSINGNESS SENSITIVITY
# ============================================================

print()
print("Creating Table 7 - Missingness sensitivity...")

missing_file = os.path.join(
    OUTPUT_BASE,
    "missingness_sensitivity",
    "missingness_model_comparison.csv"
)

missing = pd.read_csv(missing_file)

print("Missingness columns:")
print(missing.columns.tolist())

check_columns(
    missing,
    [
        "horizon",
        "f1_with_indicators",
        "f1_without_indicators",
        "f1_difference",
        "roc_auc_with_indicators",
        "roc_auc_without_indicators",
        "roc_auc_difference",
        "pr_auc_with_indicators",
        "pr_auc_without_indicators",
        "pr_auc_difference"
    ],
    missing_file
)

table7 = missing.copy()

table7 = table7.round(4)

save_table(
    table7,
    "table_7_missingness_sensitivity.csv"
)


# ============================================================
# TABLE INDEX
# ============================================================

print()
print("Creating research table index...")

index = pd.DataFrame({
    "table_number": [
        1,
        2,
        3,
        4,
        5,
        6,
        7
    ],
    "filename": [
        "table_1_dataset_summary.csv",
        "table_2_model_performance.csv",
        "table_3_walk_forward_validation.csv",
        "table_4_event_warning_performance.csv",
        "table_5_pump_policy_comparison.csv",
        "table_6_feature_importance.csv",
        "table_7_missingness_sensitivity.csv"
    ],
    "description": [
        "Dataset and turbidity summary",
        "Chronological test-set model performance",
        "Walk-forward validation results",
        "Strict out-of-sample event warning performance",
        "Reactive versus predictive pump-control simulation",
        "XGBoost feature importance for 12h and 24h horizons",
        "Sensitivity analysis for missingness indicators"
    ]
})

save_table(
    index,
    "results_table_index.csv"
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("STEP 32 COMPLETED")
print("=" * 70)

print()
print("Research tables saved to:")
print(TABLE_DIR)

print()
print("Tables created:")
for filename in index["filename"]:
    print(" -", filename)

print()
print("Table index:")
print(
    os.path.join(
        TABLE_DIR,
        "results_table_index.csv"
    )
)