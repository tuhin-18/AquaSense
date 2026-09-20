import os
import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 28 - FINAL PREDICTIVE VS REACTIVE POLICY EVALUATION")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

POLICY_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "pump_control_policy",
    "pump_control_policy_comparison.csv"
)

INTERVENTION_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "intervention_episode_analysis",
    "intervention_episode_summary.csv"
)

EVENT_LEVEL_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "event_level_intervention_analysis",
    "event_level_summary.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "final_policy_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD POLICY RESULTS
# ============================================================

print()
print("Loading Step 27 policy results...")

policy = pd.read_csv(POLICY_FILE)

print(f"Policy rows: {len(policy)}")

# ============================================================
# LOAD EVENT-LEVEL RESULTS
# ============================================================

print()
print("Loading Step 26 event-level results...")

event_level = pd.read_csv(EVENT_LEVEL_FILE)

print(f"Event-level rows: {len(event_level)}")

print()
print("Event-level columns:")
print(event_level.columns.tolist())

# ============================================================
# LOAD INTERVENTION SUMMARY
# ============================================================

print()
print("Loading Step 25 intervention results...")

intervention = pd.read_csv(INTERVENTION_FILE)

print(f"Intervention summary rows: {len(intervention)}")

print()
print("Intervention summary columns:")
print(intervention.columns.tolist())

# ============================================================
# EXTRACT PREDICTIVE RESULTS
# ============================================================

def get_event_summary(horizon):

    row = event_level[
        event_level["horizon"].astype(str).str.lower() == horizon.lower()
    ]

    if len(row) == 0:
        return None

    return row.iloc[0]


def get_intervention_summary(horizon):

    row = intervention[
        intervention["horizon"].astype(str).str.lower() == horizon.lower()
    ]

    if len(row) == 0:
        return None

    return row.iloc[0]


# ============================================================
# BUILD FINAL RESULTS
# ============================================================

final_rows = []

# ------------------------------------------------------------
# REACTIVE
# ------------------------------------------------------------

reactive_policy = policy[
    policy["policy"].astype(str).str.lower() == "reactive"
]

if len(reactive_policy) > 0:

    reactive = reactive_policy.iloc[0]

    final_rows.append({
        "policy": "Reactive",
        "horizon": "0h",
        "threshold": reactive["threshold"],
        "test_events": reactive["test_events"],
        "detected_events": reactive["detected_events"],
        "missed_events": (
            reactive["test_events"] -
            reactive["detected_events"]
        ),
        "event_detection_rate": reactive["detection_rate"],
        "mean_first_warning_lead_hours": (
            reactive["mean_first_warning_lead_hours"]
        ),
        "median_first_warning_lead_hours": (
            reactive["median_first_warning_lead_hours"]
        ),
        "total_action_hours": reactive["total_action_hours"],
        "total_intervention_episodes": (
            reactive["total_intervention_episodes"]
            if not pd.isna(reactive["total_intervention_episodes"])
            else 45
        )
    })

# ------------------------------------------------------------
# PREDICTIVE 12H AND 24H
# ------------------------------------------------------------

for horizon in ["12h", "24h"]:

    policy_row = policy[
        (policy["policy"].astype(str).str.lower() == "predictive") &
        (policy["horizon"].astype(str).str.lower() == horizon)
    ]

    if len(policy_row) == 0:
        continue

    policy_row = policy_row.iloc[0]

    event_row = get_event_summary(horizon)

    if event_row is not None:
        detected_events = event_row["detected_events"]
        test_events = event_row["test_events"]
        mean_lead = event_row["mean_first_warning_lead_hours"]
        median_lead = event_row["median_first_warning_lead_hours"]
    else:
        detected_events = policy_row["detected_events"]
        test_events = policy_row["test_events"]
        mean_lead = policy_row["mean_first_warning_lead_hours"]
        median_lead = policy_row["median_first_warning_lead_hours"]

    final_rows.append({
        "policy": "Predictive",
        "horizon": horizon,
        "threshold": policy_row["threshold"],
        "test_events": test_events,
        "detected_events": detected_events,
        "missed_events": test_events - detected_events,
        "event_detection_rate": (
            detected_events / test_events
            if test_events > 0 else 0
        ),
        "mean_first_warning_lead_hours": mean_lead,
        "median_first_warning_lead_hours": median_lead,
        "total_action_hours": policy_row["total_action_hours"],
        "total_intervention_episodes": (
            policy_row["total_intervention_episodes"]
        )
    })

final_results = pd.DataFrame(final_rows)

# ============================================================
# ADD DERIVED METRICS
# ============================================================

reactive_hours = final_results.loc[
    final_results["policy"] == "Reactive",
    "total_action_hours"
].iloc[0]

final_results["additional_action_hours_vs_reactive"] = (
    final_results["total_action_hours"] -
    reactive_hours
)

final_results["action_hour_ratio_vs_reactive"] = (
    final_results["total_action_hours"] /
    reactive_hours
)

# ============================================================
# DISPLAY FINAL TABLE
# ============================================================

print()
print("=" * 70)
print("FINAL POLICY RESULTS")
print("=" * 70)

print(
    final_results.to_string(index=False)
)

# ============================================================
# EVENT DETECTION COMPARISON
# ============================================================

print()
print("=" * 70)
print("EVENT DETECTION")
print("=" * 70)

for _, row in final_results.iterrows():

    print(
        f"{row['policy']} {row['horizon']}: "
        f"{int(row['detected_events'])}/"
        f"{int(row['test_events'])} events detected "
        f"({row['event_detection_rate'] * 100:.2f}%)"
    )

# ============================================================
# WARNING LEAD TIME
# ============================================================

print()
print("=" * 70)
print("WARNING LEAD TIME")
print("=" * 70)

for _, row in final_results.iterrows():

    print(
        f"{row['policy']} {row['horizon']}: "
        f"mean = {row['mean_first_warning_lead_hours']:.2f} h, "
        f"median = {row['median_first_warning_lead_hours']:.2f} h"
    )

# ============================================================
# ACTION-HOUR COMPARISON
# ============================================================

print()
print("=" * 70)
print("ACTION-HOUR COMPARISON")
print("=" * 70)

for _, row in final_results.iterrows():

    print(
        f"{row['policy']} {row['horizon']}: "
        f"{int(row['total_action_hours'])} hours "
        f"({row['action_hour_ratio_vs_reactive']:.2f}x reactive)"
    )

# ============================================================
# SAVE FINAL CSV
# ============================================================

final_file = os.path.join(
    OUTPUT_DIR,
    "final_predictive_vs_reactive_results.csv"
)

final_results.to_csv(
    final_file,
    index=False
)

# ============================================================
# SAVE A PAPER-FRIENDLY TABLE
# ============================================================

paper_table = final_results[
    [
        "policy",
        "horizon",
        "threshold",
        "test_events",
        "detected_events",
        "missed_events",
        "event_detection_rate",
        "mean_first_warning_lead_hours",
        "median_first_warning_lead_hours",
        "total_action_hours",
        "total_intervention_episodes",
        "additional_action_hours_vs_reactive"
    ]
].copy()

paper_file = os.path.join(
    OUTPUT_DIR,
    "paper_comparison_table.csv"
)

paper_table.to_csv(
    paper_file,
    index=False
)

# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "final_policy_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("AquaSense IoT - Final Predictive vs Reactive Evaluation\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        "The evaluation compares a reactive turbidity-threshold "
        "policy with predictive XGBoost-based intervention policies.\n\n"
    )

    f.write(
        "High-turbidity threshold: 319.81 FNU\n"
    )

    f.write(
        "Predictive decision threshold: 0.15\n\n"
    )

    for _, row in final_results.iterrows():

        f.write(
            f"{row['policy']} {row['horizon']}\n"
        )

        f.write(
            f"  Test events: {int(row['test_events'])}\n"
        )

        f.write(
            f"  Detected events: {int(row['detected_events'])}\n"
        )

        f.write(
            f"  Detection rate: "
            f"{row['event_detection_rate'] * 100:.2f}%\n"
        )

        f.write(
            f"  Mean warning lead: "
            f"{row['mean_first_warning_lead_hours']:.2f} hours\n"
        )

        f.write(
            f"  Median warning lead: "
            f"{row['median_first_warning_lead_hours']:.2f} hours\n"
        )

        f.write(
            f"  Total action hours: "
            f"{int(row['total_action_hours'])}\n"
        )

        f.write(
            f"  Intervention episodes: "
            f"{row['total_intervention_episodes']}\n"
        )

        f.write("\n")

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(final_file)
print(paper_file)
print(summary_file)

print()
print("=" * 70)
print("STEP 28 COMPLETED")
print("=" * 70)