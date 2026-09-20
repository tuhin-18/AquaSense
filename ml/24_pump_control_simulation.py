import os
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# STEP 24 - REACTIVE VS PREDICTIVE PUMP CONTROL SIMULATION
# ============================================================

print("=" * 70)
print("STEP 24 - REACTIVE VS PREDICTIVE PUMP CONTROL SIMULATION")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

PREPARED_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "prepared_data"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "processed_data"
)

EVENT_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "event_level_analysis",
    "high_turbidity_events.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "pump_control_simulation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

TURBIDITY_THRESHOLD = 319.81

# Main operational warning window
WARNING_WINDOW_HOURS = 48

# Models evaluated
HORIZONS = [
    "12h",
    "24h"
]

# Fixed thresholds from Step 20
THRESHOLDS = {
    "12h": 0.15,
    "24h": 0.15
}


# ------------------------------------------------------------
# LOAD HISTORICAL EVENTS
# ------------------------------------------------------------

print("\nLoading high-turbidity events...")

events = pd.read_csv(
    EVENT_FILE
)

events["start_time"] = pd.to_datetime(
    events["start_time"],
    utc=True
)

events["end_time"] = pd.to_datetime(
    events["end_time"],
    utc=True
)

print(
    "Total historical events:",
    len(events)
)


# ------------------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------------------

print("\nLoading prepared test data...")

test_df = pd.read_csv(
    os.path.join(
        PREPARED_DIR,
        "test.csv"
    )
)

test_df["timestamp_utc"] = pd.to_datetime(
    test_df["timestamp_utc"],
    utc=True
)

test_df = (
    test_df
    .sort_values("timestamp_utc")
    .reset_index(drop=True)
)

test_start = test_df[
    "timestamp_utc"
].min()

test_end = test_df[
    "timestamp_utc"
].max()

print(
    "Test period:",
    test_start,
    "to",
    test_end
)


# ------------------------------------------------------------
# RESTRICT EVENTS TO TEST PERIOD
# ------------------------------------------------------------

test_events = events[
    (
        events["start_time"] >= test_start
    )
    &
    (
        events["start_time"] <= test_end
    )
].copy()

test_events = (
    test_events
    .sort_values("start_time")
    .reset_index(drop=True)
)

print(
    "Test-period events:",
    len(test_events)
)


# ============================================================
# PART 1
# REACTIVE CONTROL SIMULATION
# ============================================================

print("\n" + "=" * 70)
print("PART 1 - REACTIVE CONTROL")
print("=" * 70)


# ------------------------------------------------------------
# LOAD HOURLY REGULARIZED DATA
# ------------------------------------------------------------

hourly_file = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "water_quality_hourly_regularized.csv"
)

hourly_df = pd.read_csv(
    hourly_file
)

hourly_df["timestamp_utc"] = pd.to_datetime(
    hourly_df["timestamp_utc"],
    utc=True
)

hourly_df = (
    hourly_df
    .sort_values("timestamp_utc")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# RESTRICT TO TEST PERIOD
# ------------------------------------------------------------

test_hourly = hourly_df[
    (
        hourly_df["timestamp_utc"] >= test_start
    )
    &
    (
        hourly_df["timestamp_utc"] <= test_end
    )
].copy()


# ------------------------------------------------------------
# REACTIVE PUMP ACTION
# ------------------------------------------------------------

test_hourly["reactive_action"] = (
    test_hourly["turbidity"]
    >= TURBIDITY_THRESHOLD
).astype(int)


print(
    "Reactive pump actions:",
    int(
        test_hourly["reactive_action"].sum()
    )
)


# ------------------------------------------------------------
# REACTIVE EVENT ANALYSIS
# ------------------------------------------------------------

reactive_records = []


for _, event in test_events.iterrows():

    event_id = int(
        event["event_id"]
    )

    event_start = event["start_time"]

    event_end = event["end_time"]

    event_segment = event["segment_id"]


    # First actual threshold crossing
    # within the event itself.

    event_data = test_hourly[
        (
            test_hourly["segment_id"]
            == event_segment
        )
        &
        (
            test_hourly["timestamp_utc"]
            >= event_start
        )
        &
        (
            test_hourly["timestamp_utc"]
            <= event_end
        )
        &
        (
            test_hourly["turbidity"]
            >= TURBIDITY_THRESHOLD
        )
    ].copy()


    if len(event_data) > 0:

        reactive_time = (
            event_data[
                "timestamp_utc"
            ].iloc[0]
        )

        reactive_lead = (
            (
                event_start
                - reactive_time
            ).total_seconds()
            / 3600
        )

        # This will normally be <= 0 because
        # the event has already started.

        reactive_detected = 1

    else:

        reactive_time = pd.NaT

        reactive_lead = np.nan

        reactive_detected = 0


    reactive_records.append(
        {
            "event_id": event_id,

            "event_start":
                event_start,

            "event_end":
                event_end,

            "reactive_action_time":
                reactive_time,

            "reactive_lead_time_hours":
                reactive_lead,

            "reactive_detected":
                reactive_detected
        }
    )


reactive_event_df = pd.DataFrame(
    reactive_records
)


print(
    "Reactive events detected:",
    int(
        reactive_event_df[
            "reactive_detected"
        ].sum()
    ),
    "/",
    len(test_events)
)


# ============================================================
# PART 2
# PREDICTIVE CONTROL SIMULATION
# ============================================================

print("\n" + "=" * 70)
print("PART 2 - PREDICTIVE CONTROL")
print("=" * 70)


predictive_summary = []

predictive_event_results = []


# ------------------------------------------------------------
# LOOP THROUGH MODELS
# ------------------------------------------------------------

for horizon in HORIZONS:

    print("\n" + "-" * 60)

    print(
        "MODEL:",
        horizon
    )

    threshold = THRESHOLDS[
        horizon
    ]

    target_column = (
        f"future_risk_{horizon}"
    )


    # --------------------------------------------------------
    # RECREATE VALID TEST MASK
    # --------------------------------------------------------

    valid_mask = (
        test_df[
            target_column
        ].notna()
    )


    metadata = test_df.loc[
        valid_mask,
        [
            "timestamp_utc",
            "segment_id"
        ]
    ].copy()

    metadata = (
        metadata
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # LOAD X
    # --------------------------------------------------------

    X_test = np.load(
        os.path.join(
            PROCESSED_DIR,
            f"X_test_future_risk_{horizon}.npy"
        )
    )


    # --------------------------------------------------------
    # CHECK ALIGNMENT
    # --------------------------------------------------------

    if len(X_test) != len(metadata):

        raise ValueError(
            f"Alignment error for {horizon}: "
            f"X={len(X_test)}, "
            f"time={len(metadata)}"
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = xgb.XGBClassifier()

    model.load_model(
        os.path.join(
            MODEL_DIR,
            f"xgboost_{horizon}.json"
        )
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    prediction_df = metadata.copy()

    prediction_df[
        "risk_probability"
    ] = probabilities

    prediction_df[
        "predictive_action"
    ] = (
        probabilities >= threshold
    ).astype(int)


    print(
        "Predictive actions:",
        int(
            prediction_df[
                "predictive_action"
            ].sum()
        )
    )


    # --------------------------------------------------------
    # EVENT-BY-EVENT ANALYSIS
    # --------------------------------------------------------

    detected_count = 0

    missed_count = 0

    lead_times = []


    for _, event in test_events.iterrows():

        event_id = int(
            event["event_id"]
        )

        event_start = (
            event["start_time"]
        )

        event_segment = (
            event["segment_id"]
        )


        warning_start = (
            event_start
            - pd.Timedelta(
                hours=WARNING_WINDOW_HOURS
            )
        )


        # --------------------------------------------
        # Look only inside the 48-hour warning window
        # --------------------------------------------

        candidates = prediction_df[
            (
                prediction_df[
                    "segment_id"
                ]
                == event_segment
            )
            &
            (
                prediction_df[
                    "timestamp_utc"
                ]
                >= warning_start
            )
            &
            (
                prediction_df[
                    "timestamp_utc"
                ]
                < event_start
            )
            &
            (
                prediction_df[
                    "predictive_action"
                ]
                == 1
            )
        ].copy()


        if len(candidates) > 0:

            first_action = (
                candidates[
                    "timestamp_utc"
                ].iloc[0]
            )

            lead_time = (
                (
                    event_start
                    - first_action
                ).total_seconds()
                / 3600
            )

            detected = 1

            detected_count += 1

            lead_times.append(
                lead_time
            )

        else:

            first_action = pd.NaT

            lead_time = np.nan

            detected = 0

            missed_count += 1


        predictive_event_results.append(
            {
                "horizon":
                    horizon,

                "event_id":
                    event_id,

                "event_start":
                    event_start,

                "threshold":
                    threshold,

                "predictive_action_time":
                    first_action,

                "predictive_lead_time_hours":
                    lead_time,

                "predictive_detected":
                    detected
            }
        )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_events = len(
        test_events
    )

    detection_rate = (
        detected_count
        / total_events
    )


    if len(lead_times) > 0:

        mean_lead = np.mean(
            lead_times
        )

        median_lead = np.median(
            lead_times
        )

    else:

        mean_lead = np.nan

        median_lead = np.nan


    predictive_summary.append(
        {
            "strategy":
                "predictive",

            "horizon":
                horizon,

            "threshold":
                threshold,

            "warning_window_hours":
                WARNING_WINDOW_HOURS,

            "test_events":
                total_events,

            "detected_events":
                detected_count,

            "missed_events":
                missed_count,

            "detection_rate":
                detection_rate,

            "mean_lead_time_hours":
                mean_lead,

            "median_lead_time_hours":
                median_lead,

            "total_predictive_actions":
                int(
                    prediction_df[
                        "predictive_action"
                    ].sum()
                )
        }
    )


# ============================================================
# PART 3
# COMBINE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PART 3 - COMPARISON")
print("=" * 70)


# ------------------------------------------------------------
# REACTIVE SUMMARY
# ------------------------------------------------------------

reactive_detected = int(
    reactive_event_df[
        "reactive_detected"
    ].sum()
)

reactive_missed = (
    len(test_events)
    - reactive_detected
)

reactive_detection_rate = (
    reactive_detected
    / len(test_events)
)


reactive_summary = pd.DataFrame(
    [
        {
            "strategy":
                "reactive",

            "horizon":
                "N/A",

            "threshold":
                TURBIDITY_THRESHOLD,

            "warning_window_hours":
                0,

            "test_events":
                len(test_events),

            "detected_events":
                reactive_detected,

            "missed_events":
                reactive_missed,

            "detection_rate":
                reactive_detection_rate,

            "mean_lead_time_hours":
                reactive_event_df[
                    "reactive_lead_time_hours"
                ].mean(),

            "median_lead_time_hours":
                reactive_event_df[
                    "reactive_lead_time_hours"
                ].median(),

            "total_reactive_actions":
                int(
                    test_hourly[
                        "reactive_action"
                    ].sum()
                )
        }
    ]
)


predictive_summary_df = pd.DataFrame(
    predictive_summary
)


comparison_df = pd.concat(
    [
        reactive_summary,
        predictive_summary_df
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# SAVE FILES
# ------------------------------------------------------------

reactive_file = os.path.join(
    OUTPUT_DIR,
    "reactive_event_results.csv"
)

predictive_event_file = os.path.join(
    OUTPUT_DIR,
    "predictive_event_results.csv"
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "pump_control_comparison.csv"
)


reactive_event_df.to_csv(
    reactive_file,
    index=False
)

pd.DataFrame(
    predictive_event_results
).to_csv(
    predictive_event_file,
    index=False
)

comparison_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nReactive strategy:")

print(
    reactive_summary.to_string(
        index=False
    )
)


print("\nPredictive strategies:")

print(
    predictive_summary_df.to_string(
        index=False
    )
)


print("\nFiles saved:")

print(
    reactive_file
)

print(
    predictive_event_file
)

print(
    summary_file
)


print("\n" + "=" * 70)
print("STEP 24 COMPLETED")
print("=" * 70)