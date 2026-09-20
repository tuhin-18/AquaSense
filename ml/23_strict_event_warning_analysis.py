import os
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# STEP 23 - STRICT OUT-OF-SAMPLE EVENT WARNING ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 23 - STRICT OUT-OF-SAMPLE EVENT WARNING ANALYSIS")
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
    "strict_event_warning"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

HORIZONS = [
    "1h",
    "6h",
    "12h",
    "24h"
]

# Fixed thresholds from Step 20.
# These are NOT being tuned on the test events.
THRESHOLDS = {
    "1h": 0.25,
    "6h": 0.20,
    "12h": 0.15,
    "24h": 0.15
}

# Operational warning windows.
WARNING_WINDOWS = [
    24,
    48,
    72
]


# ------------------------------------------------------------
# LOAD EVENTS
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
# LOAD PREPARED TEST DATA
# ------------------------------------------------------------

test_file = os.path.join(
    PREPARED_DIR,
    "test.csv"
)

print("\nLoading prepared test data...")

test_df = pd.read_csv(
    test_file
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
    "Test period start:",
    test_start
)

print(
    "Test period end  :",
    test_end
)


# ------------------------------------------------------------
# RESTRICT EVENTS TO TEST PERIOD
# ------------------------------------------------------------

# An event is considered an out-of-sample test event when
# its START occurs inside the test period.

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
    "\nEvents starting inside test period:",
    len(test_events)
)

if len(test_events) > 0:

    print("\nTest-period events:")

    print(
        test_events[
            [
                "event_id",
                "segment_id",
                "start_time",
                "end_time",
                "duration_hours",
                "maximum_turbidity"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# RESULTS STORAGE
# ------------------------------------------------------------

all_results = []

all_summary = []


# ------------------------------------------------------------
# LOOP THROUGH HORIZONS
# ------------------------------------------------------------

for horizon in HORIZONS:

    print("\n" + "=" * 70)
    print(
        "HORIZON:",
        horizon
    )
    print("=" * 70)

    threshold = THRESHOLDS[horizon]

    print(
        "Operating threshold:",
        threshold
    )


    # --------------------------------------------------------
    # TARGET COLUMN
    # --------------------------------------------------------

    target_column = (
        f"future_risk_{horizon}"
    )


    # --------------------------------------------------------
    # RECREATE EXACT TEST ROW MASK
    # --------------------------------------------------------

    valid_mask = (
        test_df[target_column]
        .notna()
    )

    test_metadata = test_df.loc[
        valid_mask,
        [
            "timestamp_utc",
            "segment_id"
        ]
    ].copy()

    test_metadata = (
        test_metadata
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # LOAD TEST FEATURES
    # --------------------------------------------------------

    X_file = os.path.join(
        PROCESSED_DIR,
        f"X_test_future_risk_{horizon}.npy"
    )

    X_test = np.load(
        X_file
    )


    # --------------------------------------------------------
    # LOAD TEST TARGET
    # --------------------------------------------------------

    y_file = os.path.join(
        PROCESSED_DIR,
        f"y_test_future_risk_{horizon}.npy"
    )

    y_test = np.load(
        y_file
    )


    # --------------------------------------------------------
    # ALIGNMENT CHECK
    # --------------------------------------------------------

    if (
        len(X_test)
        != len(y_test)
        or
        len(X_test)
        != len(test_metadata)
    ):

        raise ValueError(
            f"Alignment error for {horizon}: "
            f"X={len(X_test)}, "
            f"y={len(y_test)}, "
            f"time={len(test_metadata)}"
        )

    print(
        "X / y / timestamp alignment: PASSED"
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model_file = os.path.join(
        MODEL_DIR,
        f"xgboost_{horizon}.json"
    )

    print(
        "Loading model:",
        os.path.basename(model_file)
    )

    model = xgb.XGBClassifier()

    model.load_model(
        model_file
    )


    # --------------------------------------------------------
    # PREDICT PROBABILITIES
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    warnings = (
        probabilities >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # PREDICTION DATAFRAME
    # --------------------------------------------------------

    prediction_df = test_metadata.copy()

    prediction_df[
        "risk_probability"
    ] = probabilities

    prediction_df[
        "warning"
    ] = warnings


    print(
        "Total test warnings:",
        int(warnings.sum())
    )


    # --------------------------------------------------------
    # EVALUATE EACH WARNING WINDOW
    # --------------------------------------------------------

    for window_hours in WARNING_WINDOWS:

        print("\n" + "-" * 60)

        print(
            f"WARNING WINDOW: {window_hours} hours"
        )

        print("-" * 60)


        detected_events = 0

        missed_events = 0

        lead_times = []

        event_records = []


        # ----------------------------------------------------
        # EVENT LOOP
        # ----------------------------------------------------

        for _, event in test_events.iterrows():

            event_id = int(
                event["event_id"]
            )

            event_start = (
                event["start_time"]
            )

            event_end = (
                event["end_time"]
            )

            event_segment = (
                event["segment_id"]
            )


            # ------------------------------------------------
            # WARNING INTERVAL
            #
            # Example:
            # 24-hour window:
            #
            # event_start - 24h
            #          ↓
            # event_start
            #          ↓
            # event
            # ------------------------------------------------

            warning_start = (
                event_start
                - pd.Timedelta(
                    hours=window_hours
                )
            )


            # ------------------------------------------------
            # ONLY SAME SEGMENT
            # ------------------------------------------------

            candidate = prediction_df[
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
            ].copy()


            # ------------------------------------------------
            # FIND WARNINGS
            # ------------------------------------------------

            warning_rows = candidate[
                candidate[
                    "warning"
                ] == 1
            ]


            if len(warning_rows) > 0:

                first_warning = (
                    warning_rows[
                        "timestamp_utc"
                    ].iloc[0]
                )

                lead_time = (
                    (
                        event_start
                        - first_warning
                    ).total_seconds()
                    / 3600
                )

                detected = 1

                detected_events += 1

                lead_times.append(
                    lead_time
                )

            else:

                first_warning = pd.NaT

                lead_time = np.nan

                detected = 0

                missed_events += 1


            event_records.append(
                {
                    "horizon":
                        horizon,

                    "warning_window_hours":
                        window_hours,

                    "event_id":
                        event_id,

                    "segment_id":
                        event_segment,

                    "event_start":
                        event_start,

                    "event_end":
                        event_end,

                    "event_duration_hours":
                        event[
                            "duration_hours"
                        ],

                    "maximum_turbidity":
                        event[
                            "maximum_turbidity"
                        ],

                    "threshold":
                        threshold,

                    "first_warning_time":
                        first_warning,

                    "lead_time_hours":
                        lead_time,

                    "detected":
                        detected
                }
            )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        total_events = len(
            test_events
        )

        if total_events > 0:

            detection_rate = (
                detected_events
                / total_events
            )

        else:

            detection_rate = np.nan


        if len(lead_times) > 0:

            mean_lead = np.mean(
                lead_times
            )

            median_lead = np.median(
                lead_times
            )

            min_lead = np.min(
                lead_times
            )

            max_lead = np.max(
                lead_times
            )

        else:

            mean_lead = np.nan
            median_lead = np.nan
            min_lead = np.nan
            max_lead = np.nan


        # ----------------------------------------------------
        # PRINT SUMMARY
        # ----------------------------------------------------

        print(
            "Total events       :",
            total_events
        )

        print(
            "Detected events    :",
            detected_events
        )

        print(
            "Missed events      :",
            missed_events
        )

        print(
            "Detection rate     :",
            round(
                detection_rate * 100,
                2
            ),
            "%"
        )

        if len(lead_times) > 0:

            print(
                "Mean lead time     :",
                round(
                    mean_lead,
                    2
                ),
                "hours"
            )

            print(
                "Median lead time   :",
                round(
                    median_lead,
                    2
                ),
                "hours"
            )

            print(
                "Minimum lead time  :",
                round(
                    min_lead,
                    2
                ),
                "hours"
            )

            print(
                "Maximum lead time  :",
                round(
                    max_lead,
                    2
                ),
                "hours"
            )

        else:

            print(
                "No events detected."
            )


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        all_results.extend(
            event_records
        )

        all_summary.append(
            {
                "horizon":
                    horizon,

                "threshold":
                    threshold,

                "warning_window_hours":
                    window_hours,

                "test_events":
                    total_events,

                "detected_events":
                    detected_events,

                "missed_events":
                    missed_events,

                "detection_rate":
                    detection_rate,

                "mean_lead_time_hours":
                    mean_lead,

                "median_lead_time_hours":
                    median_lead,

                "minimum_lead_time_hours":
                    min_lead,

                "maximum_lead_time_hours":
                    max_lead,

                "total_test_warnings":
                    int(warnings.sum())
            }
        )


# ------------------------------------------------------------
# SAVE EVENT-LEVEL RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)

results_file = os.path.join(
    OUTPUT_DIR,
    "strict_event_warning_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------

summary_df = pd.DataFrame(
    all_summary
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "strict_event_warning_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 23 - FINAL STRICT EVENT SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nFiles saved:")

print(
    results_file
)

print(
    summary_file
)

print("\n" + "=" * 70)
print("STEP 23 COMPLETED")
print("=" * 70)