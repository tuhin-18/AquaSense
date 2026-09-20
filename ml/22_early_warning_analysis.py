import os
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# STEP 22 - EVENT-LEVEL EARLY WARNING ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 22 - EVENT-LEVEL EARLY WARNING ANALYSIS")
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
    "early_warning_analysis"
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

# Thresholds selected from Step 20
THRESHOLDS = {
    "1h": 0.25,
    "6h": 0.20,
    "12h": 0.15,
    "24h": 0.15
}


# ------------------------------------------------------------
# LOAD EVENTS
# ------------------------------------------------------------

print("\nLoading detected events...")

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
    "Events loaded:",
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

test_df = test_df.sort_values(
    "timestamp_utc"
).reset_index(drop=True)

print(
    "Prepared test rows:",
    len(test_df)
)


# ------------------------------------------------------------
# ANALYZE EACH HORIZON
# ------------------------------------------------------------

all_event_results = []
all_summary_results = []


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


    if target_column not in test_df.columns:

        raise ValueError(
            f"Target column not found: "
            f"{target_column}"
        )


    # --------------------------------------------------------
    # RECONSTRUCT EXACT TEST ROWS
    # --------------------------------------------------------
    #
    # Step 12 removed rows where the target was NaN.
    # We reproduce that exact filtering here so that
    # timestamps align with the X_test array.
    #

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
    # LOAD X TEST ARRAY
    # --------------------------------------------------------

    X_file = os.path.join(
        PROCESSED_DIR,
        f"X_test_future_risk_{horizon}.npy"
    )

    print(
        "Loading:",
        os.path.basename(X_file)
    )

    X_test = np.load(
        X_file
    )


    # --------------------------------------------------------
    # LOAD Y TEST ARRAY
    # --------------------------------------------------------

    y_file = os.path.join(
        PROCESSED_DIR,
        f"y_test_future_risk_{horizon}.npy"
    )

    y_test = np.load(
        y_file
    )


    # --------------------------------------------------------
    # VERIFY ALIGNMENT
    # --------------------------------------------------------

    print(
        "X_test rows:",
        len(X_test)
    )

    print(
        "y_test rows:",
        len(y_test)
    )

    print(
        "Timestamp rows:",
        len(test_metadata)
    )


    if (
        len(X_test)
        != len(y_test)
        or
        len(X_test)
        != len(test_metadata)
    ):

        raise ValueError(
            "\nALIGNMENT ERROR!\n"
            f"X_test: {len(X_test)}\n"
            f"y_test: {len(y_test)}\n"
            f"timestamps: {len(test_metadata)}"
        )

    print(
        "X / y / timestamp alignment: PASSED"
    )


    # --------------------------------------------------------
    # LOAD XGBOOST MODEL
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


    print(
        "Predicted warnings:",
        int(warnings.sum())
    )


    # --------------------------------------------------------
    # CREATE PREDICTION DATAFRAME
    # --------------------------------------------------------

    prediction_df = test_metadata.copy()

    prediction_df[
        "risk_probability"
    ] = probabilities

    prediction_df[
        "warning"
    ] = warnings


    # --------------------------------------------------------
    # EVENT-LEVEL ANALYSIS
    # --------------------------------------------------------

    detected_events = 0

    missed_events = 0

    lead_times = []

    horizon_event_results = []


    for _, event in events.iterrows():

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


        # ----------------------------------------------------
        # IMPORTANT:
        # Only compare warnings from the SAME monitoring
        # segment as the event.
        #
        # This prevents a warning from a completely different
        # seasonal monitoring period from being associated
        # with this event.
        # ----------------------------------------------------

        before_event = prediction_df[
            (
                prediction_df["segment_id"]
                == event_segment
            )
            &
            (
                prediction_df["timestamp_utc"]
                < event_start
            )
        ].copy()


        # ----------------------------------------------------
        # LOOK BACK 7 DAYS
        # ----------------------------------------------------

        warning_window_start = (
            event_start
            - pd.Timedelta(days=7)
        )

        before_event = before_event[
            before_event["timestamp_utc"]
            >= warning_window_start
        ]


        # ----------------------------------------------------
        # FIND WARNINGS
        # ----------------------------------------------------

        warning_rows = before_event[
            before_event["warning"] == 1
        ].copy()


        if len(warning_rows) > 0:

            # Earliest warning before event
            first_warning = (
                warning_rows[
                    "timestamp_utc"
                ].iloc[0]
            )

            lead_time_hours = (
                (
                    event_start
                    - first_warning
                ).total_seconds()
                / 3600
            )

            detected = 1

            detected_events += 1

            lead_times.append(
                lead_time_hours
            )

        else:

            first_warning = pd.NaT

            lead_time_hours = np.nan

            detected = 0

            missed_events += 1


        horizon_event_results.append(
            {
                "horizon": horizon,
                "event_id": event_id,
                "segment_id": event_segment,
                "event_start": event_start,
                "event_end": event_end,
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
                "warning_lead_time_hours":
                    lead_time_hours,
                "detected":
                    detected
            }
        )


    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    total_events = len(events)


    if total_events > 0:

        detection_rate = (
            detected_events
            / total_events
        )

    else:

        detection_rate = np.nan


    if len(lead_times) > 0:

        mean_lead_time = (
            np.mean(lead_times)
        )

        median_lead_time = (
            np.median(lead_times)
        )

        minimum_lead_time = (
            np.min(lead_times)
        )

        maximum_lead_time = (
            np.max(lead_times)
        )

    else:

        mean_lead_time = np.nan
        median_lead_time = np.nan
        minimum_lead_time = np.nan
        maximum_lead_time = np.nan


    # --------------------------------------------------------
    # SUMMARY RECORD
    # --------------------------------------------------------

    summary = {

        "horizon":
            horizon,

        "threshold":
            threshold,

        "total_events":
            total_events,

        "detected_events":
            detected_events,

        "missed_events":
            missed_events,

        "detection_rate":
            detection_rate,

        "mean_lead_time_hours":
            mean_lead_time,

        "median_lead_time_hours":
            median_lead_time,

        "minimum_lead_time_hours":
            minimum_lead_time,

        "maximum_lead_time_hours":
            maximum_lead_time,

        "total_test_warnings":
            int(warnings.sum())
    }


    all_summary_results.append(
        summary
    )

    all_event_results.extend(
        horizon_event_results
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nEvent detection results:")

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

    if not np.isnan(
        mean_lead_time
    ):

        print(
            "Mean lead time     :",
            round(
                mean_lead_time,
                2
            ),
            "hours"
        )

        print(
            "Median lead time   :",
            round(
                median_lead_time,
                2
            ),
            "hours"
        )

        print(
            "Minimum lead time  :",
            round(
                minimum_lead_time,
                2
            ),
            "hours"
        )

        print(
            "Maximum lead time  :",
            round(
                maximum_lead_time,
                2
            ),
            "hours"
        )

    else:

        print(
            "No events detected."
        )


# ------------------------------------------------------------
# SAVE EVENT-LEVEL RESULTS
# ------------------------------------------------------------

event_results_df = pd.DataFrame(
    all_event_results
)

event_results_file = os.path.join(
    OUTPUT_DIR,
    "event_level_warning_results.csv"
)

event_results_df.to_csv(
    event_results_file,
    index=False
)


# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------

summary_df = pd.DataFrame(
    all_summary_results
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "early_warning_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EARLY-WARNING SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nFiles saved:")

print(
    event_results_file
)

print(
    summary_file
)

print("\n" + "=" * 70)
print("STEP 22 COMPLETED")
print("=" * 70)