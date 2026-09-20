import os
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# STEP 26
# EVENT-LEVEL INTERVENTION EFFICIENCY ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 26 - EVENT-LEVEL INTERVENTION EFFICIENCY ANALYSIS")
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
    "event_level_intervention_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

WARNING_WINDOW_HOURS = 48

HORIZONS = [
    "12h",
    "24h"
]

THRESHOLDS = {
    "12h": 0.15,
    "24h": 0.15
}


# ============================================================
# LOAD EVENTS
# ============================================================

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


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

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
# TEST-PERIOD EVENTS
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
    "Events starting in test period:",
    len(test_events)
)


# ============================================================
# FUNCTION: BUILD PREDICTION EPISODES
# ============================================================

def build_prediction_episodes(prediction_df):

    actions = prediction_df[
        prediction_df[
            "predictive_action"
        ] == 1
    ].copy()

    actions = (
        actions
        .sort_values(
            [
                "segment_id",
                "timestamp_utc"
            ]
        )
        .reset_index(drop=True)
    )

    if len(actions) == 0:
        return pd.DataFrame()

    actions[
        "time_gap_hours"
    ] = (
        actions
        .groupby("segment_id")[
            "timestamp_utc"
        ]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    actions[
        "new_episode"
    ] = (
        actions[
            "time_gap_hours"
        ].isna()
        |
        (
            actions[
                "time_gap_hours"
            ] > 1.5
        )
    ).astype(int)

    actions[
        "episode_number"
    ] = (
        actions
        .groupby("segment_id")[
            "new_episode"
        ]
        .cumsum()
    )

    episodes = (
        actions
        .groupby(
            [
                "segment_id",
                "episode_number"
            ],
            as_index=False
        )
        .agg(
            intervention_start=(
                "timestamp_utc",
                "min"
            ),
            intervention_end=(
                "timestamp_utc",
                "max"
            ),
            warning_hours=(
                "timestamp_utc",
                "count"
            ),
            maximum_probability=(
                "risk_probability",
                "max"
            )
        )
    )

    episodes[
        "duration_hours"
    ] = (
        (
            episodes[
                "intervention_end"
            ]
            -
            episodes[
                "intervention_start"
            ]
        )
        .dt.total_seconds()
        / 3600
    ) + 1

    return episodes


# ============================================================
# ANALYZE EACH MODEL
# ============================================================

all_event_results = []

all_episode_results = []

summary_results = []


for horizon in HORIZONS:

    print("\n" + "=" * 70)
    print("MODEL:", horizon)
    print("=" * 70)


    threshold = THRESHOLDS[
        horizon
    ]

    target_column = (
        f"future_risk_{horizon}"
    )


    # --------------------------------------------------------
    # VALID TEST ROWS
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
    # LOAD TEST FEATURES
    # --------------------------------------------------------

    X_test = np.load(
        os.path.join(
            PROCESSED_DIR,
            f"X_test_future_risk_{horizon}.npy"
        )
    )


    if len(X_test) != len(metadata):

        raise ValueError(
            f"Alignment error for {horizon}: "
            f"X={len(X_test)}, "
            f"metadata={len(metadata)}"
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


    # --------------------------------------------------------
    # BUILD INTERVENTION EPISODES
    # --------------------------------------------------------

    episodes = build_prediction_episodes(
        prediction_df
    )

    print(
        "Total intervention episodes:",
        len(episodes)
    )


    # ========================================================
    # EVENT-LEVEL ANALYSIS
    # ========================================================

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

        segment_id = (
            event["segment_id"]
        )


        warning_start = (
            event_start
            -
            pd.Timedelta(
                hours=WARNING_WINDOW_HOURS
            )
        )


        # ----------------------------------------------------
        # Find intervention episodes BEFORE event
        # ----------------------------------------------------

        event_warnings = episodes[
            (
                episodes[
                    "segment_id"
                ]
                == segment_id
            )
            &
            (
                episodes[
                    "intervention_start"
                ]
                >= warning_start
            )
            &
            (
                episodes[
                    "intervention_start"
                ]
                < event_start
            )
        ].copy()


        event_warnings = (
            event_warnings
            .sort_values(
                "intervention_start"
            )
            .reset_index(drop=True)
        )


        # ----------------------------------------------------
        # First warning
        # ----------------------------------------------------

        if len(event_warnings) > 0:

            first_warning = (
                event_warnings[
                    "intervention_start"
                ].iloc[0]
            )

            first_lead = (
                (
                    event_start
                    -
                    first_warning
                )
                .total_seconds()
                / 3600
            )

            last_warning = (
                event_warnings[
                    "intervention_start"
                ].iloc[-1]
            )

            number_of_warnings = (
                len(event_warnings)
            )

            total_warning_hours = (
                event_warnings[
                    "warning_hours"
                ].sum()
            )

            detected = 1

        else:

            first_warning = pd.NaT

            first_lead = np.nan

            last_warning = pd.NaT

            number_of_warnings = 0

            total_warning_hours = 0

            detected = 0


        # ----------------------------------------------------
        # Time from first warning to event end
        # ----------------------------------------------------

        if detected:

            warning_to_event_end = (
                (
                    event_end
                    -
                    first_warning
                )
                .total_seconds()
                / 3600
            )

        else:

            warning_to_event_end = np.nan


        # ----------------------------------------------------
        # Save event result
        # ----------------------------------------------------

        all_event_results.append(
            {
                "horizon":
                    horizon,

                "event_id":
                    event_id,

                "event_start":
                    event_start,

                "event_end":
                    event_end,

                "segment_id":
                    segment_id,

                "threshold":
                    threshold,

                "warning_window_hours":
                    WARNING_WINDOW_HOURS,

                "detected":
                    detected,

                "first_warning_time":
                    first_warning,

                "first_warning_lead_hours":
                    first_lead,

                "last_warning_time":
                    last_warning,

                "number_of_intervention_episodes":
                    number_of_warnings,

                "total_warning_hours":
                    total_warning_hours,

                "warning_to_event_end_hours":
                    warning_to_event_end
            }
        )


        # ----------------------------------------------------
        # Save individual warning episodes
        # ----------------------------------------------------

        for warning_number, (_, warning) in enumerate(
            event_warnings.iterrows(),
            start=1
        ):

            all_episode_results.append(
                {
                    "horizon":
                        horizon,

                    "event_id":
                        event_id,

                    "warning_number":
                        warning_number,

                    "segment_id":
                        segment_id,

                    "intervention_start":
                        warning[
                            "intervention_start"
                        ],

                    "intervention_end":
                        warning[
                            "intervention_end"
                        ],

                    "duration_hours":
                        warning[
                            "duration_hours"
                        ],

                    "warning_hours":
                        warning[
                            "warning_hours"
                        ],

                    "maximum_probability":
                        warning[
                            "maximum_probability"
                        ],

                    "lead_time_to_event_hours":
                        (
                            (
                                event_start
                                -
                                warning[
                                    "intervention_start"
                                ]
                            )
                            .total_seconds()
                            / 3600
                        )
                }
            )


    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    model_event_df = pd.DataFrame(
        [
            row
            for row in all_event_results
            if row["horizon"] == horizon
        ]
    )


    detected_events = int(
        model_event_df[
            "detected"
        ].sum()
    )

    total_events = len(
        model_event_df
    )

    missed_events = (
        total_events
        -
        detected_events
    )

    detection_rate = (
        detected_events
        /
        total_events
    )


    lead_values = (
        model_event_df[
            "first_warning_lead_hours"
        ]
        .dropna()
    )


    if len(lead_values) > 0:

        mean_lead = (
            lead_values.mean()
        )

        median_lead = (
            lead_values.median()
        )

        minimum_lead = (
            lead_values.min()
        )

        maximum_lead = (
            lead_values.max()
        )

    else:

        mean_lead = np.nan

        median_lead = np.nan

        minimum_lead = np.nan

        maximum_lead = np.nan


    mean_warnings_per_detected_event = (
        model_event_df.loc[
            model_event_df[
                "detected"
            ] == 1,
            "number_of_intervention_episodes"
        ].mean()
    )


    median_warnings_per_detected_event = (
        model_event_df.loc[
            model_event_df[
                "detected"
            ] == 1,
            "number_of_intervention_episodes"
        ].median()
    )


    total_event_warning_episodes = int(
        model_event_df[
            "number_of_intervention_episodes"
        ].sum()
    )


    summary_results.append(
        {
            "horizon":
                horizon,

            "threshold":
                threshold,

            "test_events":
                total_events,

            "detected_events":
                detected_events,

            "missed_events":
                missed_events,

            "event_detection_rate":
                detection_rate,

            "mean_first_warning_lead_hours":
                mean_lead,

            "median_first_warning_lead_hours":
                median_lead,

            "minimum_first_warning_lead_hours":
                minimum_lead,

            "maximum_first_warning_lead_hours":
                maximum_lead,

            "total_warning_episodes_associated_with_events":
                total_event_warning_episodes,

            "mean_warning_episodes_per_detected_event":
                mean_warnings_per_detected_event,

            "median_warning_episodes_per_detected_event":
                median_warnings_per_detected_event
        }
    )


    # --------------------------------------------------------
    # PRINT MODEL SUMMARY
    # --------------------------------------------------------

    print(
        "\nEvents detected:",
        detected_events,
        "/",
        total_events
    )

    print(
        "Missed events:",
        missed_events
    )

    print(
        "Detection rate:",
        round(
            detection_rate,
            4
        )
    )

    print(
        "Mean first-warning lead:",
        round(
            mean_lead,
            2
        ),
        "hours"
    )

    print(
        "Median first-warning lead:",
        round(
            median_lead,
            2
        ),
        "hours"
    )

    print(
        "Warning episodes associated with detected events:",
        total_event_warning_episodes
    )

    print(
        "Mean warning episodes per detected event:",
        round(
            mean_warnings_per_detected_event,
            2
        )
    )


# ============================================================
# SAVE EVENT-LEVEL RESULTS
# ============================================================

event_results_df = pd.DataFrame(
    all_event_results
)

episode_results_df = pd.DataFrame(
    all_episode_results
)

summary_df = pd.DataFrame(
    summary_results
)


event_file = os.path.join(
    OUTPUT_DIR,
    "event_level_results.csv"
)

episode_file = os.path.join(
    OUTPUT_DIR,
    "event_warning_episode_details.csv"
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "event_level_summary.csv"
)


event_results_df.to_csv(
    event_file,
    index=False
)

episode_results_df.to_csv(
    episode_file,
    index=False
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PRINT EVENT-BY-EVENT TABLE
# ============================================================

print("\n" + "=" * 70)
print("EVENT-BY-EVENT RESULTS")
print("=" * 70)


display_columns = [
    "horizon",
    "event_id",
    "event_start",
    "detected",
    "first_warning_lead_hours",
    "number_of_intervention_episodes",
    "total_warning_hours"
]


print(
    event_results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


print("\nFiles saved:")

print(
    event_file
)

print(
    episode_file
)

print(
    summary_file
)


print("\n" + "=" * 70)
print("STEP 26 COMPLETED")
print("=" * 70)