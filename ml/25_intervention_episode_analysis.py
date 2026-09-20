import os
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# STEP 25
# PREDICTIVE INTERVENTION EPISODE ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 25 - PREDICTIVE INTERVENTION EPISODE ANALYSIS")
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

HOURLY_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "water_quality_hourly_regularized.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "intervention_episode_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

TURBIDITY_THRESHOLD = 319.81

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
    "Total events:",
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
# FUNCTION: BUILD INTERVENTION EPISODES
# ============================================================

def build_episodes(prediction_df):
    """
    Convert consecutive predictive-action hours
    into intervention episodes.

    A new episode starts when:
    - the previous row was not an action, OR
    - the time gap is greater than 1 hour, OR
    - the segment changes.
    """

    actions = prediction_df[
        prediction_df["predictive_action"] == 1
    ].copy()

    actions = (
        actions
        .sort_values(
            ["segment_id", "timestamp_utc"]
        )
        .reset_index(drop=True)
    )

    if len(actions) == 0:
        return pd.DataFrame()

    actions["time_gap_hours"] = (
        actions
        .groupby("segment_id")[
            "timestamp_utc"
        ]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    actions["new_episode"] = (
        actions["time_gap_hours"].isna()
        |
        (actions["time_gap_hours"] > 1.5)
    ).astype(int)

    actions["episode_number"] = (
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
            number_of_warning_hours=(
                "timestamp_utc",
                "count"
            ),
            maximum_risk_probability=(
                "risk_probability",
                "max"
            ),
            mean_risk_probability=(
                "risk_probability",
                "mean"
            )
        )
    )

    episodes[
        "duration_hours"
    ] = (
        (
            episodes["intervention_end"]
            -
            episodes["intervention_start"]
        )
        .dt.total_seconds()
        / 3600
    ) + 1

    return episodes


# ============================================================
# ANALYZE EACH MODEL
# ============================================================

all_episode_results = []

summary_results = []


for horizon in HORIZONS:

    print("\n" + "=" * 70)

    print(
        "ANALYZING:",
        horizon
    )

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
            f"Alignment problem for {horizon}: "
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
    # BUILD EPISODES
    # --------------------------------------------------------

    episodes = build_episodes(
        prediction_df
    )


    if len(episodes) == 0:

        print(
            "No intervention episodes found."
        )

        continue


    print(
        "Intervention episodes:",
        len(episodes)
    )


    # --------------------------------------------------------
    # MATCH EACH EPISODE TO FUTURE EVENT
    # --------------------------------------------------------

    episode_records = []


    for episode_index, episode in episodes.iterrows():

        intervention_time = (
            episode["intervention_start"]
        )

        segment_id = (
            episode["segment_id"]
        )


        future_limit = (
            intervention_time
            +
            pd.Timedelta(
                hours=WARNING_WINDOW_HOURS
            )
        )


        # Events occurring after intervention
        # and within 48 hours.

        candidate_events = test_events[
            (
                test_events["segment_id"]
                == segment_id
            )
            &
            (
                test_events["start_time"]
                > intervention_time
            )
            &
            (
                test_events["start_time"]
                <= future_limit
            )
        ].copy()


        if len(candidate_events) > 0:

            matched_event = (
                candidate_events
                .sort_values("start_time")
                .iloc[0]
            )

            event_id = int(
                matched_event["event_id"]
            )

            event_start = (
                matched_event["start_time"]
            )

            lead_time = (
                (
                    event_start
                    - intervention_time
                )
                .total_seconds()
                / 3600
            )

            useful = 1

        else:

            event_id = np.nan

            event_start = pd.NaT

            lead_time = np.nan

            useful = 0


        episode_records.append(
            {
                "horizon":
                    horizon,

                "segment_id":
                    segment_id,

                "intervention_start":
                    intervention_time,

                "intervention_end":
                    episode[
                        "intervention_end"
                    ],

                "duration_hours":
                    episode[
                        "duration_hours"
                    ],

                "warning_hours":
                    episode[
                        "number_of_warning_hours"
                    ],

                "maximum_risk_probability":
                    episode[
                        "maximum_risk_probability"
                    ],

                "mean_risk_probability":
                    episode[
                        "mean_risk_probability"
                    ],

                "matched_event_id":
                    event_id,

                "matched_event_start":
                    event_start,

                "lead_time_hours":
                    lead_time,

                "useful_intervention":
                    useful
            }
        )


    episode_df = pd.DataFrame(
        episode_records
    )


    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    total_episodes = len(
        episode_df
    )

    useful_episodes = int(
        episode_df[
            "useful_intervention"
        ].sum()
    )

    unnecessary_episodes = (
        total_episodes
        -
        useful_episodes
    )


    if total_episodes > 0:

        intervention_precision = (
            useful_episodes
            /
            total_episodes
        )

    else:

        intervention_precision = np.nan


    lead_values = (
        episode_df[
            "lead_time_hours"
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

    else:

        mean_lead = np.nan

        median_lead = np.nan


    # --------------------------------------------------------
    # EVENT DETECTION
    # --------------------------------------------------------

    detected_event_ids = set(
        episode_df.loc[
            episode_df[
                "useful_intervention"
            ] == 1,
            "matched_event_id"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )


    detected_events = len(
        detected_event_ids
    )

    total_events = len(
        test_events
    )

    event_detection_rate = (
        detected_events
        /
        total_events
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        "Total intervention episodes:",
        total_episodes
    )

    print(
        "Useful episodes:",
        useful_episodes
    )

    print(
        "Unnecessary episodes:",
        unnecessary_episodes
    )

    print(
        "Intervention precision:",
        round(
            intervention_precision,
            4
        )
    )

    print(
        "Events detected:",
        detected_events,
        "/",
        total_events
    )

    print(
        "Event detection rate:",
        round(
            event_detection_rate,
            4
        )
    )

    print(
        "Mean lead time:",
        round(
            mean_lead,
            2
        ),
        "hours"
    )

    print(
        "Median lead time:",
        round(
            median_lead,
            2
        ),
        "hours"
    )


    # --------------------------------------------------------
    # SAVE MODEL EPISODES
    # --------------------------------------------------------

    episode_file = os.path.join(
        OUTPUT_DIR,
        f"{horizon}_intervention_episodes.csv"
    )

    episode_df.to_csv(
        episode_file,
        index=False
    )


    all_episode_results.append(
        episode_df
    )


    summary_results.append(
        {
            "strategy":
                "predictive",

            "horizon":
                horizon,

            "threshold":
                threshold,

            "evaluation_window_hours":
                WARNING_WINDOW_HOURS,

            "total_intervention_episodes":
                total_episodes,

            "useful_intervention_episodes":
                useful_episodes,

            "unnecessary_intervention_episodes":
                unnecessary_episodes,

            "intervention_precision":
                intervention_precision,

            "events_in_test_period":
                total_events,

            "events_detected":
                detected_events,

            "event_detection_rate":
                event_detection_rate,

            "mean_lead_time_hours":
                mean_lead,

            "median_lead_time_hours":
                median_lead,

            "mean_episode_duration_hours":
                episode_df[
                    "duration_hours"
                ].mean(),

            "median_episode_duration_hours":
                episode_df[
                    "duration_hours"
                ].median()
        }
    )


# ============================================================
# REACTIVE BASELINE
# ============================================================

print("\n" + "=" * 70)
print("REACTIVE BASELINE")
print("=" * 70)


hourly_df = pd.read_csv(
    HOURLY_FILE
)

hourly_df["timestamp_utc"] = pd.to_datetime(
    hourly_df["timestamp_utc"],
    utc=True
)

test_hourly = hourly_df[
    (
        hourly_df["timestamp_utc"]
        >= test_start
    )
    &
    (
        hourly_df["timestamp_utc"]
        <= test_end
    )
].copy()


test_hourly[
    "reactive_action"
] = (
    test_hourly[
        "turbidity"
    ]
    >= TURBIDITY_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# Group reactive actions into episodes
# ------------------------------------------------------------

reactive_actions = test_hourly[
    test_hourly[
        "reactive_action"
    ] == 1
].copy()

reactive_actions = (
    reactive_actions
    .sort_values(
        [
            "segment_id",
            "timestamp_utc"
        ]
    )
    .reset_index(drop=True)
)


if len(reactive_actions) > 0:

    reactive_actions[
        "time_gap_hours"
    ] = (
        reactive_actions
        .groupby("segment_id")[
            "timestamp_utc"
        ]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    reactive_actions[
        "new_episode"
    ] = (
        reactive_actions[
            "time_gap_hours"
        ].isna()
        |
        (
            reactive_actions[
                "time_gap_hours"
            ] > 1.5
        )
    ).astype(int)

    reactive_actions[
        "episode_number"
    ] = (
        reactive_actions
        .groupby("segment_id")[
            "new_episode"
        ]
        .cumsum()
    )


    reactive_episodes = (
        reactive_actions
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
            )
        )
    )


    reactive_episodes[
        "duration_hours"
    ] = (
        (
            reactive_episodes[
                "intervention_end"
            ]
            -
            reactive_episodes[
                "intervention_start"
            ]
        )
        .dt.total_seconds()
        / 3600
    ) + 1

else:

    reactive_episodes = pd.DataFrame()


print(
    "Reactive intervention episodes:",
    len(reactive_episodes)
)


# ============================================================
# COMBINE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_results
)


if len(reactive_episodes) > 0:

    reactive_summary = pd.DataFrame(
        [
            {
                "strategy":
                    "reactive",

                "horizon":
                    "N/A",

                "threshold":
                    TURBIDITY_THRESHOLD,

                "evaluation_window_hours":
                    0,

                "total_intervention_episodes":
                    len(reactive_episodes),

                "useful_intervention_episodes":
                    np.nan,

                "unnecessary_intervention_episodes":
                    np.nan,

                "intervention_precision":
                    np.nan,

                "events_in_test_period":
                    len(test_events),

                "events_detected":
                    len(test_events),

                "event_detection_rate":
                    1.0,

                "mean_lead_time_hours":
                    0.0,

                "median_lead_time_hours":
                    0.0,

                "mean_episode_duration_hours":
                    reactive_episodes[
                        "duration_hours"
                    ].mean(),

                "median_episode_duration_hours":
                    reactive_episodes[
                        "duration_hours"
                    ].median()
            }
        ]
    )

    final_summary = pd.concat(
        [
            reactive_summary,
            summary_df
        ],
        ignore_index=True
    )

else:

    final_summary = summary_df


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "intervention_episode_summary.csv"
)

final_summary.to_csv(
    summary_file,
    index=False
)


if len(all_episode_results) > 0:

    all_episodes_df = pd.concat(
        all_episode_results,
        ignore_index=True
    )

    all_episode_file = os.path.join(
        OUTPUT_DIR,
        "all_predictive_intervention_episodes.csv"
    )

    all_episodes_df.to_csv(
        all_episode_file,
        index=False
    )


# ============================================================
# FINAL PRINT
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    final_summary.to_string(
        index=False
    )
)


print("\nFiles saved to:")

print(
    OUTPUT_DIR
)

print("\n" + "=" * 70)
print("STEP 25 COMPLETED")
print("=" * 70)