import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# AQUASENSE - WATER QUALITY TIME-SERIES VISUALIZATION
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "SEAN_FQ_Q_2011-2024_KLGO_TA.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "ml_outputs"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "water_quality_plots"
)

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

PARAMETERS = [
    "water_temperature",
    "sp_conductance",
    "ph",
    "do_concentration",
    "do_saturation",
    "turbidity"
]


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("AQUASENSE WATER QUALITY TIME-SERIES VISUALIZATION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)

print("Dataset loaded successfully.")


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    errors="coerce",
    utc=True
)

df = df.dropna(
    subset=["timestamp_utc"]
).copy()

df = df.sort_values(
    "timestamp_utc"
).reset_index(drop=True)


# ============================================================
# CREATE CONTINUOUS SEGMENTS
# ============================================================

print("\nIdentifying continuous monitoring segments...")

df["time_difference"] = (
    df["timestamp_utc"].diff()
)

SEGMENT_GAP = pd.Timedelta(hours=6)

df["new_segment"] = (
    df["time_difference"] > SEGMENT_GAP
)

df.loc[0, "new_segment"] = True

df["segment_id"] = (
    df["new_segment"].cumsum()
)

print(
    f"Continuous segments found: "
    f"{df['segment_id'].nunique()}"
)


# ============================================================
# PLOT EACH PARAMETER
# ============================================================

for parameter in PARAMETERS:

    print(
        f"\nCreating plot for: {parameter}"
    )

    plt.figure(
        figsize=(16, 7)
    )

    # Plot every continuous segment separately
    for segment_id, group in df.groupby(
        "segment_id"
    ):

        plt.plot(
            group["timestamp_utc"],
            group[parameter],
            linewidth=0.6
        )

    plt.title(
        f"{parameter} - Continuous Monitoring Segments"
    )

    plt.xlabel(
        "Time (UTC)"
    )

    plt.ylabel(
        parameter
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_file = os.path.join(
        PLOT_DIR,
        f"{parameter}_full_timeseries.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# TURBIDITY FOCUSED PLOT
# ============================================================

print("\nCreating turbidity focused plot...")

plt.figure(
    figsize=(16, 7)
)

for segment_id, group in df.groupby(
    "segment_id"
):

    plt.plot(
        group["timestamp_utc"],
        group["turbidity"],
        linewidth=0.6
    )

plt.title(
    "Turbidity - Continuous Monitoring Segments"
)

plt.xlabel(
    "Time (UTC)"
)

plt.ylabel(
    "Turbidity"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

turbidity_output = os.path.join(
    PLOT_DIR,
    "turbidity_focused.png"
)

plt.savefig(
    turbidity_output,
    dpi=200
)

plt.close()

print(
    f"Saved: {turbidity_output}"
)


# ============================================================
# SAVE SEGMENT INFORMATION
# ============================================================

segment_information = []

for segment_id, group in df.groupby(
    "segment_id"
):

    segment_information.append({
        "segment_id": int(segment_id),
        "start_time": group["timestamp_utc"].min(),
        "end_time": group["timestamp_utc"].max(),
        "observations": len(group)
    })

segment_df = pd.DataFrame(
    segment_information
)

segment_output = os.path.join(
    OUTPUT_DIR,
    "plot_segment_information.csv"
)

segment_df.to_csv(
    segment_output,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PLOTS CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nPlot folder:\n{PLOT_DIR}"
)

print(
    f"\nSegment information:\n"
    f"{segment_output}"
)

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)