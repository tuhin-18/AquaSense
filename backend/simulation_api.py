from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os


# ============================================================
# AQUASENSE - HISTORICAL SIMULATION API
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "simulation",
    "pump_control_2022_simulation.csv"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AquaSense Simulation API",
    description="Historical ML-based water quality simulation",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Simulation file not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "system": "AquaSense",
        "status": "running",
        "mode": "historical_simulation",
        "dataset": "2022 water-quality event"
    }


# ============================================================
# SIMULATION INFO
# ============================================================

@app.get("/simulation/info")
def simulation_info():

    return {
        "total_rows": len(df),
        "start_time": df["timestamp"].min().isoformat(),
        "end_time": df["timestamp"].max().isoformat(),
        "event_threshold_fnu": 319.81,
        "predictive_threshold": 0.25
    }


# ============================================================
# ALL SIMULATION DATA
# ============================================================

@app.get("/simulation/data")
def simulation_data():

    records = []

    for _, row in df.iterrows():

        records.append({
            "timestamp": row["timestamp"].isoformat(),

            "turbidity": None if pd.isna(row["turbidity"])
                else float(row["turbidity"]),

            "ph": None if pd.isna(row["ph"])
                else float(row["ph"]),

            "do_concentration": None if pd.isna(row["do_concentration"])
                else float(row["do_concentration"]),

            "do_saturation": None if pd.isna(row["do_saturation"])
                else float(row["do_saturation"]),

            "sp_conductance": None if pd.isna(row["sp_conductance"])
                else float(row["sp_conductance"]),

            "water_temperature": None if pd.isna(row["water_temperature"])
                else float(row["water_temperature"]),

            "risk_probability_1h": float(
                row["risk_probability_1h"]
            ),

            "risk_probability_6h": float(
                row["risk_probability_6h"]
            ),

            "risk_probability_12h": float(
                row["risk_probability_12h"]
            ),

            "risk_probability_24h": float(
                row["risk_probability_24h"]
            ),

            "predicted_risk_horizon": str(
                row["predicted_risk_horizon"]
            ),

            "risk_level": str(
                row["risk_level"]
            ),

            "predictive_pump_action": bool(
                row["predictive_pump_action"]
            ),

            "reactive_pump_action": bool(
                row["reactive_pump_action"]
            ),

            "actual_high_turbidity": bool(
                row["actual_high_turbidity"]
            )
        })

    return records


# ============================================================
# SINGLE TIMESTAMP
# ============================================================

@app.get("/simulation/row/{index}")
def simulation_row(index: int):

    if index < 0 or index >= len(df):
        return {
            "error": "Simulation index out of range"
        }

    row = df.iloc[index]

    return {
        "index": index,
        "timestamp": row["timestamp"].isoformat(),

        "turbidity": float(row["turbidity"])
            if pd.notna(row["turbidity"]) else None,

        "ph": float(row["ph"])
            if pd.notna(row["ph"]) else None,

        "do_concentration": float(row["do_concentration"])
            if pd.notna(row["do_concentration"]) else None,

        "do_saturation": float(row["do_saturation"])
            if pd.notna(row["do_saturation"]) else None,

        "sp_conductance": float(row["sp_conductance"])
            if pd.notna(row["sp_conductance"]) else None,

        "water_temperature": float(row["water_temperature"])
            if pd.notna(row["water_temperature"]) else None,

        "risk_probability_1h": float(
            row["risk_probability_1h"]
        ),

        "risk_probability_6h": float(
            row["risk_probability_6h"]
        ),

        "risk_probability_12h": float(
            row["risk_probability_12h"]
        ),

        "risk_probability_24h": float(
            row["risk_probability_24h"]
        ),

        "predicted_risk_horizon": str(
            row["predicted_risk_horizon"]
        ),

        "risk_level": str(
            row["risk_level"]
        ),

        "predictive_pump_action": bool(
            row["predictive_pump_action"]
        ),

        "reactive_pump_action": bool(
            row["reactive_pump_action"]
        ),

        "actual_high_turbidity": bool(
            row["actual_high_turbidity"]
        )
    }