/**
 * ============================================================
 * AquaSense
 * Predictive Water Quality Dashboard
 * ============================================================
 *
 * Historical ML Water Quality Simulation
 *
 * Data source:
 * FastAPI -> historical pump-control simulation
 *
 * This dashboard replays the real 2022 water-quality event.
 *
 * IMPORTANT:
 * - Uses the elements already present in index.html.
 * - Does NOT dynamically create duplicate cards/buttons.
 * - Displays ML risk probabilities.
 * - Displays 6-hour parameter trends.
 * - Displays predictive pump-control status.
 * ============================================================
 */


// ============================================================
// CONFIGURATION
// ============================================================

const API_BASE = "https://aquasense-api-gjib.onrender.com";
// Starting point:
// 26 Sep 2022 14:00 UTC
//
// At this point the model produced approximately:
// 12h risk = 40.59%
// 24h risk = 57.16%
// Preventive pump action = TRUE

const START_INDEX = 62;


// ============================================================
// APPLICATION STATE
// ============================================================

const state = {
    data: [],
    currentIndex: START_INDEX,
    controlMode: "auto",
    pumpActive: false,
    loaded: false
};


// ============================================================
// DOM ELEMENTS
// ============================================================

const elements = {

    // --------------------------------------------------------
    // Main sensor cards
    // --------------------------------------------------------

    ph: {
        card: document.getElementById("card-ph"),
        val: document.getElementById("val-ph"),
        prog: document.getElementById("prog-ph"),
        stat: document.getElementById("stat-ph")
    },

    turbidity: {
        card: document.getElementById("card-tur"),
        val: document.getElementById("val-tur"),
        prog: document.getElementById("prog-tur"),
        stat: document.getElementById("stat-tur")
    },

    dissolvedOxygen: {
        card: document.getElementById("card-do"),
        val: document.getElementById("val-do"),
        prog: document.getElementById("prog-do"),
        stat: document.getElementById("stat-do")
    },

    temperature: {
        card: document.getElementById("card-temp"),
        val: document.getElementById("val-temp"),
        prog: document.getElementById("prog-temp"),
        stat: document.getElementById("stat-temp")
    },

    conductance: {
        card: document.getElementById("card-conductance"),
        val: document.getElementById("val-conductance"),
        prog: document.getElementById("prog-conductance"),
        stat: document.getElementById("stat-conductance")
    },


    // --------------------------------------------------------
    // AI Assessment
    // --------------------------------------------------------

    aiCard: document.getElementById("ai-card"),
    aiPrediction: document.getElementById("ai-prediction"),
    aiDetail: document.getElementById("ai-detail"),


    // --------------------------------------------------------
    // Risk Forecast
    // --------------------------------------------------------

    simulationTime: document.getElementById("simulation-time"),

    risk6h: document.getElementById("risk-6h"),
    risk6hLabel: document.getElementById("risk-6h-label"),

    risk12h: document.getElementById("risk-12h"),
    risk12hLabel: document.getElementById("risk-12h-label"),

    risk24h: document.getElementById("risk-24h"),
    risk24hLabel: document.getElementById("risk-24h-label"),

    riskHorizon: document.getElementById("risk-horizon"),

    eventStatus: document.getElementById("event-status"),


    // --------------------------------------------------------
    // Water Quality Status / Trends
    // --------------------------------------------------------

    trendTurbidity: document.getElementById("trend-turbidity"),
    trendDO: document.getElementById("trend-do"),
    trendConductance: document.getElementById("trend-conductance"),
    trendPH: document.getElementById("trend-ph"),


    // --------------------------------------------------------
    // Pump Control
    // --------------------------------------------------------

    modeToggle: document.getElementById("mode-toggle"),
    modeText: document.querySelector(".mode-text"),

    pumpCard: document.getElementById("pump-card"),
    pumpState: document.getElementById("pump-state"),
    manualPumpBtn: document.getElementById("manual-pump-btn"),


    // --------------------------------------------------------
    // Header
    // --------------------------------------------------------

    systemStatusText:
        document.getElementById("system-status-text"),

    statusIndicator:
        document.querySelector(".status-indicator"),


    // --------------------------------------------------------
    // Existing simulation controls
    // --------------------------------------------------------

    previousHour:
        document.getElementById("previous-hour"),

    nextHour:
        document.getElementById("next-hour"),

    resetSimulation:
        document.getElementById("reset-simulation")
};


// ============================================================
// UTILITY FUNCTIONS
// ============================================================


// ------------------------------------------------------------
// Check whether a value is valid
// ------------------------------------------------------------

function isValidNumber(value) {

    return (
        value !== null &&
        value !== undefined &&
        value !== "" &&
        Number.isFinite(Number(value))
    );
}


// ------------------------------------------------------------
// Format timestamp
// ------------------------------------------------------------

function formatTimestamp(timestamp) {

    if (!timestamp) {
        return "--";
    }

    const date = new Date(timestamp);

    if (Number.isNaN(date.getTime())) {
        return "--";
    }

    return (
        date.toLocaleString(
            "en-IN",
            {
                timeZone: "UTC",
                day: "2-digit",
                month: "short",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit",
                hour12: false
            }
        ) + " UTC"
    );
}


// ------------------------------------------------------------
// Format probability
// ------------------------------------------------------------

function probabilityPercent(value) {

    if (!isValidNumber(value)) {
        return "--";
    }

    return `${(Number(value) * 100).toFixed(2)}%`;
}


// ------------------------------------------------------------
// Clamp number between 0 and 100
// ------------------------------------------------------------

function clamp(value, min, max) {

    return Math.max(
        min,
        Math.min(max, value)
    );
}


// ============================================================
// SENSOR CLASSIFICATION
// ============================================================


// ------------------------------------------------------------
// pH
// ------------------------------------------------------------

function classifyPH(value) {

    if (!isValidNumber(value)) {

        return {
            text: "Unavailable",
            state: "warning"
        };
    }

    value = Number(value);

    if (value < 6.5) {

        return {
            text: "Acidic",
            state: "warning"
        };
    }

    if (value > 8.5) {

        return {
            text: "Alkaline",
            state: "danger"
        };
    }

    return {
        text: "Optimal",
        state: "safe"
    };
}


// ------------------------------------------------------------
// Turbidity
// ------------------------------------------------------------

function classifyTurbidity(value) {

    if (!isValidNumber(value)) {

        return {
            text: "Unavailable",
            state: "warning"
        };
    }

    value = Number(value);

    // Research event threshold
    if (value >= 319.81) {

        return {
            text: "High-Risk Event",
            state: "danger"
        };
    }

    if (value >= 100) {

        return {
            text: "Elevated",
            state: "warning"
        };
    }

    return {
        text: "Normal",
        state: "safe"
    };
}


// ------------------------------------------------------------
// Dissolved Oxygen
// ------------------------------------------------------------

function classifyDO(value) {

    if (!isValidNumber(value)) {

        return {
            text: "Unavailable",
            state: "warning"
        };
    }

    value = Number(value);

    if (value < 5) {

        return {
            text: "Low Oxygen",
            state: "danger"
        };
    }

    if (value < 7) {

        return {
            text: "Reduced",
            state: "warning"
        };
    }

    return {
        text: "Healthy",
        state: "safe"
    };
}


// ------------------------------------------------------------
// Water temperature
// ------------------------------------------------------------

function classifyTemperature(value) {

    if (!isValidNumber(value)) {

        return {
            text: "Unavailable",
            state: "warning"
        };
    }

    return {
        text: "Measured",
        state: "safe"
    };
}


// ------------------------------------------------------------
// Conductance
// ------------------------------------------------------------

function classifyConductance(value) {

    if (!isValidNumber(value)) {

        return {
            text: "Unavailable",
            state: "warning"
        };
    }

    return {
        text: "Measured",
        state: "safe"
    };
}


// ============================================================
// UPDATE SENSOR CARD
// ============================================================

function updateSensorCard(
    sensor,
    value,
    decimals,
    progress,
    classification
) {

    if (!sensor || !sensor.val) {
        return;
    }


    // --------------------------------------------------------
    // Value
    // --------------------------------------------------------

    if (isValidNumber(value)) {

        sensor.val.textContent =
            Number(value).toFixed(decimals);

    } else {

        sensor.val.textContent = "--";
    }


    // --------------------------------------------------------
    // Value state
    // --------------------------------------------------------

    sensor.val.className =
        `metric-value ${classification.state}`;


    // --------------------------------------------------------
    // Progress bar
    // --------------------------------------------------------

    if (sensor.prog) {

        sensor.prog.style.width =
            `${clamp(progress, 0, 100)}%`;

        sensor.prog.className =
            `progress-fill ${classification.state}`;
    }


    // --------------------------------------------------------
    // Status
    // --------------------------------------------------------

    if (sensor.stat) {

        sensor.stat.textContent =
            classification.text;

        sensor.stat.className =
            `metric-status ${classification.state}`;
    }


    // --------------------------------------------------------
    // Card high-risk border
    // --------------------------------------------------------

    if (sensor.card) {

        sensor.card.classList.remove(
            "high-risk"
        );

        if (classification.state === "danger") {

            sensor.card.classList.add(
                "high-risk"
            );
        }
    }
}


// ============================================================
// UPDATE SENSOR DATA
// ============================================================

function updateSensors(row) {

    // --------------------------------------------------------
    // pH
    // --------------------------------------------------------

    const phClass =
        classifyPH(row.ph);

    updateSensorCard(
        elements.ph,
        row.ph,
        2,
        isValidNumber(row.ph)
            ? (Number(row.ph) / 14) * 100
            : 0,
        phClass
    );


    // --------------------------------------------------------
    // Turbidity
    // --------------------------------------------------------

    const turbidityClass =
        classifyTurbidity(row.turbidity);

    updateSensorCard(
        elements.turbidity,
        row.turbidity,
        1,
        isValidNumber(row.turbidity)
            ? (Number(row.turbidity) / 500) * 100
            : 0,
        turbidityClass
    );


    // --------------------------------------------------------
    // Dissolved Oxygen
    // --------------------------------------------------------

    const doClass =
        classifyDO(row.do_concentration);

    updateSensorCard(
        elements.dissolvedOxygen,
        row.do_concentration,
        2,
        isValidNumber(row.do_concentration)
            ? (Number(row.do_concentration) / 15) * 100
            : 0,
        doClass
    );


    // --------------------------------------------------------
    // Water Temperature
    // --------------------------------------------------------

    const temperatureClass =
        classifyTemperature(
            row.water_temperature
        );

    updateSensorCard(
        elements.temperature,
        row.water_temperature,
        2,
        isValidNumber(row.water_temperature)
            ? (Number(row.water_temperature) / 20) * 100
            : 0,
        temperatureClass
    );


    // --------------------------------------------------------
    // Specific Conductance
    // --------------------------------------------------------

    const conductanceClass =
        classifyConductance(
            row.sp_conductance
        );

    updateSensorCard(
        elements.conductance,
        row.sp_conductance,
        4,
        isValidNumber(row.sp_conductance)
            ? Number(row.sp_conductance) * 100
            : 0,
        conductanceClass
    );
}


// ============================================================
// AI ASSESSMENT
// ============================================================

function updateAI(row) {

    if (
        !elements.aiCard ||
        !elements.aiPrediction ||
        !elements.aiDetail
    ) {
        return;
    }


    const risk12 =
        Number(row.risk_probability_12h || 0);


    // --------------------------------------------------------
    // Reset AI card classes
    // --------------------------------------------------------

    elements.aiCard.classList.remove(
        "warning",
        "danger"
    );


    // --------------------------------------------------------
    // HIGH RISK
    // --------------------------------------------------------

    if (
        row.risk_level === "HIGH" ||
        risk12 >= 0.50
    ) {

        elements.aiCard.classList.add(
            "danger"
        );

        elements.aiPrediction.textContent =
            "HIGH WATER-QUALITY RISK";

        elements.aiPrediction.className =
            "prediction-text";

        elements.aiDetail.textContent =
            `12h deterioration risk: ` +
            `${probabilityPercent(risk12)}. ` +
            `Preventive pump action recommended.`;

        return;
    }


    // --------------------------------------------------------
    // PREVENTIVE WARNING
    // --------------------------------------------------------

    if (
        risk12 >= 0.25 ||
        row.predictive_pump_action === true
    ) {

        elements.aiCard.classList.add(
            "warning"
        );

        elements.aiPrediction.textContent =
            "PREVENTIVE WARNING";

        elements.aiPrediction.className =
            "prediction-text";

        elements.aiDetail.textContent =
            `12h AI risk: ` +
            `${probabilityPercent(risk12)}. ` +
            `Elevated risk detected before the event.`;

        return;
    }


    // --------------------------------------------------------
    // LOW RISK
    // --------------------------------------------------------

    elements.aiPrediction.textContent =
        "LOW WATER-QUALITY RISK";

    elements.aiPrediction.className =
        "prediction-text";

    elements.aiDetail.textContent =
        `12h AI risk: ` +
        `${probabilityPercent(risk12)}. ` +
        `No actionable deterioration warning.`;
}


// ============================================================
// RISK FORECAST
// ============================================================

function classifyRiskProbability(value) {

    if (!isValidNumber(value)) {
        return "safe";
    }

    value = Number(value);

    if (value >= 0.50) {
        return "danger";
    }

    if (value >= 0.25) {
        return "warning";
    }

    return "safe";
}


// ------------------------------------------------------------
// Update individual risk card
// ------------------------------------------------------------

function updateRiskValue(
    valueElement,
    labelElement,
    value
) {

    if (!valueElement) {
        return;
    }


    const stateClass =
        classifyRiskProbability(value);


    valueElement.textContent =
        probabilityPercent(value);


    valueElement.className =
        `metric-value ${stateClass}`;


    if (labelElement) {

        if (!isValidNumber(value)) {

            labelElement.textContent =
                "Unavailable";

        } else if (Number(value) >= 0.50) {

            labelElement.textContent =
                "High Risk";

        } else if (Number(value) >= 0.25) {

            labelElement.textContent =
                "Actionable";

        } else {

            labelElement.textContent =
                "Low";
        }


        labelElement.className =
            `metric-status ${stateClass}`;
    }
}


// ------------------------------------------------------------
// Update risk forecast panel
// ------------------------------------------------------------

function updateRiskPanel(row) {

    if (!row) {
        return;
    }


    // --------------------------------------------------------
    // Simulation time
    // --------------------------------------------------------

    if (elements.simulationTime) {

        elements.simulationTime.textContent =
            formatTimestamp(row.timestamp);
    }


    // --------------------------------------------------------
    // 6-hour risk
    // --------------------------------------------------------

    updateRiskValue(
        elements.risk6h,
        elements.risk6hLabel,
        row.risk_probability_6h
    );


    // --------------------------------------------------------
    // 12-hour risk
    // --------------------------------------------------------

    updateRiskValue(
        elements.risk12h,
        elements.risk12hLabel,
        row.risk_probability_12h
    );


    // --------------------------------------------------------
    // 24-hour risk
    // --------------------------------------------------------

    updateRiskValue(
        elements.risk24h,
        elements.risk24hLabel,
        row.risk_probability_24h
    );


    // --------------------------------------------------------
    // Risk horizon
    // --------------------------------------------------------

    if (elements.riskHorizon) {

        elements.riskHorizon.textContent =
            row.predicted_risk_horizon || "None";

        const risk12 =
            Number(row.risk_probability_12h || 0);

        elements.riskHorizon.className =
            `metric-value ${
                classifyRiskProbability(risk12)
            }`;
    }


    // --------------------------------------------------------
    // Event status
    // --------------------------------------------------------

    if (elements.eventStatus) {

        if (row.actual_high_turbidity === true) {

            elements.eventStatus.textContent =
                "HIGH TURBIDITY EVENT";

            elements.eventStatus.className =
                "metric-value danger";

        } else if (
            row.predictive_pump_action === true
        ) {

            elements.eventStatus.textContent =
                "PREVENTIVE WARNING";

            elements.eventStatus.className =
                "metric-value warning";

        } else {

            elements.eventStatus.textContent =
                "NORMAL";

            elements.eventStatus.className =
                "metric-value safe";
        }
    }
}


// ============================================================
// TREND ANALYSIS
// ============================================================


// ------------------------------------------------------------
// Get row from 6 hours earlier
// ------------------------------------------------------------

function getPreviousSixHourRow() {

    const previousIndex =
        state.currentIndex - 6;

    if (previousIndex < 0) {
        return null;
    }

    return state.data[previousIndex] || null;
}


// ------------------------------------------------------------
// Calculate change
// ------------------------------------------------------------

function calculateChange(
    currentValue,
    previousValue
) {

    if (
        !isValidNumber(currentValue) ||
        !isValidNumber(previousValue)
    ) {
        return null;
    }

    return Number(currentValue) -
        Number(previousValue);
}


// ------------------------------------------------------------
// Format trend
// ------------------------------------------------------------

function formatTrend(
    currentValue,
    previousValue,
    decimals,
    unit
) {

    const change =
        calculateChange(
            currentValue,
            previousValue
        );


    if (change === null) {

        return {
            text: "--",
            state: "safe"
        };
    }


    let arrow = "→";

    if (change > 0) {
        arrow = "↑";
    } else if (change < 0) {
        arrow = "↓";
    }


    const sign =
        change > 0
            ? "+"
            : "";


    return {
        text:
            `${arrow} ${sign}${change.toFixed(decimals)} ${unit}`,

        state:
            change === 0
                ? "safe"
                : "warning"
    };
}


// ------------------------------------------------------------
// Update trend element
// ------------------------------------------------------------

function updateTrendElement(
    element,
    currentValue,
    previousValue,
    decimals,
    unit
) {

    if (!element) {
        return;
    }


    const trend =
        formatTrend(
            currentValue,
            previousValue,
            decimals,
            unit
        );


    element.textContent =
        trend.text;


    element.className =
        trend.state;
}


// ------------------------------------------------------------
// Update Water Quality Status
// ------------------------------------------------------------

function updateTrends() {

    const current =
        getCurrentRow();

    const previous =
        getPreviousSixHourRow();


    if (!current || !previous) {

        if (elements.trendTurbidity) {
            elements.trendTurbidity.textContent =
                "--";
        }

        if (elements.trendDO) {
            elements.trendDO.textContent =
                "--";
        }

        if (elements.trendConductance) {
            elements.trendConductance.textContent =
                "--";
        }

        if (elements.trendPH) {
            elements.trendPH.textContent =
                "--";
        }

        return;
    }


    // --------------------------------------------------------
    // Turbidity
    // --------------------------------------------------------

    updateTrendElement(
        elements.trendTurbidity,
        current.turbidity,
        previous.turbidity,
        1,
        "FNU"
    );


    // --------------------------------------------------------
    // Dissolved Oxygen
    // --------------------------------------------------------

    updateTrendElement(
        elements.trendDO,
        current.do_concentration,
        previous.do_concentration,
        2,
        "mg/L"
    );


    // --------------------------------------------------------
    // Conductance
    // --------------------------------------------------------

    updateTrendElement(
        elements.trendConductance,
        current.sp_conductance,
        previous.sp_conductance,
        4,
        "S/cm"
    );


    // --------------------------------------------------------
    // pH
    // --------------------------------------------------------

    updateTrendElement(
        elements.trendPH,
        current.ph,
        previous.ph,
        2,
        "pH"
    );
}


// ============================================================
// PUMP CONTROL
// ============================================================

function updatePump(row) {

    if (
        !elements.pumpCard ||
        !elements.pumpState ||
        !elements.manualPumpBtn
    ) {
        return;
    }


    // --------------------------------------------------------
    // Manual mode
    // --------------------------------------------------------

    if (state.controlMode === "manual") {

        elements.manualPumpBtn.disabled = false;

        return;
    }


    // --------------------------------------------------------
    // Automatic mode
    // --------------------------------------------------------

    const shouldActivate =
        row.predictive_pump_action === true;


    state.pumpActive =
        shouldActivate;


    if (shouldActivate) {

        elements.pumpCard.classList.add(
            "active"
        );

        elements.pumpState.textContent =
            "PREVENTIVE HOLD";

        elements.pumpState.className =
            "pump-state active";

        elements.manualPumpBtn.textContent =
            "Override";

    } else {

        elements.pumpCard.classList.remove(
            "active"
        );

        elements.pumpState.textContent =
            "RUNNING / NORMAL";

        elements.pumpState.className =
            "pump-state";

        elements.manualPumpBtn.textContent =
            "Override";
    }


    elements.manualPumpBtn.disabled =
        true;
}


// ============================================================
// HEADER STATUS
// ============================================================

function updateHeader(row) {

    if (!elements.systemStatusText) {
        return;
    }


    if (row.actual_high_turbidity === true) {

        elements.systemStatusText.textContent =
            "High-Risk Event Detected";


        if (elements.statusIndicator) {

            elements.statusIndicator.style.background =
                "var(--red)";

            elements.statusIndicator.style.boxShadow =
                "0 0 8px var(--red)";
        }

        return;
    }


    if (row.predictive_pump_action === true) {

        elements.systemStatusText.textContent =
            "AI Preventive Warning";


        if (elements.statusIndicator) {

            elements.statusIndicator.style.background =
                "var(--yellow)";

            elements.statusIndicator.style.boxShadow =
                "0 0 8px var(--yellow)";
        }

        return;
    }


    elements.systemStatusText.textContent =
        "ML Simulation Connected";


    if (elements.statusIndicator) {

        elements.statusIndicator.style.background =
            "var(--cyan)";

        elements.statusIndicator.style.boxShadow =
            "0 0 8px var(--cyan)";
    }
}


// ============================================================
// RENDER CURRENT ROW
// ============================================================

function renderCurrentRow() {

    const row =
        getCurrentRow();


    if (!row) {
        return;
    }


    // Main sensor cards
    updateSensors(row);


    // AI assessment
    updateAI(row);


    // Risk forecast
    updateRiskPanel(row);


    // Parameter trends
    updateTrends();


    // Pump control
    updatePump(row);


    // Header
    updateHeader(row);


    console.log(
        "AquaSense Simulation",
        {
            timestamp: row.timestamp,
            turbidity: row.turbidity,
            ph: row.ph,
            dissolvedOxygen: row.do_concentration,
            temperature: row.water_temperature,
            conductance: row.sp_conductance,
            risk6h: row.risk_probability_6h,
            risk12h: row.risk_probability_12h,
            risk24h: row.risk_probability_24h,
            riskHorizon: row.predicted_risk_horizon,
            pumpAction: row.predictive_pump_action
        }
    );
}


// ============================================================
// GET CURRENT ROW
// ============================================================

function getCurrentRow() {

    if (
        !state.data ||
        state.data.length === 0
    ) {
        return null;
    }


    if (
        state.currentIndex < 0 ||
        state.currentIndex >= state.data.length
    ) {
        return null;
    }


    return state.data[
        state.currentIndex
    ];
}


// ============================================================
// NEXT HOUR
// ============================================================

function nextHour() {

    if (!state.loaded) {
        return;
    }


    if (
        state.currentIndex <
        state.data.length - 1
    ) {

        state.currentIndex++;

        renderCurrentRow();

    } else {

        alert(
            "The historical simulation has reached the end of the dataset."
        );
    }
}


// ============================================================
// PREVIOUS HOUR
// ============================================================

function previousHour() {

    if (!state.loaded) {
        return;
    }


    if (state.currentIndex > 0) {

        state.currentIndex--;

        renderCurrentRow();
    }
}


// ============================================================
// RESET SIMULATION
// ============================================================

function resetSimulation() {

    if (!state.loaded) {
        return;
    }


    state.currentIndex =
        START_INDEX;


    renderCurrentRow();
}

// ============================================================
// MOVE SIMULATION BUTTONS TO HEADER
// ============================================================

function moveSimulationButtonsToHeader() {

    const header =
        document.querySelector(".app-header");

    const simulationCard =
        document.getElementById("simulation-controls");

    if (!header || !simulationCard) {
        return;
    }

    const buttons =
        simulationCard.querySelector(
            ".simulation-buttons"
        );

    if (!buttons) {
        return;
    }

    // Move the existing buttons into the header.
    // This does NOT create duplicate buttons.
    header.appendChild(buttons);
}


// ============================================================
// SIMULATION CONTROLS
// ============================================================

function setupSimulationControls() {

    // --------------------------------------------------------
    // Previous
    // --------------------------------------------------------

    if (elements.previousHour) {

        elements.previousHour.addEventListener(
            "click",
            previousHour
        );
    }


    // --------------------------------------------------------
    // Next
    // --------------------------------------------------------

    if (elements.nextHour) {

        elements.nextHour.addEventListener(
            "click",
            nextHour
        );
    }


    // --------------------------------------------------------
    // Reset
    // --------------------------------------------------------

    if (elements.resetSimulation) {

        elements.resetSimulation.addEventListener(
            "click",
            resetSimulation
        );
    }
}


// ============================================================
// CONTROL MODE
// ============================================================

function setupControlMode() {

    if (!elements.modeToggle) {
        return;
    }


    // --------------------------------------------------------
    // Initial state
    // --------------------------------------------------------

    state.controlMode =
        "auto";


    if (elements.modeText) {

        elements.modeText.textContent =
            "Auto";
    }


    if (elements.manualPumpBtn) {

        elements.manualPumpBtn.disabled =
            true;
    }


    // --------------------------------------------------------
    // Toggle Auto / Manual
    // --------------------------------------------------------

    elements.modeToggle.addEventListener(
        "click",
        () => {

            elements.modeToggle.classList.toggle(
                "manual"
            );


            const manual =
                elements.modeToggle.classList.contains(
                    "manual"
                );


            if (manual) {

                state.controlMode =
                    "manual";


                if (elements.modeText) {

                    elements.modeText.textContent =
                        "Manual";
                }


                if (elements.manualPumpBtn) {

                    elements.manualPumpBtn.disabled =
                        false;

                    elements.manualPumpBtn.textContent =
                        state.pumpActive
                            ? "Stop Pump"
                            : "Start Pump";
                }


            } else {

                state.controlMode =
                    "auto";


                if (elements.modeText) {

                    elements.modeText.textContent =
                        "Auto";
                }


                if (elements.manualPumpBtn) {

                    elements.manualPumpBtn.disabled =
                        true;
                }


                renderCurrentRow();
            }
        }
    );


    // --------------------------------------------------------
    // Manual pump button
    // --------------------------------------------------------

    if (elements.manualPumpBtn) {

        elements.manualPumpBtn.addEventListener(
            "click",
            () => {

                if (
                    state.controlMode !==
                    "manual"
                ) {
                    return;
                }


                state.pumpActive =
                    !state.pumpActive;


                if (state.pumpActive) {

                    elements.pumpCard.classList.add(
                        "active"
                    );

                    elements.pumpState.textContent =
                        "MANUAL HOLD";

                    elements.pumpState.className =
                        "pump-state active";

                    elements.manualPumpBtn.textContent =
                        "Stop Pump";

                } else {

                    elements.pumpCard.classList.remove(
                        "active"
                    );

                    elements.pumpState.textContent =
                        "MANUAL STOP";

                    elements.pumpState.className =
                        "pump-state danger";

                    elements.manualPumpBtn.textContent =
                        "Start Pump";
                }
            }
        );
    }
}


// ============================================================
// LOAD SIMULATION DATA
// ============================================================

async function loadSimulationData() {

    try {

        console.log(
            "Connecting to AquaSense simulation API..."
        );


        const response =
            await fetch(
                `${API_BASE}/simulation/data`
            );


        if (!response.ok) {

            throw new Error(
                `HTTP error ${response.status}`
            );
        }


        const data =
            await response.json();


        if (
            !Array.isArray(data) ||
            data.length === 0
        ) {

            throw new Error(
                "Simulation API returned no data."
            );
        }


        state.data =
            data;


        state.loaded =
            true;


        // Make sure starting index exists
        state.currentIndex =
            Math.min(
                START_INDEX,
                state.data.length - 1
            );


        console.log(
            `Loaded ${state.data.length} simulation records.`
        );


        renderCurrentRow();


    } catch (error) {

        console.error(
            "Failed to load simulation data:",
            error
        );


        showConnectionError();
    }
}


// ============================================================
// CONNECTION ERROR
// ============================================================

function showConnectionError() {

    if (elements.aiPrediction) {

        elements.aiPrediction.textContent =
            "SIMULATION OFFLINE";

        elements.aiPrediction.className =
            "prediction-text";
    }


    if (elements.aiDetail) {

        elements.aiDetail.textContent =
            "Unable to connect to AquaSense simulation API.";
    }


    if (elements.pumpState) {

        elements.pumpState.textContent =
            "API OFFLINE";

        elements.pumpState.className =
            "pump-state danger";
    }


    if (elements.systemStatusText) {

        elements.systemStatusText.textContent =
            "Simulation Offline";
    }


    if (elements.statusIndicator) {

        elements.statusIndicator.style.background =
            "var(--red)";

        elements.statusIndicator.style.boxShadow =
            "0 0 8px var(--red)";
    }


    state.loaded =
        false;
}


// ============================================================
// INITIALIZE
// ============================================================

async function initialize() {

    console.log(
        "Starting AquaSense ML simulation..."
    );


    // IMPORTANT:
    // We DO NOT create any cards here.
    //
    // index.html already contains:
    // - Previous button
    // - Next Hour button
    // - Reset button
    // - pH card
    // - Turbidity card
    // - Dissolved Oxygen card
    // - Water Temperature card
    // - Conductance card
    // - Risk Forecast
    // - Water Quality Status
    // - Pump Control

    moveSimulationButtonsToHeader();

    setupSimulationControls();


    setupControlMode();

    await loadSimulationData();
}


// ============================================================
// START APPLICATION
// ============================================================

initialize();