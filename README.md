# 🔴 Mars MEDA Atmospheric Pressure Forecasting
### NASA Perseverance Rover · Sensor Recovery · Kaggle Competition

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Polars](https://img.shields.io/badge/Polars-LazyFrame-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat-square&logo=kaggle)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---

## 🌌 Project Overview

The **Mars Environmental Dynamics Analyser (MEDA)** is a weather station onboard NASA's **Perseverance Rover**, currently operating on the surface of Mars. MEDA collects continuous telemetry — temperature, wind speed, humidity, and critically, **atmospheric pressure** — to help scientists understand the Martian climate and plan future human missions.

In the real world, sensors fail. When a sensor dropout occurs on Mars, there is no technician to fix it. This project solves that problem:

> **Goal:** Recover missing atmospheric PRESSURE readings from the MEDA sensor array by training a machine learning model on the surrounding healthy sensor data — effectively creating a *virtual sensor* that can predict pressure when the real one goes offline.

This was built for the **Kaggle MEDA Virtual Sensor Recovery Competition**, working with over **8 million rows** of real NASA telemetry data.

---

## 🧠 The Problem in Plain English

Imagine a weather station on Mars that records dozens of sensor readings every second. One day, the pressure sensor goes blank — but all the other sensors (temperature, rover status, solar time, etc.) keep working fine. 

Can we use those 35 healthy sensors to *predict* what the missing pressure sensor *would have read*?

That is exactly what this model does.

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — MEDA Virtual Sensor Recovery](https://www.kaggle.com/competitions/mars-environmental-dynamics-analyzer-meda-virtual-sensor-recovery) |
| Format | `.parquet` (train + test) |
| Scale | 8,000,000+ rows |
| Target Variable | `PRESSURE` (atmospheric pressure in Pa) |
| Time Reference | `SCLK` — NASA Spacecraft Clock timestamp |
| Martian Time | `LMST` — Local Mean Solar Time (text format: `Sol:HH:MM:SS`) |

> ⚠️ **Note:** Due to Kaggle's data license, the raw dataset is not included in this repo. To reproduce results, download it directly from the competition page and place it in `/kaggle/input/competitions/mars-environmental-dynamics-analyzer-meda-virtual-sensor-recovery/`.

---

## 🔬 Technical Approach

### Phase 1 — Exploratory Data Analysis (EDA)

Before touching the model, a structured EDA was performed on the 8M-row dataset using **Polars LazyFrames** — which read metadata without loading the full data into RAM, making the pipeline memory-safe.

Key findings:

- **Timeline check (SCLK):** Train and test sets share the same time window — meaning we are filling *holes in the middle* of a known timeline, not predicting an unknown future. This unlocked forward-fill and backward-fill imputation as valid strategies.
- **Missing data audit:** Two sensors (`ROVER_HGA_OFF`, `SKYCAM_OFF`) were missing **99%+ of all rows** — effectively dead columns. Four other sensors had ~1.5% dropout.
- **LMST format:** The Martian local time column was stored as raw text (e.g. `00234:14:32:07`) — not usable by any ML algorithm in that form.

---

### Phase 2 — Data Cleaning

```python
clean_train_lazy = (
    train_lazy
    .drop(["ROVER_HGA_OFF", "SKYCAM_OFF"])   # Drop 99%-empty dead sensors
    .sort("SCLK")                             # Chronological order
    .fill_null(strategy="forward")            # Forward-fill sensor dropouts
    .fill_null(strategy="backward")           # Backward-fill edge cases
)
```

- **Dead columns dropped:** `ROVER_HGA_OFF`, `SKYCAM_OFF` — 99% null, feeding these to a model would inject pure noise
- **Imputation strategy:** Forward-fill then backward-fill on remaining nulls — valid because a rover parked at 1:00 PM is statistically still parked at 1:01 PM

---

### Phase 3 — Feature Engineering: Bending Time into a Circle

The `LMST` column contains the **Martian solar time** as a string. Two transformations were needed:

**Step 1 — Parse the NASA time format:**
```python
# LMST format: "00234:14:32:07" → Sol=234, Hour=14.535...
Sol    = LMST[0:5]          # Martian day number
LMST_Decimal = Hour + Min/60 + Sec/3600
```

**Step 2 — Cyclical encoding with trigonometry:**

Time is circular — 23:59 wraps back to 00:00. If we feed raw decimal hours to a model, it thinks 23.9 and 0.1 are far apart, when they are actually 12 minutes apart. The fix:

```python
LMST_sin = sin(LMST_Decimal × 2π / 24)
LMST_cos = cos(LMST_Decimal × 2π / 24)
```

This is a classical technique in **astrostatistics** — mapping time onto the unit circle so the model understands the repeating rhythm of a planetary day.

---

### Phase 4 — Model Training

```python
model = xgb.XGBRegressor(
    n_estimators  = 100,
    learning_rate = 0.1,
    tree_method   = "hist",    # Optimised for 8M rows — histogram-based splits
    random_state  = 42
)
model.fit(X_train, y_train)
```

| Parameter | Value | Why |
|---|---|---|
| Algorithm | XGBoost Regressor | Handles tabular sensor data extremely well |
| Tree method | `hist` | Fast approximate splits — essential at 8M row scale |
| Features | 35 engineered columns | All sensors + Sin/Cos time + Sol number |
| Target | `PRESSURE` | Continuous atmospheric pressure readings |
| Evaluation | MSE on training set | Competition metric |

---

## 📊 Pipeline Summary

```
Raw Parquet (8M rows)
        │
        ▼
  Polars LazyFrame EDA
  ├── SCLK timeline check
  ├── Null count per sensor
  └── PRESSURE health check
        │
        ▼
  Data Cleaning
  ├── Drop ROVER_HGA_OFF, SKYCAM_OFF (99% null)
  ├── Sort by SCLK (chronological)
  ├── Forward-fill → Backward-fill nulls
  └── Apply to both train and test
        │
        ▼
  Feature Engineering
  ├── Parse LMST → Sol + LMST_Decimal
  ├── Cyclical encoding → LMST_sin, LMST_cos
  └── Drop raw string columns (LMST, LTST)
        │
        ▼
  XGBoost Regressor
  ├── 35 features × 8M rows
  ├── hist tree method
  └── MSE evaluation
        │
        ▼
  submission.csv → Kaggle submission
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Polars** | Memory-efficient LazyFrame processing of 8M+ rows |
| **XGBoost** | Gradient boosted regression with hist method |
| **Pandas / NumPy** | Final DataFrame handling and array operations |
| **Scikit-learn** | MSE evaluation metric |
| **Python `math`** | Trigonometric cyclical feature encoding |

---

## 🔭 Real-World Relevance

This project has direct parallels to challenges in **commercial space operations**:

- **In-orbit telemetry gaps** — satellites and space stations experience sensor dropouts just like MEDA. The same forward-fill + ML recovery pipeline applies directly.
- **Planetary mission planning** — accurate atmospheric pressure forecasting on Mars is critical for entry, descent, and landing (EDL) profile design.
- **Cyclical feature engineering** — the sin/cos time encoding used here is directly applicable to orbital period modelling, eclipse cycle prediction, and any time-series problem with periodic structure.

---

## 👩‍💻 Author

**Mubeena Hussain**  
MSc Statistics 
📧 mubeenahussain1205@gmail.com  
🔗 [LinkedIn](www.linkedin.com/in/mubeena-hussain-a357b920b)  


---

## 📄 License

This project is open source under the [MIT License](LICENSE).  
Dataset belongs to NASA / Kaggle competition organizers — please refer to the competition page for data usage terms.

---

*"Understanding Mars today is how we survive Mars tomorrow."*
