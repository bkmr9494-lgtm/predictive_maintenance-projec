import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ==========================================
# GET PROJECT ROOT DIRECTORY
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==========================================
# FILE PATHS
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn_lstm_rul.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "processed_data.csv"
)


# ==========================================
# SETTINGS
# ==========================================

AIRCRAFT_ID = 66
SEQUENCE_LENGTH = 30


# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

print("Loading predictive maintenance system...")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully!")
print("Scaler loaded successfully!")


# ==========================================
# LOAD PROCESSED DATA
# ==========================================

data = pd.read_csv(DATA_PATH)

print("Processed dataset loaded!")
print("Dataset shape:", data.shape)


# ==========================================
# SENSOR FEATURES
# ==========================================

sensor_columns = [
    column for column in data.columns
    if column.startswith("sensor_")
    or column.startswith("virtual_sensor_")
]

print("Number of sensor features:", len(sensor_columns))


# ==========================================
# SELECT AIRCRAFT
# ==========================================

aircraft_data = data[data["A_2"] == AIRCRAFT_ID].copy()

if len(aircraft_data) < SEQUENCE_LENGTH:
    raise ValueError(
        f"Aircraft {AIRCRAFT_ID} does not have enough observations."
    )

print("\nSelected Aircraft/Unit:", AIRCRAFT_ID)
print("Number of observations:", len(aircraft_data))


# ==========================================
# SELECT LATEST 30 OBSERVATIONS
# ==========================================

latest_data = aircraft_data[sensor_columns].tail(SEQUENCE_LENGTH)

print(
    f"Latest {SEQUENCE_LENGTH} sensor observations selected."
)

print("Sensor data shape:", latest_data.shape)


# ==========================================
# NORMALIZE SENSOR DATA
# ==========================================

scaled_data = scaler.transform(latest_data)

print("Normalized sensor data shape:", scaled_data.shape)


# ==========================================
# CREATE MODEL INPUT
# ==========================================

X_input = np.array(scaled_data).reshape(
    1,
    SEQUENCE_LENGTH,
    len(sensor_columns)
)

print("Model input shape:", X_input.shape)


# ==========================================
# PREDICT RUL
# ==========================================

prediction = model.predict(X_input, verbose=0)

predicted_rul = float(prediction[0][0])

# RUL cannot be negative
predicted_rul = max(0, predicted_rul)


# ==========================================
# DETERMINE MAINTENANCE STATUS
# ==========================================

if predicted_rul <= 10:
    status = "HIGH PRIORITY - Maintenance Recommended"

elif predicted_rul <= 25:
    status = "MEDIUM PRIORITY - Monitor Closely"

else:
    status = "LOW PRIORITY - Normal Operation"


# ==========================================
# DISPLAY FINAL RESULT
# ==========================================

print("\n====================================")
print("      PREDICTIVE MAINTENANCE SYSTEM")
print("====================================")

print(f"Aircraft / Unit ID : {AIRCRAFT_ID}")
print(f"Observations       : {len(aircraft_data)}")
print(f"Predicted RUL      : {predicted_rul:.2f} cycles")

print("------------------------------------")
print("Maintenance Status :", status)
print("====================================")