from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_lstm_rul.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")

# Load model and scaler
print("Loading predictive maintenance system...")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load dataset
data = pd.read_csv(DATA_PATH)

# Sensor columns
sensor_columns = [
    col for col in data.columns
    if col.startswith("sensor_") or col.startswith("virtual_sensor_")
]

print("System loaded successfully!")
print("Dataset shape:", data.shape)
print("Sensor features:", len(sensor_columns))

@app.route("/health")
def health():
    return "OK", 200
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    status = None
    aircraft_id = None
    observations = None
    error = None

    if request.method == "POST":

        try:
            aircraft_id = int(request.form["aircraft_id"])

            # Select aircraft
            aircraft_data = data[data["A_2"] == aircraft_id]

            if len(aircraft_data) < 30:
                error = "This aircraft does not have enough observations."

            else:
                observations = len(aircraft_data)

                # Get latest 30 observations
                latest_data = aircraft_data[
                    sensor_columns
                ].tail(30)

                # Convert to numpy
                sensor_data = latest_data.values

                # Normalize
                sensor_scaled = scaler.transform(sensor_data)

                # Reshape for CNN-LSTM
                model_input = sensor_scaled.reshape(1, 30, 28)

                # Predict
                result = model.predict(
                    model_input,
                    verbose=0
                )

                prediction = float(result[0][0])

                # Maintenance status
                if prediction <= 10:
                    status = "HIGH PRIORITY - Maintenance Recommended"
                elif prediction <= 30:
                    status = "MEDIUM PRIORITY - Monitor Closely"
                else:
                    status = "LOW PRIORITY - Normal Operation"

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        status=status,
        aircraft_id=aircraft_id,
        observations=observations,
        error=error
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )