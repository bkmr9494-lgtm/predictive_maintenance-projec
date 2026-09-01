from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_lstm_rul.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "processed_data.csv"
)

print("Loading predictive maintenance system...")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load scaler
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded successfully!")

# Load dataset
data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)

# Identify sensor columns
sensor_columns = [
    col for col in data.columns
    if col.startswith("sensor_") or col.startswith("virtual_sensor_")
]

print("Number of sensor features:", len(sensor_columns))
print("System loaded successfully!")


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
                error = (
                    "This aircraft does not have enough observations "
                    "for prediction."
                )

            else:
                observations = len(aircraft_data)

                # Take latest 30 observations
                latest_data = aircraft_data[
                    sensor_columns
                ].tail(30)

                # Convert to NumPy
                sensor_data = latest_data.to_numpy()

                # Normalize sensor values
                sensor_scaled = scaler.transform(sensor_data)

                # CNN-LSTM input shape
                model_input = sensor_scaled.reshape(1, 30, 28)

                # Make prediction
                result = model.predict(
                    model_input,
                    verbose=0
                )

                prediction = float(result[0][0])

                # Maintenance classification
                if prediction <= 10:
                    status = (
                        "HIGH PRIORITY - "
                        "Maintenance Recommended"
                    )

                elif prediction <= 30:
                    status = (
                        "MEDIUM PRIORITY - "
                        "Monitor Closely"
                    )

                else:
                    status = (
                        "LOW PRIORITY - "
                        "Normal Operation"
                    )

        except ValueError:
            error = "Please enter a valid aircraft ID."

        except Exception as e:
            print("Prediction error:", e)
            error = "Unable to generate prediction. Please try again."

    return render_template(
        "index.html",
        prediction=prediction,
        status=status,
        aircraft_id=aircraft_id,
        observations=observations,
        error=error
    )


if __name__ == "__main__":

    # Render provides the PORT environment variable.
    # Locally it defaults to 5000.
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )