from flask import Flask, jsonify
import pickle
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)

# --------------------------------------------------
# Load ML Model
# --------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# --------------------------------------------------
# Firebase Setup
# --------------------------------------------------
cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-irrigation-c801b-default-rtdb.asia-southeast1.firebasedatabase.app"
})

# --------------------------------------------------
# Home Route
# --------------------------------------------------
@app.route("/")
def home():
    return "Smart Irrigation API Running"

# --------------------------------------------------
# Process Latest Firebase Reading
#
# Reads latest timestamp node from:
# Live_readings/plants/plant_1/
#
# Predicts:
# Healthy / Needs Water / Problem
#
# Writes back:
# prediction/class
# prediction/label
#
# Updates:
# Live_readings/pump_control/state
#
# NOTE:
# ESP handles:
# ON for 10 sec → OFF automatically
# --------------------------------------------------
@app.route("/process", methods=["GET"])
def process():
    try:
        # ------------------------------------------
        # Firebase Plant Path
        # ------------------------------------------
        plant_path = "Live_readings/plants/plant_1"
        ref = db.reference(plant_path)

        all_data = ref.get()

        if not all_data:
            return jsonify({
                "error": "No plant data found in Firebase"
            }), 404

        # ------------------------------------------
        # Get Latest Timestamp Node
        # Example:
        # 23042026104532
        # ------------------------------------------
        latest_key = sorted(all_data.keys())[-1]
        latest_data = all_data[latest_key]

        # ------------------------------------------
        # Extract Firebase Values
        # Structure:
        #
        # leaf_readings/leaf_1/
        # aggregated/
        # ------------------------------------------
        leaf_data = latest_data["leaf_readings"]["leaf_1"]
        aggregated = latest_data["aggregated"]

        soil_moisture = leaf_data["soil_moisture"]
        R = leaf_data["R"]
        G = leaf_data["G"]
        B = leaf_data["B"]

        temperature = aggregated["temperature"]
        humidity = aggregated["humidity"]
        days = aggregated["days_since_last_watering"]
        times = aggregated["no_of_times_watered"]

        # ------------------------------------------
        # Feature Engineering
        # MUST match training exactly
        # ------------------------------------------
        G_minus_R = G - R
        G_minus_B = G - B
        stress_index = abs(G - R) / (G + R)
        green_ratio = G / (R + B)

        # ------------------------------------------
        # Create DataFrame
        # ------------------------------------------
        features = pd.DataFrame([{
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "days_since_last_watering": days,
            "no_of_times_watered": times,
            "R": R,
            "G": G,
            "B": B,
            "G_minus_R": G_minus_R,
            "G_minus_B": G_minus_B,
            "stress_index": stress_index,
            "green_ratio": green_ratio
        }])

        # ------------------------------------------
        # Prediction
        # ------------------------------------------
        prediction = int(model.predict(features)[0])

        label_map = {
            0: "Healthy",
            1: "Needs Water",
            2: "Problem"
        }

        label = label_map[prediction]

        # ------------------------------------------
        # Write Prediction Back
        #
        # Live_readings/plants/plant_1/
        # latest_timestamp/prediction/
        # ------------------------------------------
        prediction_ref = db.reference(
            f"{plant_path}/{latest_key}/prediction"
        )

        prediction_ref.set({
            "class": prediction,
            "label": label
        })

        # ------------------------------------------
        # Update Pump State
        #
        # If Needs Water → ON
        # Else → OFF
        #
        # ESP will auto turn OFF after 10 sec
        # ------------------------------------------
        pump_state = "ON" if prediction == 1 else "OFF"

        pump_ref = db.reference(
            "Live_readings/pump_control"
        )

        pump_ref.update({
            "state": pump_state,
            "last_updated_at": latest_key
        })

        # ------------------------------------------
        # Final Response
        # ------------------------------------------
        return jsonify({
            "status": "success",
            "latest_timestamp": latest_key,
            "prediction": prediction,
            "label": label,
            "pump_state": pump_state
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
