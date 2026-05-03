from flask import Flask, jsonify
import pickle
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import os
import json
from datetime import datetime

app = Flask(__name__)

# ==================================================
# Load ML Model
# ==================================================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ==================================================
# Firebase Setup using Render Environment Variable
# ==================================================
firebase_json = os.environ.get("FIREBASE_CREDENTIALS")

if not firebase_json:
    raise ValueError("FIREBASE_CREDENTIALS environment variable not found")

cred_dict = json.loads(firebase_json)
cred = credentials.Certificate(cred_dict)

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-irrigation-c801b-default-rtdb.asia-southeast1.firebasedatabase.app"
})

# ==================================================
# Helper Function
# ==================================================
def calculate_days_since_last_watering(last_watered_at):
    try:
        last_watered_date = datetime.strptime(
            last_watered_at,
            "%d%m%Y%H%M%S"
        )
        current_date = datetime.now()
        difference = current_date - last_watered_date
        return difference.days
    except:
        return 0


@app.route("/")
def home():
    return "Smart Irrigation API Running"


@app.route("/process", methods=["GET"])
def process():
    try:

        plant_path = "Live_readings/plants/plant_1"
        plant_ref = db.reference(plant_path)

        all_data = plant_ref.get()

        if not all_data:
            return jsonify({"error": "No plant data found"}), 404

        # ==================================================
        # ✅ FIXED: Proper latest timestamp selection
        # ==================================================
        valid_items = []

        for k, v in all_data.items():
            if k.isdigit() and isinstance(v, dict) and "aggregated" in v:
                try:
                    dt = datetime.strptime(k, "%d%m%Y%H%M%S")
                    valid_items.append((dt, k))
                except:
                    continue

        if not valid_items:
            return jsonify({"error": "No valid data with aggregated"}), 400

        latest_key = sorted(valid_items)[-1][1]
        latest_data = all_data[latest_key]

        # ==================================================
        # Plant Status
        # ==================================================
        status_path = "Live_readings/plant_status/plant_1"
        status_ref = db.reference(status_path)

        plant_status = status_ref.get()

        if not plant_status:
            plant_status = {
                "no_of_times_watered": 0,
                "last_watered_at": latest_key,
                "failed_recovery_count": 0,
                "last_stress_index": 0.0,
                "last_health_state": "Healthy",
                "last_improved_at": latest_key
            }

        no_of_times_watered = plant_status.get("no_of_times_watered", 0)
        last_watered_at = plant_status.get("last_watered_at", latest_key)
        failed_recovery_count = plant_status.get("failed_recovery_count", 0)
        last_stress_index = plant_status.get("last_stress_index", 0.0)

        days_since_last_watering = calculate_days_since_last_watering(
            last_watered_at
        )

        # ==================================================
        # ✅ FIXED: Safe aggregated access
        # ==================================================
        aggregated = latest_data.get("aggregated")

        if not aggregated:
            return jsonify({"error": "Aggregated data missing"}), 400

        soil_moisture = aggregated.get("soil_moisture", 0)
        temperature = aggregated.get("temperature", 0)
        humidity = aggregated.get("humidity", 0)

        R = aggregated.get("R", 0)
        G = aggregated.get("G", 0)
        B = aggregated.get("B", 0)

        # ==================================================
        # Feature Engineering
        # ==================================================
        G_minus_R = G - R
        G_minus_B = G - B
        stress_index = abs(G - R) / (G + R) if (G + R) != 0 else 0
        green_ratio = G / (R + B) if (R + B) != 0 else 0

        aggregated_ref = db.reference(
            f"{plant_path}/{latest_key}/aggregated"
        )

        aggregated_ref.update({
            "days_since_last_watering": days_since_last_watering
        })

        features = pd.DataFrame([{
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "humidity": humidity,
            "days_since_last_watering": days_since_last_watering,
            "no_of_times_watered": no_of_times_watered,
            "R": R,
            "G": G,
            "B": B,
            "G_minus_R": G_minus_R,
            "G_minus_B": G_minus_B,
            "stress_index": stress_index,
            "green_ratio": green_ratio
        }])

        model_prediction = int(model.predict(features)[0])

        model_label_map = {
            0: "Healthy",
            1: "Needs Water"
        }

        model_label = model_label_map.get(
            model_prediction,
            "Needs Water"
        )

        final_prediction = model_prediction
        final_label = model_label

        if model_prediction == 0:
            failed_recovery_count = 0
            final_prediction = 0
            final_label = "Healthy"
        else:
            if stress_index >= last_stress_index:
                failed_recovery_count += 1
            else:
                failed_recovery_count = 0

            if failed_recovery_count >= 3:
                final_prediction = 2
                final_label = "Problem"
            else:
                final_prediction = 1
                final_label = "Needs Water"

        prediction_ref = db.reference(
            f"{plant_path}/{latest_key}/prediction"
        )

        prediction_ref.set({
            "model_prediction": model_prediction,
            "model_label": model_label,
            "final_prediction": final_prediction,
            "final_label": final_label
        })

        status_ref.update({
            "failed_recovery_count": failed_recovery_count,
            "last_stress_index": stress_index,
            "last_health_state": final_label,
            "last_improved_at": latest_key if final_prediction == 0
            else plant_status.get("last_improved_at", latest_key)
        })

        pump_state = "ON" if final_prediction == 1 else "OFF"

        pump_ref = db.reference("Live_readings/pump_control")

        pump_ref.update({
            "state": pump_state,
            "last_updated_at": latest_key
        })

        return jsonify({
            "status": "success",
            "latest_timestamp": latest_key,
            "model_prediction": model_prediction,
            "model_label": model_label,
            "final_prediction": final_prediction,
            "final_label": final_label,
            "days_since_last_watering": days_since_last_watering,
            "no_of_times_watered": no_of_times_watered,
            "failed_recovery_count": failed_recovery_count,
            "pump_state": pump_state
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
