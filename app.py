from flask import Flask, jsonify
import pickle
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)

# ==================================================
# Load ML Model
# ==================================================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ==================================================
# Firebase Setup
# ==================================================
cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-irrigation-c801b-default-rtdb.asia-southeast1.firebasedatabase.app"
})

# ==================================================
# Home Route
# ==================================================
@app.route("/")
def home():
    return "Smart Irrigation API Running"


# ==================================================
# Process Latest Plant Reading
#
# Reads:
# Live_readings/plants/plant_1/latest_timestamp
#
# Updates:
# prediction/
# pump_control/
# plant_status/
#
# Logic:
# Model predicts:
#   0 = Healthy
#   1 = Needs Water
#
# Flask decides:
#   2 = Problem
#
# based on failed_recovery_count
# ==================================================
@app.route("/process", methods=["GET"])
def process():
    try:

        # ==========================================
        # Firebase Plant Path
        # ==========================================
        plant_path = "Live_readings/plants/plant_1"
        plant_ref = db.reference(plant_path)

        all_data = plant_ref.get()

        if not all_data:
            return jsonify({
                "error": "No plant data found"
            }), 404

        # ==========================================
        # Get Latest Timestamp Node
        # ==========================================
        latest_key = sorted(all_data.keys())[-1]
        latest_data = all_data[latest_key]

        # ==========================================
        # Read Aggregated Values
        # (IMPORTANT: now using aggregated values)
        # ==========================================
        aggregated = latest_data["aggregated"]

        soil_moisture = aggregated["soil_misture"] if "soil_misture" in aggregated else aggregated["soil_moisture"]
        temperature = aggregated["temperature"]
        humidity = aggregated["humidity"]
        days = aggregated["days_since_last_watering"]
        times = aggregated["no_of_times_watered"]

        R = aggregated["R"]
        G = aggregated["G"]
        B = aggregated["B"]

        # ==========================================
        # Feature Engineering
        # Must match training exactly
        # ==========================================
        G_minus_R = G - R
        G_minus_B = G - B
        stress_index = abs(G - R) / (G + R)
        green_ratio = G / (R + B)

        # ==========================================
        # Create DataFrame
        # ==========================================
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

        # ==========================================
        # Model Prediction
        # ONLY:
        # 0 = Healthy
        # 1 = Needs Water
        # ==========================================
        model_prediction = int(model.predict(features)[0])

        model_label_map = {
            0: "Healthy",
            1: "Needs Water"
        }

        model_label = model_label_map.get(
            model_prediction,
            "Needs Water"
        )

        # ==========================================
        # Read plant_status
        # ==========================================
        status_path = "plant_status/plant_1"
        status_ref = db.reference(status_path)

        plant_status = status_ref.get()

        if not plant_status:
            plant_status = {
                "failed_recovery_count": 0,
                "last_stress_index": stress_index,
                "last_health_state": "Healthy",
                "last_improved_at": latest_key
            }

        failed_recovery_count = plant_status.get(
            "failed_recovery_count", 0
        )

        last_stress_index = plant_status.get(
            "last_stress_index", stress_index
        )

        # ==========================================
        # Flask Logic for FINAL Prediction
        #
        # if stress not improving
        # for 3+ watering cycles
        #
        # → Problem
        # ==========================================
        final_prediction = model_prediction
        final_label = model_label

        if model_prediction == 0:
            # Healthy → Reset everything
            failed_recovery_count = 0
            final_prediction = 0
            final_label = "Healthy"

        else:
            # Needs Water → compare stress

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

        # ==========================================
        # Write Prediction Back
        # ==========================================
        prediction_ref = db.reference(
            f"{plant_path}/{latest_key}/prediction"
        )

        prediction_ref.set({
            "model_prediction": model_prediction,
            "model_label": model_label,
            "final_prediction": final_prediction,
            "final_label": final_label
        })

        # ==========================================
        # Update plant_status
        # ==========================================
        status_ref.update({
            "failed_recovery_count": failed_recovery_count,
            "last_stress_index": stress_index,
            "last_health_state": final_label,
            "last_improved_at": latest_key if final_prediction == 0 else plant_status.get("last_improved_at", latest_key)
        })

        # ==========================================
        # Pump Control
        #
        # Needs Water → ON
        # Healthy → OFF
        # Problem → OFF
        #
        # ESP handles:
        # ON → 10 sec → OFF
        # ==========================================
        if final_prediction == 1:
            pump_state = "ON"
        else:
            pump_state = "OFF"

        pump_ref = db.reference(
            "Live_readings/pump_control"
        )

        pump_ref.update({
            "state": pump_state,
            "last_updated_at": latest_key
        })

        # ==========================================
        # Final Response
        # ==========================================
        return jsonify({
            "status": "success",
            "latest_timestamp": latest_key,

            "model_prediction": model_prediction,
            "model_label": model_label,

            "final_prediction": final_prediction,
            "final_label": final_label,

            "failed_recovery_count": failed_recovery_count,
            "pump_state": pump_state
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
