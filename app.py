from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Smart Irrigation API Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract inputs
        soil_moisture = data['soil_moisture']
        temperature = data['temperature']
        humidity = data['humidity']
        days = data['days_since_last_watering']
        times = data['no_of_times_watered']
        R = data['R']
        G = data['G']
        B = data['B']

        # Feature engineering (MATCH TRAINING EXACTLY)
        G_minus_R = G - R
        G_minus_B = G - B
        stress_index = abs(G - R) / (G + R)
        green_ratio = G / (R + B)

        # Create DataFrame with EXACT feature names
        features = pd.DataFrame([{
            'soil_moisture': soil_moisture,
            'temperature': temperature,
            'humidity': humidity,
            'days_since_last_watering': days,
            'no_of_times_watered': times,
            'R': R,
            'G': G,
            'B': B,
            'G_minus_R': G_minus_R,
            'G_minus_B': G_minus_B,
            'stress_index': stress_index,
            'green_ratio': green_ratio
        }])

        # Prediction
        prediction = model.predict(features)[0]

        label_map = {
            0: "Healthy",
            1: "Needs Water",
            2: "Problem"
        }

        return jsonify({
            "prediction": int(prediction),
            "label": label_map[prediction]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)