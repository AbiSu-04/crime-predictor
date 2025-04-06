from flask import Flask, request, jsonify 
import pickle
import numpy as np
import os
from auth import check_auth

app = Flask(__name__)

# Get the absolute path to the model_training directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'model_training')

# Load saved components using correct paths
with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "crime_mapping.pkl"), "rb") as f:
    crime_mapping = pickle.load(f)

@app.route("/predict", methods=["POST"])
@check_auth
def predict():
    data = request.json
    if "Area Name Encoded" not in data:
        return jsonify({"error": "Missing Area Name Encoded"}), 400

    input_features = np.array([[data["Area Name Encoded"]]])
    input_scaled = scaler.transform(input_features)

    prediction = model.predict(input_scaled)[0]

    crime_name = crime_mapping.get(prediction, "Unknown")

    return jsonify({
        "predicted_crime_code": int(prediction),
        "predicted_crime": crime_name
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"})

if __name__ == "__main__":
    app.run(debug=True)
