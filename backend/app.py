from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import sys


# Add the parent directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from auth import check_auth

# Semantic search
from model_training.semantic_search import load_and_preprocess_data, generate_embeddings, search_similar_events
import torch

app = Flask(__name__)

# === Model loading ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'model_training')

with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "crime_mapping.pkl"), "rb") as f:
    crime_mapping = pickle.load(f)

# === Semantic search setup ===
print("🔍 Loading data and generating semantic embeddings...")
search_df = load_and_preprocess_data()
search_embeddings = generate_embeddings(search_df)

# === Routes ===

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


@app.route("/search", methods=["POST"])
@check_auth
def search():
    data = request.json
    if "query" not in data:
        return jsonify({"error": "Missing query in request"}), 400

    query = data["query"]
    top_k = data.get("top_k", 5)

    try:
        results = search_similar_events(query, search_df, search_embeddings, top_k=top_k)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"})


if __name__ == "__main__":
    app.run(debug=True)
