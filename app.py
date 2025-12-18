from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

MODEL_PATH = "iris_model.pkl"

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Iris Classification Flask App Running"}

@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet"}, 400

    data = request.json.get("features")
    model = joblib.load(MODEL_PATH)

    features = np.array(data).reshape(1, -1)
    prediction = int(model.predict(features)[0])

    return jsonify({"prediction": prediction})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
