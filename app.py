import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "artifacts/student_performance_model.joblib"

FEATURES = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course"
]

model = None


def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run src/train_model.py first."
        )

    model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "student-performance-prediction"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 503

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    missing = [column for column in FEATURES if column not in data]

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "fields": missing
        }), 400

    try:
        student = pd.DataFrame([{
            column: data[column] for column in FEATURES
        }])

        prediction = model.predict(student)[0]

        return jsonify({"prediction": str(prediction)})

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
