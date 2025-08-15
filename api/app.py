from flask import Flask, request, jsonify
from model import color_net
import os

app = Flask(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "color_net.npz")
model = color_net.load(MODEL_PATH)

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

@app.route("/health")
def health():
    return "ok", 200

@app.route("/predict", methods=["OPTIONS"])
def predict_options():
    return ("", 204)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    r = int(data.get("r", 0))
    g = int(data.get("g", 0))
    b = int(data.get("b", 0))
    p = model.predict_proba([r, g, b])
    label = "white" if p >= 0.5 else "black"
    return jsonify({"Prediction": label, "P (White)": float(round(p, 4))})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
