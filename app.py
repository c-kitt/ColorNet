from flask import Flask, request, jsonify, send_from_directory
from model import color_net
import os

app = Flask(__name__, static_folder="static", static_url_path="")
model = color_net.load("color_net.npz")

@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    r = int(data.get("r", 0))
    g = int(data.get("g", 0))
    b = int(data.get("b", 0))
    p = model.predict_proba([r, g, b])
    label = "white" if p >= 0.5 else "black"
    return jsonify({"Prediction": label, "P (White)": float(round(p, 4))})

if __name__ == "__main__":
    app.run(debug=True)
