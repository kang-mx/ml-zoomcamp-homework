from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib

# -------------------------
# Setup
# -------------------------
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

scaler = joblib.load("scaler.pkl")

# -------------------------
# Model
# -------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

input_dim = 16
model = LSTMRegressor(input_dim=input_dim, hidden_dim=128, dropout=0.0).to(device)
model.load_state_dict(torch.load("lstm_rul_final.pth", map_location=device))
model.eval()

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return "✅ LSTM RUL API is running!"

@app.route("/docs", methods=["GET"])
def docs():
    return """
    <h3>LSTM RUL API</h3>
    <p>POST /predict</p>
    """

@app.route("/predict", methods=["POST"])
def predict_rul():
    try:
        data = request.json
        if "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' key"}), 400

        seq = np.array(data["sequence"], dtype=np.float32)

        if seq.shape[1] != input_dim:
            return jsonify({"error": f"Expected {input_dim} features per timestep"}), 400

        seq_scaled = scaler.transform(seq)
        seq_tensor = torch.tensor(seq_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(seq_tensor).cpu().numpy()[0][0]

        return jsonify({"predicted_RUL": float(pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
