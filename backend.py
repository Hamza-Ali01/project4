"""
backend.py
----------
Single-file Flask API backend for Urdu OCR.

Contains everything needed to serve predictions:
  - CRNN model definition
  - Vocabulary (char <-> index) loader
  - Image preprocessing
  - CTC greedy decoder
  - Flask API with one endpoint: POST /predict

"best_model.pth" is downloaded automatically at startup from a GitHub
Release asset if it isn't already present locally. "vocab.json" must
still be committed to the repo (it's small).

Run locally:
    python backend.py

Run in production (e.g. Render):
    gunicorn backend:app

The server starts at http://localhost:5000
Frontend (a separate index.html) calls POST http://localhost:5000/predict
with the image as multipart/form-data, and gets back JSON: {"text": "..."}
"""

import json
import os

import numpy as np
import requests
import torch
import torch.nn as nn

# Keep torch's thread pool small - reduces memory overhead on low-RAM hosts
# like Render's free tier. Must be set before any model/tensor work happens.
torch.set_num_threads(1)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# Config
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "best_model.pth")
VOCAB_PATH = os.path.join(APP_ROOT, "vocab.json")

# Direct download URL for the GitHub Release asset containing best_model.pth
# Replace this with your actual release asset URL if it changes.
MODEL_URL = "https://github.com/Hamza-Ali01/project4/releases/download/v1.0-model/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_model_if_missing():
    """Download best_model.pth from GitHub Releases if it isn't already on disk."""
    if os.path.exists(MODEL_PATH):
        return
    print("Model not found locally, downloading from GitHub Releases...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model download complete.")


# Vocabulary (char <-> index mapping, index 0 = CTC blank)
class Vocab:
    def __init__(self, chars):
        self.chars = chars
        self.char_to_idx = {ch: idx + 1 for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: ch for idx, ch in enumerate(self.chars)}
        self.idx_to_char[0] = "<blank>"

    @property
    def size(self):
        return len(self.chars) + 1

    def decode_indices(self, indices):
        return "".join(self.idx_to_char.get(i, "") for i in indices)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            chars = json.load(f)
        return cls(chars)


def ctc_greedy_decode(pred_indices, vocab):
    collapsed = []
    prev = None
    for idx in pred_indices:
        idx = int(idx)
        if idx != prev:
            if idx != 0:
                collapsed.append(idx)
        prev = idx
    return vocab.decode_indices(collapsed)


# CRNN model (CNN + BiLSTM + CTC-ready output)
class CRNN(nn.Module):
    def __init__(self, img_height=32, num_channels=1, num_classes=58, rnn_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(512, rnn_hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        logits = self.fc(rnn_out)
        log_probs = torch.log_softmax(logits, dim=2).permute(1, 0, 2)
        return log_probs

# Preprocessing (must match what train.py used)
def preprocess_image(pil_image, img_height=32, max_width=512):
    img = pil_image.convert("L")
    w, h = img.size
    new_w = max(1, int(w * (img_height / h)))
    img = img.resize((new_w, img_height), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    if new_w < max_width:
        pad = np.ones((img_height, max_width - new_w), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    else:
        arr = arr[:, :max_width]

    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
    return tensor


# Load model + vocab once at startup
vocab = None
model = None
img_height = 32
max_width = 512


def load_model_and_vocab():
    global vocab, model, img_height, max_width
    vocab = Vocab.load(VOCAB_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    img_height = checkpoint.get("img_height", 32)
    max_width = checkpoint.get("max_width", 512)
    model = CRNN(
        img_height=img_height,
        num_channels=1,
        num_classes=checkpoint.get("num_classes", vocab.size),
        rnn_hidden=checkpoint.get("rnn_hidden", 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()


# Flask API
app = Flask(__name__)
CORS(app)  # allow the separate frontend HTML file to call this API from any origin

# --- Load the model at import time so this works both with `python backend.py`
# --- AND with `gunicorn backend:app` (gunicorn never runs the __main__ block below).
try:
    download_model_if_missing()
    if not os.path.exists(VOCAB_PATH):
        print(f"ERROR: could not find '{VOCAB_PATH}'. Make sure vocab.json is committed to the repo.")
    else:
        load_model_and_vocab()
        print(f"Model loaded (vocab size: {vocab.size}).")
except Exception as e:
    print(f"ERROR during model startup: {e}")


@app.route("/")
def index():
    # index.html sits at the repo root next to backend.py, not in a
    # templates/ folder, so serve it directly instead of using render_template.
    return send_from_directory(APP_ROOT, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vocab is None:
        return jsonify({"error": "Model is not loaded on the server. Check server logs."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send it as form field 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        pil_image = Image.open(file.stream)
        tensor = preprocess_image(pil_image, img_height, max_width)
        tensor = tensor.unsqueeze(0).to(device)  # add batch dim

        with torch.no_grad():
            log_probs = model(tensor)
            preds = log_probs.argmax(dim=2).squeeze(1).cpu().tolist()

        text = ctc_greedy_decode(preds, vocab)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    # Model is already loaded above at import time; just start the dev server.
    print("Starting API at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
