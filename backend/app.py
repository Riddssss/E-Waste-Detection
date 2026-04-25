"""
E-Waste Detector — Flask Backend API
=====================================
Endpoints:
  GET  /health                    → health check
  POST /predict                   → classify image + nearest disposal site
  GET  /sites/<device_type>       → list all disposal sites for a device type
"""
 
import os
import math
import io
import base64
import logging
from pathlib import Path
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
MODEL_PATH  = Path("model.pth")
CLASS_NAMES = ["AC", "Laptop", "Mobile"]   # must match training ImageFolder sorted order
IMG_SIZE    = 224
DEFAULT_LAT = 19.0981   # Juhu, Mumbai
DEFAULT_LON = 72.8265
 
DISPOSAL_SITES = {
    "Laptop": [
        {"name": "Laptop Center - Fort",    "lat": 18.9318, "lon": 72.8330},
        {"name": "Laptop Center - Andheri", "lat": 19.1180, "lon": 72.8467},
        {"name": "Laptop Center - Kurla",   "lat": 19.0721, "lon": 72.8795},
    ],
    "Mobile": [
        {"name": "Mobile Center - Dadar",   "lat": 19.0176, "lon": 72.8562},
        {"name": "Mobile Center - Chembur", "lat": 19.0621, "lon": 72.9006},
        {"name": "Mobile Center - Powai",   "lat": 19.1187, "lon": 72.9060},
    ],
    "AC": [
        {"name": "AC Center - Ghatkopar",   "lat": 19.0850, "lon": 72.9080},
        {"name": "AC Center - Sion",        "lat": 19.0370, "lon": 72.8644},
        {"name": "AC Center - Kandivali",   "lat": 19.2056, "lon": 72.8530},
    ],
}
 
# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
 
 
def load_model(path: Path, class_names: list) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run train.py first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(len(class_names))
 
    try:
        state = torch.load(str(path), map_location=device)
    except Exception:
        state = torch.load(str(path), weights_only=False, map_location=device)
 
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
 
    # strip DataParallel prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    logger.info(f"Model loaded from {path} ({len(class_names)} classes)")
    return model, device
 
 
device_global = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_global, device_global = load_model(MODEL_PATH, CLASS_NAMES)
 
# ─────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
 
 
def preprocess(img: Image.Image) -> torch.Tensor:
    return _transform(img.convert("RGB")).unsqueeze(0).to(device_global)
 
# ─────────────────────────────────────────────
#  Haversine distance
# ─────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * (2 * math.asin(min(1.0, math.sqrt(a))))
 
 
def nearest_site(device_type: str, user_lat: float, user_lon: float) -> dict:
    sites = DISPOSAL_SITES.get(device_type, [])
    if not sites:
        return {}
    best = min(sites, key=lambda s: haversine_km(user_lat, user_lon, s["lat"], s["lon"]))
    return {**best, "distance_km": round(haversine_km(user_lat, user_lon, best["lat"], best["lon"]), 2)}
 
# ─────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────
def predict(img: Image.Image, user_lat: float, user_lon: float) -> dict:
    x = preprocess(img)
    with torch.no_grad():
        out   = model_global(x)
        probs = F.softmax(out[0], dim=0).cpu().numpy()
 
    pred_idx    = int(probs.argmax())
    confidence  = float(probs[pred_idx])
    device_type = CLASS_NAMES[pred_idx] if confidence >= 0.4 else "Uncertain"
 
    result = {
        "device":     device_type,
        "confidence": round(confidence * 100, 2),
        "all_probs":  {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))},
    }
 
    site = nearest_site(device_type, user_lat, user_lon)
    if site:
        result["nearest_site"] = site
 
    return result
 
# ─────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
 
 
@app.get("/health")
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES})
 
 
@app.post("/predict")
def predict_endpoint():
    """
    Accepts multipart/form-data with:
      - file: image file
      - lat:  float (optional, default Juhu)
      - lon:  float (optional, default Juhu)
 
    OR JSON body with:
      - image_b64: base64-encoded image string
      - lat / lon
    """
    try:
        user_lat = float(request.form.get("lat", DEFAULT_LAT) or request.json.get("lat", DEFAULT_LAT) if request.is_json else request.form.get("lat", DEFAULT_LAT))
        user_lon = float(request.form.get("lon", DEFAULT_LON) or request.json.get("lon", DEFAULT_LON) if request.is_json else request.form.get("lon", DEFAULT_LON))
    except Exception:
        user_lat, user_lon = DEFAULT_LAT, DEFAULT_LON
 
    # image from file upload
    if "file" in request.files:
        file = request.files["file"]
        img  = Image.open(file.stream)
 
    # image from base64 JSON
    elif request.is_json:
        body = request.get_json(force=True)
        b64  = body.get("image_b64", "")
        lat  = float(body.get("lat", DEFAULT_LAT))
        lon  = float(body.get("lon", DEFAULT_LON))
        if not b64:
            return jsonify({"error": "No image provided. Send 'file' or 'image_b64'."}), 400
        img_bytes = base64.b64decode(b64)
        img       = Image.open(io.BytesIO(img_bytes))
        user_lat, user_lon = lat, lon
 
    else:
        return jsonify({"error": "No image provided. Send multipart 'file' or JSON 'image_b64'."}), 400
 
    try:
        result = predict(img, user_lat, user_lon)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500
 
 
@app.get("/sites/<device_type>")
def get_sites(device_type: str):
    sites = DISPOSAL_SITES.get(device_type.capitalize())
    if sites is None:
        return jsonify({"error": f"Unknown device type '{device_type}'. Valid: {list(DISPOSAL_SITES.keys())}"}), 404
    return jsonify({"device": device_type.capitalize(), "sites": sites})
 
 
# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)