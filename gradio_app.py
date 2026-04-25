"""
E-Waste Detector — Gradio UI
==============================
Detects device type (Laptop / Mobile / AC) from an uploaded image
and finds the nearest e-waste disposal site based on user location.
 
Run: python gradio_app.py
Opens at: http://127.0.0.1:7860
"""
 
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import gradio as gr
 
# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
MODEL_PATH  = "model.pth"
CLASS_NAMES = ["AC", "Laptop", "Mobile"]   # must match training folder sort order
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
def build_model(num_classes: int) -> nn.Module:
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
 
 
def load_model():
    model = build_model(len(CLASS_NAMES))
    if not os.path.exists(MODEL_PATH):
        print(f"[warn] '{MODEL_PATH}' not found. Run train.py first. Using untrained weights.")
        model.to(device).eval()
        return model
    try:
        state = torch.load(MODEL_PATH, map_location=device)
    except Exception:
        state = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"[info] Model loaded from {MODEL_PATH}")
    return model
 
 
model = load_model()
 
# ─────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
 
# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * (2 * math.asin(min(1.0, math.sqrt(a))))
 
 
# ─────────────────────────────────────────────
#  Prediction function
# ─────────────────────────────────────────────
def detect_and_find(img, user_lat: float, user_lon: float):
    if img is None:
        return "⚠️ Please upload an image."
 
    pil = Image.fromarray(img).convert("RGB") if not isinstance(img, Image.Image) else img.convert("RGB")
    x   = _transform(pil).unsqueeze(0).to(device)
 
    with torch.no_grad():
        out   = model(x)
        probs = F.softmax(out[0], dim=0).cpu().numpy()
 
    pred_idx    = int(probs.argmax())
    confidence  = float(probs[pred_idx])
    device_type = CLASS_NAMES[pred_idx] if confidence >= 0.4 else "Uncertain"
 
    result = {
        "🔍 Detected Device":   device_type,
        "📊 Confidence (%)":    f"{confidence * 100:.2f}",
        "📈 All Probabilities": {CLASS_NAMES[i]: f"{probs[i]*100:.2f}%" for i in range(len(CLASS_NAMES))},
    }
 
    if device_type in DISPOSAL_SITES:
        sites   = DISPOSAL_SITES[device_type]
        nearest = min(sites, key=lambda s: haversine_km(user_lat, user_lon, s["lat"], s["lon"]))
        dist    = haversine_km(user_lat, user_lon, nearest["lat"], nearest["lon"])
        result["📍 Nearest Disposal Site"] = nearest["name"]
        result["📏 Distance (km)"]         = f"{dist:.2f}"
    else:
        result["📍 Nearest Disposal Site"] = "N/A (low confidence)"
 
    return result
 
# ─────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────
with gr.Blocks(title="E-Waste Detector") as demo:
    gr.Markdown(
        """
        # ♻️ E-Waste Disposal Finder
        Upload a photo of an electronic device (Laptop, Mobile, or AC).
        The model will classify it and find the **nearest disposal site** based on your location.
        """
    )
 
    with gr.Row():
        with gr.Column(scale=1):
            img_input  = gr.Image(type="numpy", label="Upload Device Image")
            lat_input  = gr.Number(label="Your Latitude",  value=DEFAULT_LAT)
            lon_input  = gr.Number(label="Your Longitude", value=DEFAULT_LON)
            submit_btn = gr.Button("🔍 Detect & Find Site", variant="primary")
 
        with gr.Column(scale=1):
            output = gr.JSON(label="Result")
 
    submit_btn.click(
        fn=detect_and_find,
        inputs=[img_input, lat_input, lon_input],
        outputs=output,
    )
 
    gr.Examples(
        examples=[
            [None, 19.0981, 72.8265],
        ],
        inputs=[img_input, lat_input, lon_input],
    )
 
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)