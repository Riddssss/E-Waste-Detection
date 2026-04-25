# WasteWatch 🔬♻️

> **Classify. Locate. Dispose Responsibly.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black)](https://flask.palletsprojects.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.0-orange)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An intelligent e-waste classification system that identifies electronic devices from images (Laptop, Mobile, AC) using **Transfer Learning on MobileNetV2** and guides users to the **nearest certified disposal site** using Haversine distance calculation. Built as a Deep Learning Final Year Project.

---

## 📊 Key Results

| Model | Stage | Val Accuracy | Avg F1 Score |
|-------|-------|-------------|--------------|
| MobileNetV2 | Head Only (6 epochs) | — | — |
| MobileNetV2 | Fine-tuned (10 epochs) | — | — |

> Fill in after running `train.py` on your dataset.

---

## 🎯 Problem Statement

E-waste is one of the fastest-growing waste streams globally. Most consumers are unaware of where and how to dispose of electronic devices responsibly. This project proposes an intelligent system that uses **computer vision** to automatically classify e-waste devices from photos and provides real-time guidance to the **nearest certified disposal facility** based on the user's GPS coordinates.

---

## 🏗️ System Architecture

```
User Uploads Image
        ↓
  Image Preprocessing
  (Resize 224×224, Normalize)
        ↓
  MobileNetV2 Backbone
  (ImageNet Pretrained)
        ↓
  Custom Classifier Head
  (3 classes: AC / Laptop / Mobile)
        ↓
  Softmax → Device Type + Confidence
        ↓
  Haversine Distance Calculation
  (User GPS ↔ Disposal Sites)
        ↓
  Nearest Certified Disposal Site
```

---

## ✨ Features

### Core ML
- 🧠 **MobileNetV2 Transfer Learning** — lightweight backbone pretrained on ImageNet
- 🔄 **Two-Stage Training** — freeze backbone → train head, then full fine-tuning
- ⚖️ **Weighted Random Sampler** — handles class imbalance automatically
- 📐 **Strong Augmentation** — random crop, flip, rotation, color jitter

### Application Features
- 📸 **Image Classification** — detects Laptop, Mobile, and AC units from photos
- 📍 **Nearest Site Finder** — Haversine formula calculates real-world distances
- 🌐 **Flask REST API** — file upload or base64 JSON input supported
- ⚡ **Gradio UI** — upload image + enter coordinates → instant result

---

## 🗂️ Project Structure

```
waste-detection/
│
├── backend/
│   ├── app.py              ← Flask REST API
│   ├── train.py            ← Two-stage MobileNetV2 training pipeline
│   └── requirements.txt    ← Python dependencies
│
├── gradio_app.py           ← Gradio UI (standalone, no build step)
├── README.md
└── .gitignore
```

---

## 🧠 Model Details

### Architecture

| Component | Detail |
|-----------|--------|
| Backbone | MobileNetV2 (ImageNet pretrained) |
| Classifier | `Dropout → Linear(1280 → 3)` |
| Input Size | 224 × 224 RGB |
| Output | Softmax probabilities over 3 classes |

### Training Strategy

| Stage | What's trained | LR | Epochs |
|-------|---------------|-----|--------|
| Stage 1 (Head) | Classifier only (backbone frozen) | 1e-3 | 6 |
| Stage 2 (Fine-tune) | Entire model unfrozen | 1e-4 | 10 |

### Data Augmentation

| Transform | Parameters |
|-----------|-----------|
| RandomResizedCrop | scale (0.6–1.0), size 224 |
| RandomHorizontalFlip | p=0.5 |
| RandomRotation | ±25° |
| ColorJitter | brightness/contrast/saturation ±0.3 |
| Normalize | ImageNet mean & std |

### Classes

| Class | Description |
|-------|-------------|
| AC | Air Conditioner units |
| Laptop | Laptops and notebooks |
| Mobile | Smartphones and mobile devices |

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| Classes | AC, Laptop, Mobile |
| Split | 80% Train / 20% Val |
| Balancing | WeightedRandomSampler per epoch |
| Format | ImageFolder structure |

```
dataset/
  AC/
    img1.jpg ...
  Laptop/
    img1.jpg ...
  Mobile/
    img1.jpg ...
```

---

## 📍 Disposal Sites (Mumbai)

| Device | Sites |
|--------|-------|
| Laptop | Fort, Andheri, Kurla |
| Mobile | Dadar, Chembur, Powai |
| AC | Ghatkopar, Sion, Kandivali |

Distance is computed using the **Haversine formula** over the user's live GPS coordinates.

---

## 🚀 How to Run

### Prerequisites
- Python 3.12+

### Step 1 — Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Prepare dataset
Organise your images into class subfolders:
```
dataset/
  AC/      ← all AC images
  Laptop/  ← all laptop images
  Mobile/  ← all mobile images
```

### Step 3 — Train the model
```bash
cd backend
python train.py --data_dir ../dataset
```

This saves `model.pth` in the `backend/` directory.

Optional flags:
```bash
python train.py --data_dir ../dataset \
                --epochs_head 6 \
                --epochs_finetune 10 \
                --batch_size 16
```

### Step 4 — Launch Gradio UI
```bash
python gradio_app.py
```
Open: `http://127.0.0.1:7860`

### Step 5 — (Optional) Launch Flask API
```bash
cd backend
python app.py
```
API available at: `http://127.0.0.1:5000`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + class list |
| POST | `/predict` | Classify image + nearest site |
| GET | `/sites/<device_type>` | List all disposal sites for a device |

### Example — File Upload
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@laptop.jpg" \
  -F "lat=19.0981" \
  -F "lon=72.8265"
```

### Example Response
```json
{
  "device": "Laptop",
  "confidence": 94.32,
  "all_probs": {
    "AC": 2.11,
    "Laptop": 94.32,
    "Mobile": 3.57
  },
  "nearest_site": {
    "name": "Laptop Center - Andheri",
    "lat": 19.118,
    "lon": 72.8467,
    "distance_km": 1.73
  }
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Deep Learning | PyTorch 2.2, TorchVision |
| Model | MobileNetV2 (Transfer Learning) |
| Distance | Haversine Formula (pure Python) |
| API | Flask, flask-cors |
| UI | Gradio |
| Version Control | GitHub |

---

## 👩‍💻 Authors

**Riddhima Shah**  **Avni Sethi**
BTech Project
Subject: Deep Learning

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
