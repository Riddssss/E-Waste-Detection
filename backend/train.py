"""
E-Waste Detector — Training Pipeline
======================================
Two-stage transfer learning with MobileNetV2:
  Stage 1 → freeze backbone, train classifier head
  Stage 2 → unfreeze all, fine-tune with smaller LR
 
Usage:
  python train.py --data_dir ./dataset
"""
 
import argparse
import math
import os
import random
import shutil
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
 
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────
#  Config defaults
# ─────────────────────────────────────────────
RANDOM_SEED    = 42
BATCH_SIZE     = 16
HEAD_EPOCHS    = 6
FINETUNE_EPOCHS = 10
TRAIN_RATIO    = 0.8
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MODEL_SAVE     = Path("model.pth")
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
 
# ─────────────────────────────────────────────
#  Transforms
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
 
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
 
# ─────────────────────────────────────────────
#  Dataset helpers
# ─────────────────────────────────────────────
def stratified_split(dataset_root: Path, train_ratio: float = TRAIN_RATIO):
    """Split dataset indices per class to maintain balance."""
    base_ds = datasets.ImageFolder(str(dataset_root), transform=val_transform)
    class_to_indices = {}
    for idx, (_, cls) in enumerate(base_ds.samples):
        class_to_indices.setdefault(cls, []).append(idx)
 
    train_idx, val_idx = [], []
    for cls, idxs in class_to_indices.items():
        random.shuffle(idxs)
        split = max(1, int(len(idxs) * train_ratio))
        train_idx.extend(idxs[:split])
        val_idx.extend(idxs[split:])
 
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    logger.info(f"Split → train: {len(train_idx)}, val: {len(val_idx)}")
    return train_idx, val_idx, base_ds.classes
 
 
def build_loaders(dataset_root: Path, batch_size: int = BATCH_SIZE):
    train_idx, val_idx, classes = stratified_split(dataset_root)
 
    train_ds = datasets.ImageFolder(str(dataset_root), transform=train_transform)
    val_ds   = datasets.ImageFolder(str(dataset_root), transform=val_transform)
 
    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(val_ds,   val_idx)
 
    train_labels  = [train_ds.samples[i][1] for i in train_idx]
    counts        = np.bincount(train_labels, minlength=len(classes))
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
 
    logger.info(f"Classes: {classes}")
    logger.info(f"Train class counts: {dict(zip(classes, counts))}")
 
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,  num_workers=0)
    return classes, train_loader, val_loader
 
# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)
 
# ─────────────────────────────────────────────
#  Evaluation helpers
# ─────────────────────────────────────────────
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm
 
 
def per_class_metrics(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp+fp) != 0)
    rec  = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn) != 0)
    f1   = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(tp), where=(prec+rec) != 0)
    return prec, rec, f1
 
 
def evaluate(model, loader, num_classes):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.cpu().numpy().tolist())
    cm               = confusion_matrix(y_true, y_pred, num_classes)
    prec, rec, f1    = per_class_metrics(cm)
    acc              = float(np.diag(cm).sum()) / float(cm.sum())
    return acc, float(np.nanmean(f1)), prec, rec, f1
 
# ─────────────────────────────────────────────
#  Two-stage training
# ─────────────────────────────────────────────
def train(dataset_root: Path, model_save: Path = MODEL_SAVE):
    classes, train_loader, val_loader = build_loaders(dataset_root)
    num_classes  = len(classes)
    model        = build_model(num_classes)
    criterion    = nn.CrossEntropyLoss()
    best_f1      = -1.0
 
    # ── Stage 1: train head only ──
    logger.info("=== Stage 1: Training classifier head ===")
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
 
    for epoch in range(HEAD_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
 
        acc, avg_f1, prec, rec, f1 = evaluate(model, val_loader, num_classes)
        logger.info(f"[Head {epoch+1}/{HEAD_EPOCHS}] acc={acc:.3f} avg_f1={avg_f1:.3f}")
        for i, c in enumerate(classes):
            logger.info(f"   {c}: prec={prec[i]:.3f} rec={rec[i]:.3f} f1={f1[i]:.3f}")
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), model_save)
            logger.info(f"   → Saved model (best F1: {best_f1:.3f})")
 
    # ── Stage 2: fine-tune whole model ──
    logger.info("=== Stage 2: Fine-tuning whole model ===")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
 
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
 
        acc, avg_f1, prec, rec, f1 = evaluate(model, val_loader, num_classes)
        logger.info(f"[FT {epoch+1}/{FINETUNE_EPOCHS}] acc={acc:.3f} avg_f1={avg_f1:.3f}")
        for i, c in enumerate(classes):
            logger.info(f"   {c}: prec={prec[i]:.3f} rec={rec[i]:.3f} f1={f1[i]:.3f}")
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), model_save)
            logger.info(f"   → Saved improved model (best F1: {best_f1:.3f})")
 
    logger.info(f"Training complete. Best avg F1: {best_f1:.3f} → {model_save}")
 
# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the E-Waste device classifier.")
    parser.add_argument("--data_dir",   default="dataset",   help="Path to dataset folder (with class subfolders)")
    parser.add_argument("--model_save", default="model.pth", help="Output model path")
    parser.add_argument("--epochs_head",     type=int, default=HEAD_EPOCHS)
    parser.add_argument("--epochs_finetune", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--batch_size",      type=int, default=BATCH_SIZE)
    args = parser.parse_args()
 
    HEAD_EPOCHS     = args.epochs_head
    FINETUNE_EPOCHS = args.epochs_finetune
    BATCH_SIZE      = args.batch_size
 
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
 
    train(Path(args.data_dir), Path(args.model_save))
    logger.info("Done! Start the API with: python app.py")
 