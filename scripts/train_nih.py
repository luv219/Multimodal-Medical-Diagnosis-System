import os
import random
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.nih_dataset import get_nih_dataloaders, compute_pos_weights_from_csv
from model.image_only_model import ImageOnlyModel
from sklearn.metrics import roc_auc_score
import numpy as np


def load_nih_config():
    """Load NIH training settings from config.yaml at project root, or use hardcoded defaults."""
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    defaults = {
        "data_dir": "./data/nih_chest_xray",
        "train_csv": "./cxr_code/data/train.csv",
        "valid_csv": "./cxr_code/data/valid.csv",
        "batch_size": 128,
        "num_epochs": 5,
        "lr": 0.0001,
        "weight_decay": 0.00001,
        "seed": 42,
        "num_classes": 5,
        "label_names": [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Effusion",
        ],
        "pos_weight": 5.0,
        "scheduler_step_size": 2,
        "scheduler_gamma": 0.1,
        "num_workers": 4,
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not config or "nih" not in config:
            return defaults
        nih = config["nih"]
        if not isinstance(nih, dict):
            return defaults
        merged = {**defaults, **nih}
        return merged
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return defaults


def resolve_project_path(rel_path):
    """Resolve a path from config (often ./...) relative to project root."""
    if os.path.isabs(rel_path):
        return rel_path
    rel = rel_path.replace("/", os.sep)
    if rel.startswith("." + os.sep):
        rel = rel[2:]
    elif rel == ".":
        rel = ""
    return os.path.normpath(os.path.join(PROJECT_ROOT, rel))


cfg = load_nih_config()

DATA_DIR = resolve_project_path(cfg["data_dir"])
TRAIN_CSV = resolve_project_path(cfg["train_csv"])
VALID_CSV = resolve_project_path(cfg["valid_csv"])

BATCH_SIZE = int(cfg["batch_size"])
NUM_EPOCHS = int(cfg["num_epochs"])
LR = float(cfg["lr"])
WEIGHT_DECAY = float(cfg["weight_decay"])
NUM_CLASSES = int(cfg["num_classes"])
LABEL_NAMES = list(cfg["label_names"])
SCHEDULER_STEP_SIZE = int(cfg["scheduler_step_size"])
SCHEDULER_GAMMA = float(cfg["scheduler_gamma"])
NUM_WORKERS = int(cfg["num_workers"])
UNIFORM_POS_WEIGHT = float(cfg["pos_weight"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = int(cfg["seed"])
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def train():
    print(f"Using device: {DEVICE}")
    print(
        f"Config: batch_size={BATCH_SIZE}, num_epochs={NUM_EPOCHS}, lr={LR}, "
        f"train_csv={TRAIN_CSV}"
    )

    per_class_weights = compute_pos_weights_from_csv(
        TRAIN_CSV, LABEL_NAMES, fallback_pos_weight=UNIFORM_POS_WEIGHT
    )
    print("Computed per-class pos_weights:")
    for name in LABEL_NAMES:
        print(f"  {name}: {per_class_weights[name]:.4f}")
    pos_weights = torch.tensor(
        [per_class_weights[name] for name in LABEL_NAMES], dtype=torch.float32
    ).to(DEVICE)

    train_loader, val_loader = get_nih_dataloaders(
        TRAIN_CSV,
        VALID_CSV,
        DATA_DIR,
        BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(
        f"Loaded {len(train_loader.dataset)} training samples and "
        f"{len(val_loader.dataset)} validation samples."
    )

    model = ImageOnlyModel(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    os.makedirs("results", exist_ok=True)
    best_val_auc = 0

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            if (i + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        scheduler.step()

        # --- Validate ---
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                        val_loss += criterion(outputs, labels).item()
                else:
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # AUC per class
        aucs = []
        for i, name in enumerate(LABEL_NAMES):
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)
            except ValueError:
                print(
                    f"Warning: Only one class present in y_true for {name}. "
                    "Setting AUC to 0.5"
                )
                aucs.append(0.5)

        mean_auc = np.mean(aucs)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Mean AUC: {mean_auc:.4f}"
        )
        for name, auc in zip(LABEL_NAMES, aucs):
            print(f"  {name}: {auc:.4f}")

        # Save best model
        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            torch.save(model.state_dict(), "results/best_model.pth")
            print(f"  ✓ Saved best model (AUC: {best_val_auc:.4f})")


if __name__ == "__main__":
    train()
