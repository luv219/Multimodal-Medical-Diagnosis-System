import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nih_dataset import get_nih_dataloaders
from model.image_only_model import ImageOnlyModel
from sklearn.metrics import roc_auc_score
import numpy as np

# Config
DATA_DIR = "./data/nih_chest_xray"
TRAIN_CSV = "./cxr_code/data/train.csv"
VALID_CSV = "./cxr_code/data/valid.csv"

BATCH_SIZE = 128
NUM_EPOCHS = 5
LR = 1e-4
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]

def train():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader = get_nih_dataloaders(TRAIN_CSV, VALID_CSV, DATA_DIR, BATCH_SIZE, num_workers=4)
    print(f"Loaded {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")
    
    model = ImageOnlyModel(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Use weighted BCE to handle class imbalance
    # NIH is heavily imbalanced (most samples are "No Finding")
    pos_weights = torch.ones(NUM_CLASSES).to(DEVICE) * 5.0  # adjust per class if needed
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    
    scaler = torch.amp.GradScaler('cuda')
    
    os.makedirs("results", exist_ok=True)
    best_val_auc = 0
    
    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # --- Validate ---
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
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
                print(f"Warning: Only one class present in y_true for {name}. Setting AUC to 0.5")
                aucs.append(0.5)
        
        mean_auc = np.mean(aucs)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Mean AUC: {mean_auc:.4f}")
        for name, auc in zip(LABEL_NAMES, aucs):
            print(f"  {name}: {auc:.4f}")
        
        # Save best model
        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            torch.save(model.state_dict(), "results/best_model.pth")
            print(f"  ✓ Saved best model (AUC: {best_val_auc:.4f})")

if __name__ == "__main__":
    train()
