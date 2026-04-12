"""
Run script for Multimodal Medical Diagnosis System training.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from data.dataset import REFLACXWithClinicalDataset
from model.xami import XAMIMultiConcatModal
from utils.train import split_dataset, train_with_chexnext


def main():
    # Device setup
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"Using device: {device}")

    # Load dataset (requires reflacx_with_clinical.csv in project root)
    # Note: image_path in CSV points to MIMIC data - update paths if needed
    print("Loading dataset...")
    reflacx_dataset = REFLACXWithClinicalDataset(image_size=256)

    # Create model: multimodal (CXR image + clinical data)
    print("Creating model...")
    model = XAMIMultiConcatModal(
        reflacx_dataset,
        device,
        use_clinical=True,
        use_image=True,
        model_dim=32,
        embeding_dim=64,
        dropout=0.2,
        pretrained=True,
        detach_image=False,
    )
    model = model.to(device)

    # Split dataset and create dataloaders
    print("Splitting dataset...")
    dataloaders = split_dataset(
        reflacx_dataset,
        batch_size=16,
        training_portion=0.8,
        test_portion=0.1,
        seed=123,
    )

    # Create saved_models directory
    os.makedirs("saved_models", exist_ok=True)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, threshold=0.001, factor=0.5
    )

    # Train with standard loss (MultiLabelSoftMarginLoss)
    print("Starting training...")
    train_with_chexnext(
        num_epochs=10,  # Use 10 for quick test; use 300 for full training
        model=model,
        dataset=reflacx_dataset,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_weighted=True,
        early_stop_count=5,
    )


if __name__ == "__main__":
    main()
