"""
Data loading and preprocessing for NIH ChestX-ray14.

Changes from original:
  • Training augmentation pipeline extended with:
      - RandomRotation(10) for positional invariance
      - ColorJitter for intensity normalisation robustness
      - Device-simulating transforms (RandomLinearArtifact, RandomCircularBlob,
        RandomEdgeWire) applied with low probability so the model learns to
        ignore catheter/drain/lead artifacts (Task 2c).
  • ``WeightedFocalLoss`` added (Task 4a): reduces loss contribution from easy
    negatives and focuses training on hard examples.  gamma=2, alpha=class weights.
  • ``get_nih_dataloaders`` accepts ``use_device_augmentation`` flag.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Positive-weight computation
# ---------------------------------------------------------------------------

def compute_pos_weights_from_csv(
    csv_path: str,
    label_names: list[str],
    fallback_pos_weight: float = 5.0,
) -> dict[str, float]:
    """Compute per-class positive-to-negative weight ratios for BCEWithLogitsLoss.

    For each class: ``pos_weight = num_negatives / num_positives``
    (PyTorch convention).

    Parameters
    ----------
    csv_path : str
        Path to the training CSV with binary label columns.
    label_names : list[str]
        List of column names to compute weights for.
    fallback_pos_weight : float
        Weight used when a class has zero positive examples.

    Returns
    -------
    dict[str, float]
        ``{class_name: pos_weight}``.
    """
    df = pd.read_csv(csv_path)
    weights: dict[str, float] = {}
    for name in label_names:
        if name not in df.columns:
            weights[name] = float(fallback_pos_weight)
            continue
        col = df[name]
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        weights[name] = (
            float(fallback_pos_weight) if n_pos == 0 else float(n_neg) / float(n_pos)
        )
    return weights


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class WeightedFocalLoss(nn.Module):
    """Multi-label focal loss with per-class alpha weights.

    Reduces loss contribution from easy negatives so training focuses on
    ambiguous and hard-to-classify examples — particularly useful for the
    device-confounded chest X-ray setting.

    Formula (per sample, per class):
        p_t  = sigmoid(logit)  for positive labels
             = 1 - sigmoid(logit) for negative labels
        FL   = -alpha * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    class_weights : torch.Tensor
        Per-class alpha weights, shape ``(num_classes,)``.
        Typically the ``pos_weights`` from :func:`compute_pos_weights_from_csv`,
        normalised to [0, 1] so they act as class-balance factors.
    gamma : float
        Focusing parameter.  ``gamma=0`` → standard BCE; ``gamma=2`` is
        the value from the original focal loss paper (Lin et al. 2017).
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted focal loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw (pre-sigmoid) model outputs, shape ``(B, C)``.
        targets : torch.Tensor
            Binary labels, shape ``(B, C)``, values in ``{0, 1}``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        probs = torch.sigmoid(logits)
        # p_t for focal weighting
        p_t = targets * probs + (1 - targets) * (1 - probs)
        # Standard binary cross-entropy term
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_weight = (1.0 - p_t) ** self.gamma
        # Alpha: use class_weights for positives, (1 - normalised weight) for negatives
        alpha_t = targets * self.class_weights + (1 - targets) * (1.0 - self.class_weights)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def prepare_nih_csv(
    train_csv_path: str,
    valid_csv_path: str,
    data_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load train/val CSVs and resolve full image paths.

    Parameters
    ----------
    train_csv_path, valid_csv_path : str
        Paths to CSVs with a ``Path`` column and binary label columns.
    data_dir : str
        Root directory containing the ``images/`` folder.

    Returns
    -------
    tuple
        ``(train_df, val_df, target_labels)``
    """
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(valid_csv_path)

    target_labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]

    train_df["image_path"] = train_df["Path"].apply(
        lambda x: os.path.join(data_dir, x)
    )
    val_df["image_path"] = val_df["Path"].apply(
        lambda x: os.path.join(data_dir, x)
    )
    return train_df, val_df, target_labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NIHChestDataset(Dataset):
    """PyTorch Dataset for NIH ChestX-ray14 images.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``image_path`` column and binary label columns.
    target_labels : list[str]
        Label columns to extract as the target tensor.
    transform : callable, optional
        torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_labels: list[str],
        transform=None,
    ):
        self.df = df
        self.target_labels = target_labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"Warning: could not load {image_path}: {exc}. Using blank image.")
            image = Image.new("RGB", (256, 256))

        if self.transform:
            image = self.transform(image)

        labels = row[self.target_labels].values.astype(np.float32)
        return image, torch.tensor(labels)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_nih_dataloaders(
    train_csv_path: str,
    valid_csv_path: str,
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    use_device_augmentation: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for NIH ChestX-ray14.

    Training augmentations (Task 2c):
      • Standard geometric/colour augmentations for generalisation.
      • Device-simulating transforms (RandomLinearArtifact, RandomCircularBlob,
        RandomEdgeWire) applied with low probability so the model is exposed
        to medical device artifacts during training.

    Parameters
    ----------
    train_csv_path, valid_csv_path : str
        Paths to training and validation CSV files.
    data_dir : str
        Root directory of the NIH dataset.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of DataLoader worker processes.
    use_device_augmentation : bool
        When ``True`` (default) inject device-simulating transforms into
        the training pipeline.

    Returns
    -------
    tuple
        ``(train_loader, val_loader)``
    """
    train_df, val_df, target_labels = prepare_nih_csv(
        train_csv_path, valid_csv_path, data_dir
    )

    _normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Build training transform list
    train_transforms: list = [
        transforms.Resize(128),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]

    if use_device_augmentation:
        try:
            from utils.device_detector import (
                RandomCircularBlob,
                RandomEdgeWire,
                RandomLinearArtifact,
            )
            train_transforms += [
                RandomLinearArtifact(p=0.15, n_lines=2, width=2),
                RandomCircularBlob(p=0.15, n_blobs=5, radius=6),
                RandomEdgeWire(p=0.10, width=2),
            ]
        except ImportError:
            print("Warning: device_detector not available — skipping device augmentations.")

    train_transforms += [transforms.ToTensor(), _normalize]

    train_transform = transforms.Compose(train_transforms)

    val_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        _normalize,
    ])

    train_dataset = NIHChestDataset(train_df, target_labels, transform=train_transform)
    val_dataset = NIHChestDataset(val_df, target_labels, transform=val_transform)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader
