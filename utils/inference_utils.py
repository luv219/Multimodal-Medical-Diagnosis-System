"""
Inference utilities: anatomy-aware localization scoring, device-aware confidence
adjustment, and test-time augmentation (TTA).

These helpers are designed to be composable and backward-compatible with the
existing inference API in scripts/predict_nih.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anatomical expected activation regions (normalised 0-1 image coordinates)
# Each entry is a list of (x_min, y_min, x_max, y_max) boxes.
# Multiple boxes are used when activation can occur in bilateral regions.
# ---------------------------------------------------------------------------

ANATOMICAL_BOUNDS: dict[str, list[tuple[float, float, float, float]]] = {
    # Central cardiac silhouette — lower-mid mediastinum
    "Cardiomegaly":  [(0.30, 0.30, 0.70, 0.75)],
    # Bilateral costophrenic angles and dependent pleural spaces
    "Effusion":      [(0.00, 0.60, 0.35, 1.00), (0.65, 0.60, 1.00, 1.00)],
    # Perihilar bilateral zones (butterfly / bat-wing pattern)
    "Edema":         [(0.20, 0.20, 0.80, 0.80)],
    # Mid and lower lung parenchyma
    "Consolidation": [(0.10, 0.30, 0.90, 0.90)],
    # Lower lobe bilateral zones (commonest site for subsegmental collapse)
    "Atelectasis":   [(0.10, 0.50, 0.90, 1.00)],
}


def compute_localization_score(
    gradcam_heatmap: np.ndarray,
    pathology: str,
    image_shape: Optional[tuple] = None,
) -> float:
    """Measure how well the Grad-CAM activation aligns with expected anatomy.

    The score is the fraction of total heatmap activation that falls inside the
    expected anatomical bounding box(es) for *pathology*.

    Parameters
    ----------
    gradcam_heatmap : np.ndarray
        2-D float array in [0, 1] at any spatial resolution.
    pathology : str
        One of the model's LABEL_NAMES (e.g. ``"Cardiomegaly"``).
    image_shape : tuple, optional
        ``(H, W)`` of the original image — reserved for future use; the
        heatmap is already spatially normalised so this parameter is ignored.

    Returns
    -------
    float
        Localisation score in [0, 1].
        • 1.0 — all activation is inside the expected anatomical region.
        • 0.0 — no activation overlaps the expected region.
    """
    if pathology not in ANATOMICAL_BOUNDS:
        return 1.0  # unknown pathology — no penalty

    h, w = gradcam_heatmap.shape[:2]
    total = float(gradcam_heatmap.sum())
    if total <= 0:
        return 0.0

    inside = 0.0
    for x_min, y_min, x_max, y_max in ANATOMICAL_BOUNDS[pathology]:
        r0, r1 = int(y_min * h), int(y_max * h)
        c0, c1 = int(x_min * w), int(x_max * w)
        inside += float(gradcam_heatmap[r0:r1, c0:c1].sum())

    score = float(np.clip(inside / (total + 1e-8), 0.0, 1.0))
    logger.debug("Localisation score for %s: %.3f", pathology, score)
    return score


def check_gradcam_overlap_with_devices(
    gradcam_mask: np.ndarray,
    device_mask: np.ndarray,
    threshold: float = 0.4,
) -> bool:
    """Return True when Grad-CAM activation strongly overlaps detected device regions.

    Parameters
    ----------
    gradcam_mask : np.ndarray
        2-D float heatmap in [0, 1].
    device_mask : np.ndarray
        Binary uint8 mask (H, W) from
        :func:`device_detector.detect_medical_devices`.
    threshold : float
        Overlap ratio above which the prediction is considered device-confounded.

    Returns
    -------
    bool
        ``True`` when the weighted device-region activation fraction exceeds
        *threshold*.
    """
    gh, gw = gradcam_mask.shape[:2]
    dev_resized = cv2.resize(device_mask.astype(np.float32), (gw, gh)) / 255.0

    total = float(gradcam_mask.sum()) + 1e-8
    overlap = float((gradcam_mask * dev_resized).sum())
    ratio = overlap / total
    logger.debug("Device overlap ratio: %.3f (threshold %.2f)", ratio, threshold)
    return ratio >= threshold


def adjust_confidence_for_devices(
    predictions: dict[str, float],
    device_mask: np.ndarray,
    gradcam_heatmaps: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, float]:
    """Down-weight predictions for pathologies whose Grad-CAM overlaps device regions.

    Rules applied when a device mask is non-empty:

    * **Effusion** — penalise by up to 18 % when activation overlaps device region.
    * **Atelectasis** — penalise by up to 13 % when activation overlaps device region.

    When *gradcam_heatmaps* is ``None`` (heatmaps were not generated), a flat
    half-penalty is applied as a conservative fallback.

    Parameters
    ----------
    predictions : dict[str, float]
        Raw model predictions ``{label: probability}``.
    device_mask : np.ndarray
        Binary device mask from :func:`device_detector.detect_medical_devices`.
    gradcam_heatmaps : dict, optional
        ``{label: 2-D heatmap}`` from :func:`visualizer.generate_all_heatmaps`.

    Returns
    -------
    dict[str, float]
        Adjusted ``{label: probability}`` dict.
    """
    if not (device_mask > 0).any():
        return predictions

    # Per-pathology maximum penalty fractions
    DEVICE_SENSITIVE: dict[str, float] = {
        "Effusion":    0.18,
        "Atelectasis": 0.13,
    }

    adjusted = dict(predictions)
    for label, max_penalty in DEVICE_SENSITIVE.items():
        if label not in adjusted:
            continue

        if gradcam_heatmaps and label in gradcam_heatmaps:
            if check_gradcam_overlap_with_devices(gradcam_heatmaps[label], device_mask):
                adjusted[label] = max(0.0, adjusted[label] * (1.0 - max_penalty))
                logger.info(
                    "Device artifact: penalised %s by %.0f%% (GradCAM overlap)",
                    label, max_penalty * 100,
                )
        else:
            # Conservative flat fallback — half the max penalty
            adjusted[label] = max(0.0, adjusted[label] * (1.0 - max_penalty * 0.5))
            logger.info(
                "Device artifact: flat %.0f%% penalty applied to %s (no heatmap)",
                max_penalty * 50, label,
            )

    return adjusted


def predict_with_tta(
    model: torch.nn.Module,
    image: Image.Image,
    transform: transforms.Compose,
    label_names: list[str],
    n_augments: int = 5,
    device: str = "cpu",
) -> dict[str, float]:
    """Average predictions across several augmented views (test-time augmentation).

    Augmentations applied (randomly sampled each pass):
    * Horizontal flip
    * Rotation ±5 degrees
    * Zoom 0.95–1.05×
    * Brightness shift ±15 %

    Parameters
    ----------
    model : nn.Module
        Model in ``eval()`` mode.
    image : PIL.Image
        Original (un-transformed) chest X-ray.
    transform : transforms.Compose
        Standard inference transform applied **after** TTA augmentations.
    label_names : list[str]
        Ordered list of class labels matching logit output indices.
    n_augments : int
        Number of additional augmented views (plus the unaugmented original).
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    dict[str, float]
        ``{label: probability}`` averaged across all views.
        Typically improves AUC by 1–3 % compared with single-pass inference.
    """
    tta_augs = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15),
    ])

    model.eval()
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        # Unaugmented original
        tensor = transform(image).unsqueeze(0).to(device)
        logits = model(tensor)
        all_probs.append(torch.sigmoid(logits).squeeze().float().cpu().numpy())

        # Augmented views
        for _ in range(n_augments):
            aug = tta_augs(image)
            tensor = transform(aug).unsqueeze(0).to(device)
            logits = model(tensor)
            all_probs.append(torch.sigmoid(logits).squeeze().float().cpu().numpy())

    avg = np.mean(all_probs, axis=0)
    logger.info("TTA: averaged %d views", len(all_probs))
    return {name: float(p) for name, p in zip(label_names, avg)}
