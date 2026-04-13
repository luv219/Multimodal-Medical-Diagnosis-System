"""
Grad-CAM visualiser for ImageOnlyModel (and compatible single-input models).

Uses raw PyTorch forward/backward hooks — the model class is never modified.
This module is independent of the ``pytorch_grad_cam`` library used in
``utils/gradcam.py`` (which targets the multimodal REFLACX pipeline).

Mathematical formulation (Grad-CAM, Selvaraju et al. 2017)
-----------------------------------------------------------
Given a convolutional feature map A^k (k = 1 … K channels) at the target
layer and the scalar class logit y^c:

1.  α_k = (1 / Z) Σ_i Σ_j  ∂y^c / ∂A^k_{ij}      (global-average-pooled gradient)
2.  L   = ReLU( Σ_k  α_k · A^k )                    (weighted combination, clamped)
3.  L   = upsample(L, input_spatial_size)             (bilinear resize to input dims)
4.  L   = L / max(L)                                  (normalise to [0, 1])
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Auto-detect the last convolutional feature-map layer.

    Supports:
      - ImageOnlyModel with DenseNet121 backbone  → ``backbone.features[-1]``
      - ImageOnlyModel with ResNet50 backbone      → ``backbone.layer4[-1]``
    """
    backbone = getattr(model, "backbone", model)

    # DenseNet variants expose a ``features`` Sequential
    if hasattr(backbone, "features"):
        return backbone.features[-1]

    # ResNet variants expose ``layer4``
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1]

    raise RuntimeError(
        "Cannot auto-detect target layer. Pass it explicitly via ``target_layer``."
    )


# ---------------------------------------------------------------------------
# Core Grad-CAM
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    target_layer: Optional[torch.nn.Module] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Compute the Grad-CAM heatmap for a single class.

    Parameters
    ----------
    model : nn.Module
        The model in ``eval()`` mode.
    input_tensor : Tensor
        Pre-processed image tensor of shape ``(1, C, H, W)``.
    target_class_idx : int
        Index of the target class in the logit vector.
    target_layer : nn.Module, optional
        Layer to hook.  Auto-detected when *None*.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(H_input, W_input)`` with values in ``[0, 1]``.
    """
    if target_layer is None:
        target_layer = _detect_target_layer(model)

    model.eval()
    input_tensor = input_tensor.to(device).requires_grad_(False)

    # Containers for the hook data
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    tensor_hook_handles: list = []

    def _fwd_hook(_module, _inp, out):
        feat = out[0] if isinstance(out, tuple) else out
        activations.append(feat.detach().clone())

        def _tensor_bw_hook(grad):
            gradients.append(grad.detach().clone())

        tensor_hook_handles.append(feat.register_hook(_tensor_bw_hook))

    handle_fwd = target_layer.register_forward_hook(_fwd_hook)

    try:
        # Forward — need grad for backward pass
        input_grad = input_tensor.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = model(input_grad)
            score = logits[0, target_class_idx]
            model.zero_grad()
            score.backward()
    finally:
        handle_fwd.remove()
        for h in tensor_hook_handles:
            h.remove()

    # α_k = global-average-pool of the gradient for each channel
    act = activations[0]     # (1, K, h, w)
    grad = gradients[0]      # (1, K, h, w)
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

    # Weighted combination + ReLU
    cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, h, w)
    cam = F.relu(cam)

    # Upsample to input spatial size
    cam = F.interpolate(
        cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
    )

    cam = cam.squeeze().float().cpu().numpy()
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max
    return cam


# ---------------------------------------------------------------------------
# Overlay & peak detection
# ---------------------------------------------------------------------------

def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto the original image.

    Parameters
    ----------
    original_image : PIL.Image
        The *unprocessed* chest X-ray.
    heatmap : np.ndarray
        2-D array in ``[0, 1]`` (from :func:`generate_gradcam`).
    colormap : int
        OpenCV colourmap constant (default ``cv2.COLORMAP_JET``).
    alpha : float
        Blend weight for the heatmap (0 = image only, 1 = heatmap only).

    Returns
    -------
    np.ndarray
        RGB array of shape ``(H, W, 3)`` with dtype ``uint8``.
    """
    img_rgb = np.array(original_image.convert("RGB"))
    h, w = img_rgb.shape[:2]

    # Resize heatmap to original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colour = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

    blended = (
        (1 - alpha) * img_rgb.astype(np.float32)
        + alpha * heatmap_colour.astype(np.float32)
    ).astype(np.uint8)
    return blended


def find_peak_activation(
    heatmap: np.ndarray,
    original_image_size: tuple[int, int],
) -> tuple[int, int]:
    """Return the ``(row, col)`` of peak activation in original-image pixel space.

    Parameters
    ----------
    heatmap : np.ndarray
        2-D heatmap from :func:`generate_gradcam` (model-input resolution).
    original_image_size : tuple[int, int]
        ``(height, width)`` of the original image before pre-processing.

    Returns
    -------
    tuple[int, int]
        ``(row, col)`` in the original image coordinate system.
    """
    h_orig, w_orig = original_image_size
    hm_resized = cv2.resize(heatmap, (w_orig, h_orig))
    idx = int(np.argmax(hm_resized))
    row, col = divmod(idx, w_orig)
    return (row, col)


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def generate_all_heatmaps(
    model: torch.nn.Module,
    image_path: str,
    prob_dict: dict[str, float],
    label_names: list[str],
    transform: transforms.Compose,
    threshold: float = 0.10,
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> dict:
    """Generate Grad-CAM outputs for every class above *threshold*.

    Parameters
    ----------
    model : nn.Module
        Loaded model in eval mode.
    image_path : str
        Path to the original chest X-ray.
    prob_dict : dict
        ``{label: probability}`` as produced by the inference step.
    label_names : list[str]
        Ordered list of class labels matching the model's logit indices.
    transform : torchvision.transforms.Compose
        The same pre-processing pipeline used for inference.
    threshold : float
        Minimum probability to generate a heatmap for a class.
    device : str
        ``"cpu"`` or ``"cuda"``.
    output_dir : str, optional
        Directory to save overlay images.  Skipped when *None*.

    Returns
    -------
    dict
        ``{"heatmaps": {label: ndarray}, "overlays": {label: ndarray},
          "overlay_paths": {label: str}, "peaks": {label: (row, col)}}``
    """
    original_image = Image.open(image_path).convert("RGB")
    img_tensor = transform(original_image).unsqueeze(0).to(device)
    orig_size = (original_image.height, original_image.width)

    target_layer = _detect_target_layer(model)

    heatmaps: dict[str, np.ndarray] = {}
    overlays: dict[str, np.ndarray] = {}
    overlay_paths: dict[str, str] = {}
    peaks: dict[str, tuple] = {}

    for label, prob in prob_dict.items():
        if prob < threshold:
            continue
        class_idx = label_names.index(label)
        cam = generate_gradcam(model, img_tensor, class_idx, target_layer, device)
        heatmaps[label] = cam
        overlays[label] = overlay_heatmap(original_image, cam)
        peaks[label] = find_peak_activation(cam, orig_size)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"gradcam_{label.lower()}.png")
            Image.fromarray(overlays[label]).save(save_path)
            overlay_paths[label] = save_path

    return {
        "heatmaps": heatmaps,
        "overlays": overlays,
        "overlay_paths": overlay_paths,
        "peaks": peaks,
    }
