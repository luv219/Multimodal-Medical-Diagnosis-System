"""
Grad-CAM++ visualiser for ImageOnlyModel (and compatible single-input models).

Upgrades over vanilla Grad-CAM (Task 6a):
  • Uses GradCAM++ (Chattopadhyay et al. 2018) via the ``pytorch_grad_cam``
    library when available.  Falls back to hand-rolled vanilla Grad-CAM when
    the library is not installed.
  • GradCAM++ handles multiple activations of the same class and gives tighter
    localisation on small or diffuse findings.

New in this version (Task 6b):
  • :func:`generate_all_heatmaps` now returns a ``"localization_scores"`` key
    containing a per-label anatomical localisation score (0–1) computed by
    :func:`inference_utils.compute_localization_score`.

Mathematical formulation (Grad-CAM++, Chattopadhyay et al. 2018)
-----------------------------------------------------------------
For feature map A^k and class score y^c:

    α_k^c  =  Σ_{i,j} [∂²y^c / ∂(A^k_{ij})²]
              ─────────────────────────────────────────────────
              2·∂²y^c/∂(A^k_{ij})² + Σ_{i',j'} A^k_{i'j'} · ∂y^c/∂(A^k_{ij})

    L^{c,++} = ReLU( Σ_k  α_k^c · A^k )
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GradCAM++ via pytorch_grad_cam (preferred) with vanilla fallback
# ---------------------------------------------------------------------------

def _generate_gradcam_pp(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    target_layer: torch.nn.Module,
    device: str,
) -> np.ndarray:
    """GradCAM++ using the pytorch_grad_cam library."""
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam = cam(
            input_tensor=input_tensor.to(device),
            targets=targets,
        )
    # cam returns shape (B, H, W); take first element
    heatmap = grayscale_cam[0]
    cam_max = heatmap.max()
    if cam_max > 0:
        heatmap = heatmap / cam_max
    return heatmap.astype(np.float32)


def _generate_gradcam_vanilla(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    target_layer: torch.nn.Module,
    device: str,
) -> np.ndarray:
    """Vanilla Grad-CAM fallback using raw PyTorch hooks.

    Mathematical formulation (Selvaraju et al. 2017):
      α_k = global-average-pool(∂y^c / ∂A^k)
      L   = ReLU(Σ_k  α_k · A^k), then upsampled to input resolution.
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    hook_handles: list = []

    def _fwd_hook(_module, _inp, out):
        feat = out[0] if isinstance(out, tuple) else out
        activations.append(feat.detach().clone())

        def _tensor_bw(grad):
            gradients.append(grad.detach().clone())

        hook_handles.append(feat.register_hook(_tensor_bw))

    handle_fwd = target_layer.register_forward_hook(_fwd_hook)

    try:
        inp = input_tensor.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = model(inp)
            score = logits[0, target_class_idx]
            model.zero_grad()
            score.backward()
    finally:
        handle_fwd.remove()
        for h in hook_handles:
            h.remove()

    act = activations[0]       # (1, K, h, w)
    grad = gradients[0]        # (1, K, h, w)
    weights = grad.mean(dim=(2, 3), keepdim=True)

    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(
        cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
    )
    cam = cam.squeeze().float().cpu().numpy()
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max
    return cam


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Auto-detect the last convolutional feature-map layer.

    Supports:
    • ``ImageOnlyModel`` with DenseNet121 — ``backbone.features[-1]``
    • ``ImageOnlyModel`` with ResNet50   — ``backbone.layer4[-1]``
    """
    backbone = getattr(model, "backbone", model)

    if hasattr(backbone, "features"):      # DenseNet
        return backbone.features[-1]
    if hasattr(backbone, "layer4"):        # ResNet
        return backbone.layer4[-1]

    raise RuntimeError(
        "Cannot auto-detect target layer for GradCAM. Pass it explicitly via "
        "``target_layer``."
    )


# ---------------------------------------------------------------------------
# Public Grad-CAM entry point
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    target_layer: Optional[torch.nn.Module] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Compute the GradCAM++ heatmap for a single class.

    Tries ``pytorch_grad_cam.GradCAMPlusPlus`` first; falls back to vanilla
    Grad-CAM when the library is unavailable.

    Parameters
    ----------
    model : nn.Module
        Model in ``eval()`` mode.
    input_tensor : Tensor
        Pre-processed image, shape ``(1, C, H, W)``.
    target_class_idx : int
        Index of the target class in the logit vector.
    target_layer : nn.Module, optional
        Layer to hook.  Auto-detected when ``None``.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    np.ndarray
        2-D heatmap ``(H_input, W_input)`` in ``[0, 1]``.
    """
    if target_layer is None:
        target_layer = _detect_target_layer(model)

    model.eval()

    try:
        cam = _generate_gradcam_pp(model, input_tensor, target_class_idx, target_layer, device)
        logger.debug("Used GradCAM++ (pytorch_grad_cam) for class %d", target_class_idx)
    except ImportError:
        logger.warning(
            "pytorch_grad_cam not installed — falling back to vanilla Grad-CAM. "
            "Install with: pip install grad-cam"
        )
        cam = _generate_gradcam_vanilla(model, input_tensor, target_class_idx, target_layer, device)

    return cam


# ---------------------------------------------------------------------------
# Overlay & peak detection (unchanged API)
# ---------------------------------------------------------------------------

def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a Grad-CAM++ heatmap onto the original image.

    Parameters
    ----------
    original_image : PIL.Image
        The *unprocessed* chest X-ray.
    heatmap : np.ndarray
        2-D array in ``[0, 1]`` from :func:`generate_gradcam`.
    colormap : int
        OpenCV colourmap constant (default ``cv2.COLORMAP_JET``).
    alpha : float
        Blend weight for the heatmap (0 = image only, 1 = heatmap only).

    Returns
    -------
    np.ndarray
        RGB array ``(H, W, 3)`` dtype ``uint8``.
    """
    img_rgb = np.array(original_image.convert("RGB"))
    h, w = img_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_u8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colour = cv2.applyColorMap(heatmap_u8, colormap)
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
    """Return ``(row, col)`` of peak activation in original-image pixel space.

    Parameters
    ----------
    heatmap : np.ndarray
        2-D heatmap at model-input resolution.
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
# High-level convenience — now also returns localization_scores
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
    """Generate GradCAM++ outputs for every class above *threshold*.

    Parameters
    ----------
    model : nn.Module
        Loaded model in eval mode.
    image_path : str
        Path to the original chest X-ray.
    prob_dict : dict
        ``{label: probability}`` from inference.
    label_names : list[str]
        Ordered class labels matching logit output indices.
    transform : torchvision.transforms.Compose
        The same pre-processing pipeline used for inference.
    threshold : float
        Minimum probability to generate a heatmap.
    device : str
        ``"cpu"`` or ``"cuda"``.
    output_dir : str, optional
        Directory to save overlay images.

    Returns
    -------
    dict
        Keys: ``"heatmaps"``, ``"overlays"``, ``"overlay_paths"``,
        ``"peaks"``, ``"localization_scores"``.
        ``localization_scores`` maps each label to a float in [0, 1]
        indicating how well the activation aligns with expected anatomy.
    """
    from utils.inference_utils import compute_localization_score

    original_image = Image.open(image_path).convert("RGB")
    img_tensor = transform(original_image).unsqueeze(0).to(device)
    orig_size = (original_image.height, original_image.width)

    target_layer = _detect_target_layer(model)

    heatmaps: dict[str, np.ndarray] = {}
    overlays: dict[str, np.ndarray] = {}
    overlay_paths: dict[str, str] = {}
    peaks: dict[str, tuple] = {}
    localization_scores: dict[str, float] = {}

    for label, prob in prob_dict.items():
        if prob < threshold:
            continue
        class_idx = label_names.index(label)
        cam = generate_gradcam(model, img_tensor, class_idx, target_layer, device)

        heatmaps[label] = cam
        overlays[label] = overlay_heatmap(original_image, cam)
        peaks[label] = find_peak_activation(cam, orig_size)
        localization_scores[label] = compute_localization_score(cam, label)

        logger.info(
            "GradCAM++ %s: peak=%s  localisation=%.2f",
            label, peaks[label], localization_scores[label],
        )

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
        "localization_scores": localization_scores,
    }
