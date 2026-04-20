"""
Medical device artifact detection for chest X-rays.

Detects linear artifacts (catheters, chest drains, pacemaker leads, ECG wires)
using morphological operations and the Probabilistic Hough transform.  Returns a
soft mask highlighting device regions so the inference pipeline can flag and
penalise predictions whose Grad-CAM activations overlap these regions.

Device-simulating augmentation transforms (Task 2c) are also defined here so
that the training pipeline can inject synthetic device artifacts, helping the
model learn to ignore them.
"""

from __future__ import annotations

import logging
import random as _random

import cv2
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_medical_devices(image_array: np.ndarray) -> np.ndarray:
    """Detect linear medical artifacts in a chest X-ray image.

    Uses adaptive thresholding, morphological line filters (horizontal,
    vertical, and diagonal kernels), and the Probabilistic Hough transform
    to find thin high-intensity structures (tubes, drains, pacemaker leads,
    ECG wires).

    Parameters
    ----------
    image_array : np.ndarray
        RGB (H, W, 3) or grayscale (H, W) image as uint8.

    Returns
    -------
    np.ndarray
        Binary mask (H, W) uint8 — 255 where device artifacts are likely,
        0 elsewhere.
    """
    if image_array.ndim == 3:
        gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.astype(np.uint8)

    h, w = gray.shape

    # Enhance local contrast to make thin bright structures more visible
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive threshold to isolate bright linear structures
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=-5,
    )

    # Morphological line detectors — catheters and drains are elongated
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 8, 20), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 8, 20)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

    diag_size = max(min(h, w) // 10, 15)
    kernel_d1 = np.eye(diag_size, dtype=np.uint8)
    kernel_d2 = np.fliplr(np.eye(diag_size, dtype=np.uint8))
    d1_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_d1)
    d2_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_d2)

    # Probabilistic Hough for strong linear structures (leads, wires)
    hough_mask = np.zeros_like(gray)
    edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(w // 6, 60),
        minLineLength=max(w // 8, 40),
        maxLineGap=20,
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 3)

    # Union of all detections
    device_mask = cv2.bitwise_or(h_lines, v_lines)
    device_mask = cv2.bitwise_or(device_mask, d1_lines)
    device_mask = cv2.bitwise_or(device_mask, d2_lines)
    device_mask = cv2.bitwise_or(device_mask, hough_mask)

    # Dilate to capture the region immediately around each detected structure
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    device_mask = cv2.dilate(device_mask, dil_kernel)

    return device_mask


def flag_device_presence(image_array: np.ndarray, threshold: float = 0.15) -> bool:
    """Return True if medical devices are likely present in the image.

    Parameters
    ----------
    image_array : np.ndarray
        RGB or grayscale image.
    threshold : float
        Fraction of image pixels that must be flagged as device before the
        image is considered device-heavy (default 0.15 → 15%).

    Returns
    -------
    bool
        True when the detected device footprint exceeds *threshold*.
    """
    mask = detect_medical_devices(image_array)
    device_fraction = float((mask > 0).sum()) / mask.size
    present = device_fraction >= threshold
    logger.info(
        "Device detection: fraction=%.3f, present=%s", device_fraction, present
    )
    return present


# ---------------------------------------------------------------------------
# Device-simulating augmentation transforms (torchvision-compatible)
# ---------------------------------------------------------------------------

class RandomLinearArtifact:
    """Simulate a catheter or chest drain as a thin bright line.

    Randomly places 1–n_lines lines at arbitrary angles, mimicking
    high-intensity linear implants visible on chest X-rays.

    Parameters
    ----------
    p : float
        Probability of applying this transform per image.
    n_lines : int
        Maximum number of lines to draw.
    width : int
        Line thickness in pixels.
    """

    def __init__(self, p: float = 0.3, n_lines: int = 2, width: int = 2):
        self.p = p
        self.n_lines = n_lines
        self.width = width

    def __call__(self, img: Image.Image) -> Image.Image:
        if _random.random() > self.p:
            return img
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(_random.randint(1, self.n_lines)):
            x0, y0 = _random.randint(0, w), _random.randint(0, h)
            x1, y1 = _random.randint(0, w), _random.randint(0, h)
            # Devices appear brighter than surrounding tissue on X-ray
            brightness = _random.randint(200, 255)
            draw.line(
                [(x0, y0), (x1, y1)],
                fill=(brightness, brightness, brightness),
                width=self.width,
            )
        return img


class RandomCircularBlob:
    """Simulate ECG electrode pads as small circular blobs.

    Parameters
    ----------
    p : float
        Probability of applying per image.
    n_blobs : int
        Maximum number of blobs to draw.
    radius : int
        Approximate blob radius in pixels.
    """

    def __init__(self, p: float = 0.3, n_blobs: int = 6, radius: int = 8):
        self.p = p
        self.n_blobs = n_blobs
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if _random.random() > self.p:
            return img
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(_random.randint(1, self.n_blobs)):
            cx = _random.randint(self.radius, max(w - self.radius, self.radius + 1))
            cy = _random.randint(self.radius, max(h - self.radius, self.radius + 1))
            r = _random.randint(max(self.radius // 2, 1), self.radius)
            brightness = _random.randint(180, 240)
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)],
                fill=(brightness, brightness, brightness),
                outline=(150, 150, 150),
            )
        return img


class RandomEdgeWire:
    """Simulate a pacemaker lead or edge wire along one side of the image.

    Draws a slightly wavy high-intensity path near the left or right edge,
    mimicking the appearance of pacemaker leads on PA chest X-rays.

    Parameters
    ----------
    p : float
        Probability of applying per image.
    width : int
        Wire thickness in pixels.
    """

    def __init__(self, p: float = 0.2, width: int = 2):
        self.p = p
        self.width = width

    def __call__(self, img: Image.Image) -> Image.Image:
        if _random.random() > self.p:
            return img
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        side = _random.choice(("left", "right"))
        base_x = (
            _random.randint(5, w // 6)
            if side == "left"
            else _random.randint(5 * w // 6, w - 5)
        )
        n_segments = 8
        y_points = sorted(_random.randint(0, h) for _ in range(n_segments))
        pts = [(base_x + _random.randint(-5, 5), y) for y in y_points]
        for i in range(len(pts) - 1):
            brightness = _random.randint(190, 255)
            draw.line(
                [pts[i], pts[i + 1]],
                fill=(brightness, brightness, brightness),
                width=self.width,
            )
        return img
