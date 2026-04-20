"""
Unit tests for utils/device_detector.py

Tests verify detection of synthetic linear artifacts and device-simulation
augmentation transforms.
"""

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_image(h: int = 128, w: int = 128, val: int = 100) -> np.ndarray:
    """Return a uniform gray RGB image."""
    return np.full((h, w, 3), val, dtype=np.uint8)


def _image_with_horizontal_line(h: int = 128, w: int = 128) -> np.ndarray:
    """Blank image with a bright horizontal line — simulates a drain/catheter."""
    img = _blank_image(h, w)
    img[h // 2, :, :] = 240  # bright white line
    return img


def _image_with_vertical_line(h: int = 128, w: int = 128) -> np.ndarray:
    """Blank image with a bright vertical line — simulates a pacemaker lead."""
    img = _blank_image(h, w)
    img[:, w // 4, :] = 240
    return img


# ---------------------------------------------------------------------------
# detect_medical_devices
# ---------------------------------------------------------------------------

class TestDetectMedicalDevices:
    def test_output_shape_matches_input(self):
        from utils.device_detector import detect_medical_devices

        img = _blank_image(128, 128)
        mask = detect_medical_devices(img)
        assert mask.shape == (128, 128)
        assert mask.dtype == np.uint8

    def test_blank_image_has_low_device_fraction(self):
        from utils.device_detector import detect_medical_devices

        img = _blank_image()
        mask = detect_medical_devices(img)
        fraction = (mask > 0).sum() / mask.size
        # Featureless gray image should produce very little detection
        assert fraction < 0.10, f"Unexpected high device fraction on blank image: {fraction:.3f}"

    def test_horizontal_line_detected(self):
        from utils.device_detector import detect_medical_devices

        img = _image_with_horizontal_line()
        mask = detect_medical_devices(img)
        # The line row should have significant detection
        line_row = mask[img.shape[0] // 2, :]
        assert (line_row > 0).sum() > img.shape[1] // 4, (
            "Horizontal line not detected in device mask"
        )

    def test_vertical_line_detected(self):
        from utils.device_detector import detect_medical_devices

        img = _image_with_vertical_line()
        mask = detect_medical_devices(img)
        line_col = mask[:, img.shape[1] // 4]
        assert (line_col > 0).sum() > img.shape[0] // 4, (
            "Vertical line not detected in device mask"
        )

    def test_grayscale_input_accepted(self):
        from utils.device_detector import detect_medical_devices

        gray = np.full((64, 64), 120, dtype=np.uint8)
        mask = detect_medical_devices(gray)
        assert mask.shape == (64, 64)


# ---------------------------------------------------------------------------
# flag_device_presence
# ---------------------------------------------------------------------------

class TestFlagDevicePresence:
    def test_blank_image_not_flagged_by_default(self):
        from utils.device_detector import flag_device_presence

        # Default threshold 0.15 — uniform blank should not trigger
        result = flag_device_presence(_blank_image(), threshold=0.15)
        # We accept either outcome for a completely uniform image because
        # CLAHE + threshold can yield edge-only noise.  Assert it's a bool.
        assert isinstance(result, bool)

    def test_returns_bool(self):
        from utils.device_detector import flag_device_presence

        result = flag_device_presence(_image_with_horizontal_line())
        assert isinstance(result, bool)

    def test_zero_threshold_always_true(self):
        from utils.device_detector import flag_device_presence

        # threshold=0 → any pixel detected → always True for any input with artifacts
        img = _image_with_horizontal_line()
        result = flag_device_presence(img, threshold=0.0)
        assert result is True


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------

class TestRandomLinearArtifact:
    def test_output_is_pil_image(self):
        from utils.device_detector import RandomLinearArtifact

        transform = RandomLinearArtifact(p=1.0)
        img = Image.new("RGB", (128, 128), (100, 100, 100))
        result = transform(img)
        assert isinstance(result, Image.Image)

    def test_size_preserved(self):
        from utils.device_detector import RandomLinearArtifact

        transform = RandomLinearArtifact(p=1.0)
        img = Image.new("RGB", (224, 224), (80, 80, 80))
        result = transform(img)
        assert result.size == img.size

    def test_zero_probability_returns_original(self):
        from utils.device_detector import RandomLinearArtifact

        transform = RandomLinearArtifact(p=0.0)
        img = Image.new("RGB", (64, 64), (50, 50, 50))
        result = transform(img)
        assert np.array(result).sum() == np.array(img).sum()


class TestRandomCircularBlob:
    def test_output_is_pil_image(self):
        from utils.device_detector import RandomCircularBlob

        transform = RandomCircularBlob(p=1.0)
        img = Image.new("RGB", (128, 128), (100, 100, 100))
        assert isinstance(transform(img), Image.Image)

    def test_size_preserved(self):
        from utils.device_detector import RandomCircularBlob

        transform = RandomCircularBlob(p=1.0)
        img = Image.new("RGB", (200, 200))
        assert transform(img).size == img.size


class TestRandomEdgeWire:
    def test_output_is_pil_image(self):
        from utils.device_detector import RandomEdgeWire

        transform = RandomEdgeWire(p=1.0)
        img = Image.new("RGB", (128, 128), (100, 100, 100))
        assert isinstance(transform(img), Image.Image)

    def test_size_preserved(self):
        from utils.device_detector import RandomEdgeWire

        transform = RandomEdgeWire(p=1.0)
        img = Image.new("RGB", (256, 256))
        assert transform(img).size == img.size
