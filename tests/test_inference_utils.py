"""
Unit tests for utils/inference_utils.py

Tests cover:
  - compute_localization_score
  - check_gradcam_overlap_with_devices
  - adjust_confidence_for_devices
  - predict_with_tta (shape and averaging sanity check)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _uniform_heatmap(h: int = 32, w: int = 32, val: float = 1.0) -> np.ndarray:
    """Heatmap with uniform activation everywhere."""
    return np.full((h, w), val, dtype=np.float32)


def _region_heatmap(
    h: int = 32, w: int = 32,
    row_frac: tuple = (0.3, 0.75),
    col_frac: tuple = (0.3, 0.7),
    val: float = 1.0,
) -> np.ndarray:
    """Heatmap with activation only in the specified normalised region."""
    hm = np.zeros((h, w), dtype=np.float32)
    r0, r1 = int(row_frac[0] * h), int(row_frac[1] * h)
    c0, c1 = int(col_frac[0] * w), int(col_frac[1] * w)
    hm[r0:r1, c0:c1] = val
    return hm


# ---------------------------------------------------------------------------
# compute_localization_score
# ---------------------------------------------------------------------------

class TestComputeLocalizationScore:
    def test_unknown_pathology_returns_one(self):
        from utils.inference_utils import compute_localization_score

        hm = _uniform_heatmap()
        score = compute_localization_score(hm, "UnknownDisease")
        assert score == 1.0

    def test_zero_heatmap_returns_zero(self):
        from utils.inference_utils import compute_localization_score

        hm = np.zeros((32, 32), dtype=np.float32)
        score = compute_localization_score(hm, "Cardiomegaly")
        assert score == 0.0

    def test_cardiomegaly_perfect_localisation(self):
        from utils.inference_utils import compute_localization_score

        # Activation entirely within the Cardiomegaly expected region (0.3-0.7, 0.3-0.75)
        hm = _region_heatmap(row_frac=(0.35, 0.70), col_frac=(0.35, 0.65))
        score = compute_localization_score(hm, "Cardiomegaly")
        assert score >= 0.9, f"Expected near-perfect score, got {score:.3f}"

    def test_effusion_wrong_region_low_score(self):
        from utils.inference_utils import compute_localization_score

        # Activation in upper centre — wrong for Effusion (should be lower bilateral)
        hm = _region_heatmap(row_frac=(0.0, 0.3), col_frac=(0.3, 0.7))
        score = compute_localization_score(hm, "Effusion")
        assert score < 0.3, f"Expected low localisation score for wrong region, got {score:.3f}"

    def test_score_in_valid_range(self):
        from utils.inference_utils import compute_localization_score

        hm = _uniform_heatmap()
        for pathology in ["Cardiomegaly", "Effusion", "Edema", "Consolidation", "Atelectasis"]:
            score = compute_localization_score(hm, pathology)
            assert 0.0 <= score <= 1.0, f"Score out of [0,1] for {pathology}: {score}"


# ---------------------------------------------------------------------------
# check_gradcam_overlap_with_devices
# ---------------------------------------------------------------------------

class TestCheckGradcamOverlapWithDevices:
    def test_no_overlap_returns_false(self):
        from utils.inference_utils import check_gradcam_overlap_with_devices

        # Heatmap activation in top half; device mask in bottom half
        hm = _region_heatmap(row_frac=(0.0, 0.4), col_frac=(0.0, 1.0))
        dev = np.zeros((32, 32), dtype=np.uint8)
        dev[20:, :] = 255  # bottom half
        assert check_gradcam_overlap_with_devices(hm, dev, threshold=0.4) is False

    def test_full_overlap_returns_true(self):
        from utils.inference_utils import check_gradcam_overlap_with_devices

        hm = _uniform_heatmap()
        dev = np.full((32, 32), 255, dtype=np.uint8)
        assert check_gradcam_overlap_with_devices(hm, dev, threshold=0.4) is True

    def test_empty_device_mask_returns_false(self):
        from utils.inference_utils import check_gradcam_overlap_with_devices

        hm = _uniform_heatmap()
        dev = np.zeros((32, 32), dtype=np.uint8)
        assert check_gradcam_overlap_with_devices(hm, dev, threshold=0.4) is False

    def test_threshold_respected(self):
        from utils.inference_utils import check_gradcam_overlap_with_devices

        # 50% overlap
        hm = np.zeros((32, 32), dtype=np.float32)
        hm[:, :16] = 1.0  # left half
        dev = np.zeros((32, 32), dtype=np.uint8)
        dev[:, :16] = 255  # left half

        assert check_gradcam_overlap_with_devices(hm, dev, threshold=0.3) is True
        assert check_gradcam_overlap_with_devices(hm, dev, threshold=0.9) is False


# ---------------------------------------------------------------------------
# adjust_confidence_for_devices
# ---------------------------------------------------------------------------

class TestAdjustConfidenceForDevices:
    def _make_preds(self) -> dict:
        return {
            "Atelectasis":   0.70,
            "Cardiomegaly":  0.55,
            "Consolidation": 0.30,
            "Edema":         0.20,
            "Effusion":      0.65,
        }

    def test_no_device_mask_unchanged(self):
        from utils.inference_utils import adjust_confidence_for_devices

        preds = self._make_preds()
        empty_mask = np.zeros((32, 32), dtype=np.uint8)
        adjusted = adjust_confidence_for_devices(preds, empty_mask)
        assert adjusted == preds

    def test_device_present_flat_penalty_applied(self):
        from utils.inference_utils import adjust_confidence_for_devices

        preds = self._make_preds()
        full_mask = np.full((32, 32), 255, dtype=np.uint8)
        # No heatmaps supplied → flat fallback penalty
        adjusted = adjust_confidence_for_devices(preds, full_mask, gradcam_heatmaps=None)
        # Effusion and Atelectasis should be reduced
        assert adjusted["Effusion"] < preds["Effusion"]
        assert adjusted["Atelectasis"] < preds["Atelectasis"]
        # Cardiomegaly, Consolidation, Edema should be unchanged
        assert adjusted["Cardiomegaly"] == preds["Cardiomegaly"]
        assert adjusted["Consolidation"] == preds["Consolidation"]

    def test_heatmap_no_overlap_no_penalty(self):
        from utils.inference_utils import adjust_confidence_for_devices

        preds = self._make_preds()
        full_mask = np.full((32, 32), 255, dtype=np.uint8)
        # Heatmap activation AWAY from device mask (device mask covers bottom half,
        # heatmap has activation in top half only)
        dev_mask = np.zeros((32, 32), dtype=np.uint8)
        dev_mask[20:, :] = 255

        heatmaps = {
            "Effusion": _region_heatmap(row_frac=(0.0, 0.4), col_frac=(0.0, 1.0)),
            "Atelectasis": _region_heatmap(row_frac=(0.0, 0.3), col_frac=(0.0, 1.0)),
        }
        adjusted = adjust_confidence_for_devices(preds, dev_mask, gradcam_heatmaps=heatmaps)
        # No significant overlap → no penalty (threshold=0.4)
        assert adjusted["Effusion"] == pytest.approx(preds["Effusion"], abs=0.01)

    def test_probabilities_non_negative(self):
        from utils.inference_utils import adjust_confidence_for_devices

        preds = {"Effusion": 0.05, "Atelectasis": 0.03}
        full_mask = np.full((32, 32), 255, dtype=np.uint8)
        adjusted = adjust_confidence_for_devices(preds, full_mask)
        for v in adjusted.values():
            assert v >= 0.0


# ---------------------------------------------------------------------------
# predict_with_tta
# ---------------------------------------------------------------------------

class TestPredictWithTta:
    class _TinyModel(nn.Module):
        """Constant-output model for deterministic TTA testing."""
        def __init__(self, n_classes: int, val: float):
            super().__init__()
            self._val = val
            self._n = n_classes
            self._dummy = nn.Linear(1, 1)  # makes it a proper nn.Module

        def forward(self, x):
            return torch.full((x.shape[0], self._n), self._val)

    def test_output_keys_match_label_names(self):
        from torchvision import transforms
        from PIL import Image
        from utils.inference_utils import predict_with_tta

        model = self._TinyModel(n_classes=5, val=0.0)
        model.eval()
        img = Image.new("RGB", (128, 128))
        tf = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
        ])
        labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
        result = predict_with_tta(model, img, tf, labels, n_augments=2, device="cpu")
        assert set(result.keys()) == set(labels)

    def test_constant_model_output_matches(self):
        from torchvision import transforms
        from PIL import Image
        from utils.inference_utils import predict_with_tta

        # Model always outputs logit=0 → sigmoid(0)=0.5
        model = self._TinyModel(n_classes=5, val=0.0)
        model.eval()
        img = Image.new("RGB", (128, 128))
        tf = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
        ])
        labels = ["A", "B", "C", "D", "E"]
        result = predict_with_tta(model, img, tf, labels, n_augments=3, device="cpu")
        for v in result.values():
            assert abs(v - 0.5) < 1e-4, f"Expected 0.5, got {v}"

    def test_values_in_valid_probability_range(self):
        from torchvision import transforms
        from PIL import Image
        from utils.inference_utils import predict_with_tta

        model = self._TinyModel(n_classes=3, val=2.0)
        model.eval()
        img = Image.new("RGB", (64, 64))
        tf = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
        labels = ["X", "Y", "Z"]
        result = predict_with_tta(model, img, tf, labels, n_augments=2, device="cpu")
        for v in result.values():
            assert 0.0 <= v <= 1.0
