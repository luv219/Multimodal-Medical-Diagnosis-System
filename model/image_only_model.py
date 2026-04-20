"""
Single-modality chest X-ray classification model (NIH ChestX-ray14 pipeline).

Architecture improvements over the original:
  • Separate feature extractor from backbone so GradCAM hooks still work.
  • Global Average Pooling + Global Max Pooling concatenated before the head —
    GAP captures average response, GMP captures peak activation.
  • Improved classification head: FC(2×in_features → 512) → BN → ReLU →
    Dropout(0.3) → FC(512 → num_classes).
  • Label smoothing of 0.05 recommended at the loss level (see train_nih.py).

Older ``results/best_model.pth`` files may use the legacy layout (DenseNet121
with a single ``backbone.classifier`` Linear). Use
:func:`load_nih_image_checkpoint` so inference picks the matching module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LegacyDenseNetImageOnly(nn.Module):
    """DenseNet121 + linear head; matches pre-refactor NIH checkpoints.

    State dict keys look like ``backbone.features.*`` and
    ``backbone.classifier.{weight,bias}`` (no ``classifier_head``).
    """

    def __init__(self, num_classes: int = 5, use_pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if use_pretrained else None
        base = models.densenet121(weights=weights)
        in_features = base.classifier.in_features
        base.classifier = nn.Linear(in_features, num_classes)
        self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _unwrap_checkpoint_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a state dict or a dict wrapping it.")
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj  # type: ignore[return-value]


def _strip_dataparallel_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def load_nih_image_checkpoint(
    path: str,
    num_classes: int = 5,
    map_location=None,
    backbone: str = "densenet121",
    use_pretrained: bool = False,
) -> nn.Module:
    """Build the correct image-only module and load ``path`` (strict).

    Supports current ``ImageOnlyModel`` checkpoints (``classifier_head.*``)
    and legacy DenseNet checkpoints (``backbone.classifier.weight``).
    """
    raw = torch.load(path, map_location=map_location, weights_only=True)
    state = _strip_dataparallel_prefix(_unwrap_checkpoint_state_dict(raw))

    if "classifier_head.0.weight" in state:
        model = ImageOnlyModel(
            num_classes=num_classes,
            backbone=backbone,
            use_pretrained=use_pretrained,
        )
        model.load_state_dict(state, strict=True)
        return model

    if "backbone.classifier.weight" in state:
        if backbone != "densenet121":
            raise ValueError(
                "This checkpoint is a legacy DenseNet121 image-only model; "
                f"backbone must be 'densenet121', got {backbone!r}."
            )
        model = LegacyDenseNetImageOnly(
            num_classes=num_classes,
            use_pretrained=use_pretrained,
        )
        model.load_state_dict(state, strict=True)
        return model

    sample = list(state.keys())[:12]
    raise ValueError(
        "Unrecognized NIH image checkpoint (expected classifier_head.* or "
        f"backbone.classifier.*). Sample keys: {sample}"
    )


class ImageOnlyModel(nn.Module):
    """Multi-label chest X-ray classifier with an improved pooling-based head.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 5 for the NIH 5-class subset).
    backbone : str
        ``"densenet121"`` or ``"resnet50"``.
    use_pretrained : bool
        Whether to initialise the backbone with ImageNet weights.
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "densenet121",
        use_pretrained: bool = True,
    ):
        super().__init__()
        weights = "DEFAULT" if use_pretrained else None
        self._backbone_type = backbone

        if backbone == "densenet121":
            base = models.densenet121(weights=weights)
            # Keep the convolutional feature extractor intact for GradCAM hooks
            self.backbone = base
            in_features = base.classifier.in_features  # 1024
            # Replace DenseNet's built-in classifier with Identity so our
            # forward() handles the pooling and classification explicitly.
            self.backbone.classifier = nn.Identity()
            # GAP + GMP → 2 × in_features fed to the improved head
            self.classifier_head = nn.Sequential(
                nn.Linear(in_features * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )

        elif backbone == "resnet50":
            base = models.resnet50(weights=weights)
            self.backbone = base
            in_features = base.fc.in_features  # 2048
            # Remove the built-in FC layer
            self.backbone.fc = nn.Identity()
            self.classifier_head = nn.Sequential(
                nn.Linear(in_features * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )

        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (no sigmoid — use BCEWithLogitsLoss during training).

        Parameters
        ----------
        x : torch.Tensor
            Batch of images, shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Raw logits, shape ``(B, num_classes)``.
        """
        if self._backbone_type == "densenet121":
            # Extract feature maps from DenseNet's convolutional stack
            features = self.backbone.features(x)
            features = F.relu(features, inplace=True)
            gap = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            gmp = F.adaptive_max_pool2d(features, (1, 1)).flatten(1)
            pooled = torch.cat([gap, gmp], dim=1)

        else:  # resnet50 — layer4 output needs manual pooling
            # ResNet's backbone.fc is Identity, so we use the avgpool output
            x_feat = self.backbone.conv1(x)
            x_feat = self.backbone.bn1(x_feat)
            x_feat = self.backbone.relu(x_feat)
            x_feat = self.backbone.maxpool(x_feat)
            x_feat = self.backbone.layer1(x_feat)
            x_feat = self.backbone.layer2(x_feat)
            x_feat = self.backbone.layer3(x_feat)
            x_feat = self.backbone.layer4(x_feat)
            gap = F.adaptive_avg_pool2d(x_feat, (1, 1)).flatten(1)
            gmp = F.adaptive_max_pool2d(x_feat, (1, 1)).flatten(1)
            pooled = torch.cat([gap, gmp], dim=1)

        return self.classifier_head(pooled)
