"""
Late-fusion multimodal model for NIH ChestX-ray14 classification.

Architecture
------------
The model consists of three branches fused via concatenation:

**Image branch** – A DenseNet-121 backbone extracts a 1024-dim feature
vector (after global average pooling).  A projection layer maps it to
``joint_dim`` dimensions.

**Clinical branch** – Patient metadata (age, gender, symptom codes) is
embedded and passed through a small MLP to produce a ``joint_dim``-dim
vector.

**Late fusion** – The two embeddings are *concatenated* (not added) to
form a 2 × ``joint_dim`` vector, preserving modality-specific information.
A decision MLP then maps this to ``num_classes`` raw logits.

Mathematical formulation::

    z_img  = W_proj · AvgPool(DenseNet(x_image)) + b_proj     ∈ ℝ^{joint_dim}
    z_clin = MLP_clin([age_norm ; Emb_gender(g) ; AvgPool(Emb_sym(s))])  ∈ ℝ^{joint_dim}
    z_fused = [z_img ; z_clin]                                 ∈ ℝ^{2·joint_dim}
    ŷ      = W₂ · ReLU(W₁ · z_fused + b₁) + b₂               ∈ ℝ^{num_classes}

The output is **raw logits** (no sigmoid), consistent with
``ImageOnlyModel`` and ``nn.BCEWithLogitsLoss``.

Transfer learning
-----------------
Call :meth:`load_image_branch_weights` with a checkpoint from
``ImageOnlyModel`` to initialise the DenseNet backbone with pre-trained
NIH weights.  Only the ``backbone.features.*`` parameters are copied;
the old classifier head is discarded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ClinicalBranch(nn.Module):
    """Encode patient metadata into a fixed-size embedding.

    Inputs
    ------
    metadata : dict
        - ``"age"``      – ``(B, 1)`` float tensor (raw years; normalised internally).
        - ``"gender"``   – ``(B,)`` long tensor, 0 = female, 1 = male.
        - ``"symptoms"`` – ``(B, max_symptoms)`` long tensor of symptom code indices
          (padded with 0).  Index 0 is treated as a padding / "no symptom" token.
    """

    def __init__(
        self,
        output_dim: int = 64,
        gender_emb_dim: int = 16,
        num_symptom_codes: int = 20,
        symptom_emb_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gender_emb = nn.Embedding(2, gender_emb_dim)
        self.symptom_emb = nn.Embedding(num_symptom_codes, symptom_emb_dim, padding_idx=0)

        input_dim = 1 + gender_emb_dim + symptom_emb_dim  # age + gender + symptoms
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, metadata: dict) -> torch.Tensor:
        age = metadata["age"] / 100.0  # normalise to ~[0, 1]
        gender = self.gender_emb(metadata["gender"])  # (B, gender_emb_dim)

        sym = self.symptom_emb(metadata["symptoms"])   # (B, S, symptom_emb_dim)
        # Mean-pool over the symptom sequence (ignore padding)
        mask = (metadata["symptoms"] != 0).unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1)
        sym_pooled = (sym * mask).sum(dim=1) / denom   # (B, symptom_emb_dim)

        combined = torch.cat([age, gender, sym_pooled], dim=1)
        return self.mlp(combined)


class NIHMultiModalModel(nn.Module):
    """Late-fusion multimodal classifier for NIH ChestX-ray14.

    See module docstring for full architecture description.
    """

    def __init__(
        self,
        num_classes: int = 5,
        joint_dim: int = 64,
        num_symptom_codes: int = 20,
        symptom_emb_dim: int = 32,
        gender_emb_dim: int = 16,
        decision_hidden: int = 128,
        dropout: float = 0.1,
        backbone: str = "densenet121",
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()

        # --- Image branch ---
        weights = "DEFAULT" if use_pretrained else None
        if backbone == "densenet121":
            self.backbone = models.densenet121(weights=weights)
            img_feat_dim = self.backbone.classifier.in_features  # 1024
            self.backbone.classifier = nn.Identity()  # remove original head
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=weights)
            img_feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.image_proj = nn.Linear(img_feat_dim, joint_dim)

        # --- Clinical branch ---
        self.clinical_branch = ClinicalBranch(
            output_dim=joint_dim,
            gender_emb_dim=gender_emb_dim,
            num_symptom_codes=num_symptom_codes,
            symptom_emb_dim=symptom_emb_dim,
            dropout=dropout,
        )

        # --- Decision head (after concatenation) ---
        fused_dim = 2 * joint_dim
        self.decision = nn.Sequential(
            nn.Linear(fused_dim, decision_hidden),
            nn.LayerNorm(decision_hidden),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
            nn.Linear(decision_hidden, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        metadata: dict,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        image : Tensor
            ``(B, 3, H, W)`` pre-processed chest X-ray.
        metadata : dict
            Keys ``"age"`` (B,1), ``"gender"`` (B,), ``"symptoms"`` (B, S).

        Returns
        -------
        Tensor
            ``(B, num_classes)`` raw logits (no sigmoid).
        """
        z_img = self.image_proj(self.backbone(image))       # (B, joint_dim)
        z_clin = self.clinical_branch(metadata)             # (B, joint_dim)
        z_fused = torch.cat([z_img, z_clin], dim=1)         # (B, 2*joint_dim)
        return self.decision(z_fused)

    def load_image_branch_weights(self, image_only_checkpoint: str) -> None:
        """Transfer DenseNet feature weights from a trained ``ImageOnlyModel``.

        Only ``backbone.features.*`` parameters are loaded; the old
        classifier head is discarded.

        Parameters
        ----------
        image_only_checkpoint : str
            Path to a checkpoint saved from ``ImageOnlyModel``
            (i.e. ``results/best_model.pth``).
        """
        state = torch.load(image_only_checkpoint, map_location="cpu", weights_only=True)
        filtered = {
            k: v for k, v in state.items()
            if k.startswith("backbone.features.")
        }
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(
            f"[NIHMultiModalModel] Loaded {len(filtered)} params from image-only "
            f"checkpoint ({len(missing)} missing, {len(unexpected)} unexpected)."
        )
