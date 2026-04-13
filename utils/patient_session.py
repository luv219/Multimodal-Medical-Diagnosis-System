"""
Shared PatientSession dataclass for the diagnostic pipeline.

Each feature in the suite populates its own fields on this object,
which then flows downstream to report generation and history tracking.
"""

import dataclasses
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np


@dataclasses.dataclass
class PatientSession:
    """Container for all data produced during a single diagnostic session."""

    # --- Identity ---
    patient_id: str
    image_path: str
    session_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- Input metadata (Feature 2: Multimodal Fusion) ---
    metadata: Optional[dict] = None  # {"age": 65, "gender": "M", "symptoms": [...]}

    # --- Model outputs ---
    prob_dict: dict = dataclasses.field(default_factory=dict)       # label -> probability
    impression_lines: list = dataclasses.field(default_factory=list)

    # --- Grad-CAM outputs (Feature 1) ---
    heatmaps: dict = dataclasses.field(default_factory=dict)        # label -> 2D np.ndarray
    overlay_paths: dict = dataclasses.field(default_factory=dict)   # label -> file path
    peak_coordinates: dict = dataclasses.field(default_factory=dict)  # label -> (row, col)

    # --- 3D mapping outputs (Feature 3) ---
    findings_3d: dict = dataclasses.field(default_factory=dict)     # label -> (x, y, z)

    # --- Longitudinal outputs (Feature 5) ---
    trend_data: Optional[dict] = None

    # --- Report output (Feature 4) ---
    report_path: Optional[str] = None

    def to_serializable(self) -> dict:
        """Return a JSON-serialisable snapshot (excludes numpy arrays)."""
        return {
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "metadata": self.metadata,
            "prob_dict": self.prob_dict,
            "impression_summary": self.impression_lines[0] if self.impression_lines else "",
            "peak_coordinates": {k: list(v) for k, v in self.peak_coordinates.items()},
            "findings_3d": {k: list(v) for k, v in self.findings_3d.items()},
            "trend_data": self.trend_data,
            "report_path": self.report_path,
        }
