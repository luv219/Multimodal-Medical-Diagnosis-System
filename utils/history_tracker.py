"""
Longitudinal trend analysis for patient diagnostic history.

Stores per-patient results in JSON files and computes a
Recovery/Progression Index (RPI) when comparing consecutive sessions.

Recovery / Progression Index (RPI)
----------------------------------
For each pathology *p* with current probability *c_p* and previous
probability *p_p*:

.. math::

    \\delta_p = c_p - p_p

    \\text{weighted\\_delta}_p = \\delta_p \\times w_p

    \\text{RPI} = -\\frac{1}{N} \\sum_{p=1}^{N} \\text{weighted\\_delta}_p

where *w_p* is a clinically motivated severity weight and *N* is the
number of pathologies.

========================  ====  =========================================
Interpretation            Sign  Meaning
========================  ====  =========================================
Improving (recovery)       > 0  Weighted probabilities have *decreased*
Stable                     ≈ 0  No significant change
Worsening (progression)    < 0  Weighted probabilities have *increased*
========================  ====  =========================================

Severity weights
~~~~~~~~~~~~~~~~
Higher weights amplify changes in more clinically significant conditions:

==================  =====
Pathology           w_p
==================  =====
Cardiomegaly        1.5
Edema               1.3
Consolidation       1.2
Effusion            1.0
Atelectasis         0.8
==================  =====
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

SEVERITY_WEIGHTS: dict[str, float] = {
    "Atelectasis": 0.8,
    "Cardiomegaly": 1.5,
    "Consolidation": 1.2,
    "Edema": 1.3,
    "Effusion": 1.0,
}


def load_patient_history(
    patient_id: str,
    history_dir: str = "patient_history",
) -> dict:
    """Load a patient's history from their JSON file.

    Returns an empty structure if the file does not exist.
    """
    path = os.path.join(history_dir, f"{patient_id}.json")
    if not os.path.exists(path):
        return {"patient_id": patient_id, "records": []}
    with open(path, "r") as f:
        return json.load(f)


def save_session_to_history(
    session,
    history_dir: str = "patient_history",
) -> None:
    """Append the current session's results to the patient's history file.

    Parameters
    ----------
    session : PatientSession
        The session object (must have ``patient_id``, ``to_serializable()``).
    history_dir : str
        Root directory for patient JSON files.
    """
    os.makedirs(history_dir, exist_ok=True)
    history = load_patient_history(session.patient_id, history_dir)

    record = session.to_serializable()
    history["records"].append(record)

    path = os.path.join(history_dir, f"{session.patient_id}.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def compute_trend(
    current_prob_dict: dict[str, float],
    previous_prob_dict: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict:
    """Compute the Recovery/Progression Index between two sessions.

    See module docstring for full mathematical description.

    Parameters
    ----------
    current_prob_dict : dict
        ``{label: probability}`` from the current inference.
    previous_prob_dict : dict
        ``{label: probability}`` from the most recent prior inference.
    weights : dict, optional
        Per-pathology severity weights.  Defaults to :data:`SEVERITY_WEIGHTS`.

    Returns
    -------
    dict
        ``{"rpi": float, "deltas": {label: float}, "direction": str}``
    """
    if weights is None:
        weights = SEVERITY_WEIGHTS

    common_labels = set(current_prob_dict) & set(previous_prob_dict)
    if not common_labels:
        return {"rpi": 0.0, "deltas": {}, "direction": "stable"}

    deltas = {}
    weighted_sum = 0.0
    for label in common_labels:
        delta = current_prob_dict[label] - previous_prob_dict[label]
        deltas[label] = round(delta, 4)
        weighted_sum += delta * weights.get(label, 1.0)

    rpi = -weighted_sum / len(common_labels)

    if rpi > 0.02:
        direction = "improving"
    elif rpi < -0.02:
        direction = "worsening"
    else:
        direction = "stable"

    return {
        "rpi": round(rpi, 4),
        "deltas": deltas,
        "direction": direction,
    }


def get_latest_comparison(
    session,
    history_dir: str = "patient_history",
) -> Optional[dict]:
    """Compare current session against the most recent prior session.

    Parameters
    ----------
    session : PatientSession
        Current session (uses ``patient_id`` and ``prob_dict``).
    history_dir : str
        Root directory for patient JSON files.

    Returns
    -------
    dict or None
        Trend data including RPI and per-pathology deltas, or *None* if
        no prior history exists.
    """
    history = load_patient_history(session.patient_id, history_dir)
    if not history["records"]:
        return None

    previous = history["records"][-1]
    prev_prob = previous.get("prob_dict", {})
    if not prev_prob:
        return None

    trend = compute_trend(session.prob_dict, prev_prob)
    trend["previous_timestamp"] = previous.get("timestamp", "unknown")

    # Calculate days between sessions
    try:
        prev_dt = datetime.fromisoformat(previous["timestamp"])
        curr_dt = datetime.fromisoformat(session.timestamp)
        trend["days_between"] = (curr_dt - prev_dt).days
    except (ValueError, KeyError):
        trend["days_between"] = None

    return trend


def generate_trend_chart(
    patient_id: str,
    history_dir: str = "patient_history",
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Generate a line chart of pathology probabilities over time.

    Parameters
    ----------
    patient_id : str
        Patient identifier.
    history_dir : str
        Root directory for patient JSON files.
    output_path : str, optional
        Path to save the chart image.  Uses a default name if *None*.

    Returns
    -------
    str or None
        Path to the saved chart, or *None* if insufficient data.
    """
    import matplotlib.pyplot as plt

    history = load_patient_history(patient_id, history_dir)
    records = history.get("records", [])
    if len(records) < 2:
        return None

    # Collect time-series data
    timestamps = []
    series: dict[str, list[float]] = {}
    for rec in records:
        timestamps.append(rec.get("timestamp", "")[:10])  # date portion
        for label, prob in rec.get("prob_dict", {}).items():
            series.setdefault(label, []).append(prob)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values in series.items():
        ax.plot(timestamps[: len(values)], [v * 100 for v in values], marker="o", label=label)

    ax.set_xlabel("Session Date")
    ax.set_ylabel("Confidence (%)")
    ax.set_title(f"Longitudinal Trend — Patient {patient_id}")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(history_dir, f"{patient_id}_trend.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
