"""
Clinical Impression Engine
--------------------------
Analyzes combinations of pathology confidence scores to suggest likely
underlying diseases using rule-based weighted scoring.

Rules are defined for all 14 NIH ChestX-ray pathologies. At runtime,
rules whose required findings are missing from the model output are
automatically skipped. This allows the same engine to work with a 5-class
model today and a 14-class model in the future without code changes.
"""

# ---------------------------------------------------------------------------
# Clinical condition rules
# ---------------------------------------------------------------------------
# mode: "all" = every required finding must exceed its threshold (AND)
#        "any" = at least one required finding must exceed its threshold (OR)

CLINICAL_RULES = [
    # --- Active with current 5-class model ---
    {
        "condition": "Pneumonia Suspected",
        "mode": "any",
        "required": {"Consolidation": 0.30},
        "supporting": {"Effusion": 0.20, "Edema": 0.15},
        "weights": {"Consolidation": 0.60, "Effusion": 0.25, "Edema": 0.15},
        "severity_thresholds": (0.35, 0.55),
        "recommendation": "Clinical correlation and chest CT recommended.",
    },
    {
        "condition": "Congestive Heart Failure (CHF)",
        "mode": "all",
        "required": {"Cardiomegaly": 0.40, "Edema": 0.30},
        "supporting": {"Effusion": 0.20},
        "weights": {"Cardiomegaly": 0.40, "Edema": 0.40, "Effusion": 0.20},
        "severity_thresholds": (0.40, 0.60),
        "recommendation": "Echocardiography and BNP levels recommended.",
    },
    {
        "condition": "Pleural Disease",
        "mode": "any",
        "required": {"Effusion": 0.40},
        "supporting": {"Atelectasis": 0.20},
        "weights": {"Effusion": 0.80, "Atelectasis": 0.20},
        "severity_thresholds": (0.45, 0.65),
        "recommendation": "Lateral decubitus X-ray or ultrasound recommended.",
    },
    {
        "condition": "Lung Collapse / Respiratory Distress",
        "mode": "any",
        "required": {"Atelectasis": 0.40},
        "supporting": {"Effusion": 0.20},
        "weights": {"Atelectasis": 0.80, "Effusion": 0.20},
        "severity_thresholds": (0.45, 0.65),
        "recommendation": "Bronchoscopy or follow-up imaging recommended.",
    },
    {
        "condition": "Pulmonary Edema (non-cardiac)",
        "mode": "any",
        "required": {"Edema": 0.40},
        "supporting": {"Consolidation": 0.15},
        "weights": {"Edema": 0.80, "Consolidation": 0.20},
        "severity_thresholds": (0.45, 0.65),
        "recommendation": "Assess oxygenation and fluid status.",
    },
    # --- Dormant until model is retrained on 14 classes ---
    {
        "condition": "Chronic / Obstructive Lung Disease",
        "mode": "any",
        "required": {"Emphysema": 0.30, "Fibrosis": 0.30},
        "supporting": {},
        "weights": {"Emphysema": 0.50, "Fibrosis": 0.50},
        "severity_thresholds": (0.40, 0.60),
        "recommendation": "Pulmonary function tests recommended.",
    },
    {
        "condition": "Potential Neoplasm",
        "mode": "any",
        "required": {"Mass": 0.30, "Nodule": 0.30},
        "supporting": {},
        "weights": {"Mass": 0.60, "Nodule": 0.40},
        "severity_thresholds": (0.35, 0.55),
        "recommendation": "CT-guided biopsy and oncology referral recommended.",
    },
    {
        "condition": "Pneumothorax",
        "mode": "any",
        "required": {"Pneumothorax": 0.30},
        "supporting": {},
        "weights": {"Pneumothorax": 1.00},
        "severity_thresholds": (0.40, 0.60),
        "recommendation": "Urgent chest tube evaluation if clinically significant.",
    },
    {
        "condition": "Pneumonia (with Infiltration)",
        "mode": "any",
        "required": {"Infiltration": 0.30},
        "supporting": {"Consolidation": 0.20, "Effusion": 0.20},
        "weights": {"Infiltration": 0.50, "Consolidation": 0.30, "Effusion": 0.20},
        "severity_thresholds": (0.35, 0.55),
        "recommendation": "Sputum culture and antibiotic therapy consideration.",
    },
]

# Threshold below which a finding is considered non-significant
SIGNIFICANCE_THRESHOLD = 0.20


def _rule_is_evaluable(rule, available_labels):
    """Check if all *required* pathologies in a rule are available."""
    return all(p in available_labels for p in rule["required"])


def _check_required(rule, prob_dict):
    """Return True if the required-finding condition is satisfied."""
    mode = rule.get("mode", "all")
    if mode == "all":
        return all(
            prob_dict.get(p, 0.0) >= thresh
            for p, thresh in rule["required"].items()
        )
    else:  # "any"
        return any(
            prob_dict.get(p, 0.0) >= thresh
            for p, thresh in rule["required"].items()
        )


def _weighted_score(rule, prob_dict):
    """Compute the weighted confidence score for a matched condition."""
    score = 0.0
    for pathology, weight in rule["weights"].items():
        score += prob_dict.get(pathology, 0.0) * weight
    return score


def _severity_label(score, thresholds):
    """Map a weighted score to a human-readable severity label."""
    low, high = thresholds
    if score >= high:
        return "High"
    elif score >= low:
        return "Moderate"
    return "Low"


def evaluate_condition(prob_dict, rule):
    """Evaluate a single clinical rule against the probability dictionary.

    Returns (condition_name, severity, score, recommendation, detail_dict)
    or None if the rule does not match.
    """
    if not _check_required(rule, prob_dict):
        return None

    score = _weighted_score(rule, prob_dict)
    severity = _severity_label(score, rule["severity_thresholds"])

    # Collect contributing findings for display
    details = {}
    for p in list(rule["required"]) + list(rule.get("supporting", {})):
        val = prob_dict.get(p, 0.0)
        if val > 0.0:
            details[p] = val

    return (rule["condition"], severity, score, rule["recommendation"], details)


def generate_impression(prob_dict):
    """Generate clinical impression lines from a pathology probability dict.

    Parameters
    ----------
    prob_dict : dict[str, float]
        Mapping of pathology name -> probability (0.0 to 1.0).
        Only pathologies present in the model output should be included.

    Returns
    -------
    list[str]
        Formatted lines ready for printing / writing to a report.
    """
    available = set(prob_dict.keys())
    lines = []

    # Evaluate all applicable rules
    matches = []
    for rule in CLINICAL_RULES:
        if not _rule_is_evaluable(rule, available):
            continue
        result = evaluate_condition(prob_dict, rule)
        if result is not None:
            matches.append(result)

    # Sort by weighted score descending
    matches.sort(key=lambda m: m[2], reverse=True)

    if matches:
        for condition, severity, score, recommendation, details in matches:
            lines.append(f"[{severity}] {condition} (weighted score: {score * 100:.1f}%)")
            detail_parts = [f"{p}: {v * 100:.2f}%" for p, v in details.items()]
            lines.append(f"  - {' | '.join(detail_parts)}")
            lines.append(f"  {recommendation}")
            lines.append("")

    # List individually elevated findings not already covered
    covered = set()
    for _, _, _, _, details in matches:
        covered.update(details.keys())

    elevated = [
        (p, v) for p, v in prob_dict.items()
        if v >= SIGNIFICANCE_THRESHOLD and p not in covered
    ]
    elevated.sort(key=lambda x: x[1], reverse=True)

    if elevated:
        lines.append("Elevated individual findings:")
        for p, v in elevated:
            lines.append(f"  - {p} ({v * 100:.2f}%) -- consider clinical correlation")
        lines.append("")

    # No significant findings at all
    if not matches and not elevated:
        any_significant = any(v >= SIGNIFICANCE_THRESHOLD for v in prob_dict.values())
        if not any_significant:
            lines.append("No significant acute findings.")
            lines.append("")

    # Disclaimer
    lines.append("Note: AI-generated impression for decision support only.")
    lines.append("  Not a substitute for professional medical judgment.")

    return lines
