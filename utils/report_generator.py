"""
Automated radiology report generator (PDF).

Produces a formal, multi-section PDF report using ReportLab's *platypus*
layout engine.  The report is populated from a :class:`PatientSession`
object and includes embedded Grad-CAM overlays, 3D anatomical maps, and
longitudinal trend charts when available.

Sections
--------
1. Header & institution banner
2. Medical device warning banner (Task 7a — when devices detected)
3. Patient information
4. Clinical metadata (age, gender, symptoms)
5. Findings table with severity + localization quality columns (Task 7b)
6. Clinical impression (from rule-based engine)
7. Confidence calibration guide (Task 7c)
8. Grad-CAM visualisation (embedded images with localization quality)
9. 3D anatomical map (embedded screenshot)
10. Longitudinal trend chart (if prior sessions exist)
11. Disclaimer
"""

from __future__ import annotations

import os
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _get_styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        "ReportTitle",
        parent=ss["Title"],
        fontSize=18,
        spaceAfter=6 * mm,
    ))
    ss.add(ParagraphStyle(
        "SectionHeading",
        parent=ss["Heading2"],
        fontSize=13,
        spaceBefore=6 * mm,
        spaceAfter=3 * mm,
        textColor=colors.HexColor("#1a5276"),
    ))
    ss.add(ParagraphStyle(
        "BodyText2",
        parent=ss["BodyText"],
        fontSize=10,
        alignment=TA_JUSTIFY,
    ))
    ss.add(ParagraphStyle(
        "Disclaimer",
        parent=ss["BodyText"],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
    ))
    return ss


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _severity_colour(prob: float) -> colors.Color:
    if prob >= 0.6:
        return colors.HexColor("#e74c3c")
    if prob >= 0.3:
        return colors.HexColor("#f39c12")
    return colors.HexColor("#27ae60")


def _localization_colour(score: float) -> colors.Color:
    """Green = well-localised, yellow = uncertain, red = mislocalized."""
    if score >= 0.6:
        return colors.HexColor("#27ae60")
    if score >= 0.4:
        return colors.HexColor("#f39c12")
    return colors.HexColor("#e74c3c")


def _localization_label(score: float) -> str:
    if score >= 0.6:
        return "Good"
    if score >= 0.4:
        return "Uncertain"
    return "Mislocalized"


def _build_device_warning_banner(styles) -> list:
    """Build a prominent warning block for images with detected medical devices (Task 7a)."""
    warning_style = ParagraphStyle(
        "DeviceWarning",
        parent=styles["BodyText"],
        fontSize=10,
        textColor=colors.HexColor("#7d3c00"),
        backColor=colors.HexColor("#fef9e7"),
        borderColor=colors.HexColor("#f39c12"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=4,
        spaceAfter=4,
    )
    text = (
        "<b>WARNING — Medical devices detected.</b> "
        "Chest drains, catheters, pacemaker leads or ECG electrodes were detected "
        "in this image. Effusion and Atelectasis predictions may be affected by "
        "device artifacts. Confidence scores have been adjusted where Grad-CAM "
        "activation overlaps device regions. "
        "<b>Manual review by a qualified radiologist is strongly recommended.</b>"
    )
    return [Paragraph(text, warning_style), Spacer(1, 4)]


def _build_findings_table(
    prob_dict: dict[str, float],
    localization_scores: Optional[dict[str, float]] = None,
) -> Table:
    """Build a formatted findings table with an optional Localisation column (Task 7b)."""
    show_loc = bool(localization_scores)
    header = ["Pathology", "Confidence", "Severity"]
    col_widths = [6 * cm, 3 * cm, 3 * cm]
    if show_loc:
        header.append("Localisation")
        col_widths.append(3 * cm)

    data = [header]
    for label in sorted(prob_dict, key=prob_dict.get, reverse=True):
        prob = prob_dict[label]
        pct = f"{prob * 100:.1f}%"
        severity = "HIGH" if prob >= 0.6 else ("Moderate" if prob >= 0.3 else "Low")
        row = [label, pct, severity]
        if show_loc:
            score = localization_scores.get(label, 1.0)
            row.append(_localization_label(score))
        data.append(row)

    table = Table(data, colWidths=col_widths)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ])

    for i in range(1, len(data)):
        label = data[i][0]
        prob = prob_dict.get(label, 0)
        style.add("TEXTCOLOR", (2, i), (2, i), _severity_colour(prob))
        style.add("FONTNAME", (2, i), (2, i), "Helvetica-Bold")
        if show_loc:
            score = localization_scores.get(label, 1.0)
            style.add("TEXTCOLOR", (3, i), (3, i), _localization_colour(score))
            style.add("FONTNAME", (3, i), (3, i), "Helvetica-Bold")

    table.setStyle(style)
    return table


def _build_calibration_section(styles) -> list:
    """Build the confidence calibration guide section (Task 7c)."""
    elements = []
    elements.append(Paragraph("Confidence Calibration Guide", styles["SectionHeading"]))
    cal_data = [
        ["Confidence Tier", "Range", "Clinical Interpretation"],
        ["Low", "< 40%", "Elevated for monitoring only — low likelihood of pathology."],
        ["Moderate", "40% – 70%", "Warrants clinical correlation — findings may be significant."],
        ["High", "> 70%", "Strong positive signal — further workup is recommended."],
    ]
    cal_table = Table(cal_data, colWidths=[3.5 * cm, 3 * cm, 10 * cm])
    cal_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TEXTCOLOR", (0, 1), (0, 1), colors.HexColor("#27ae60")),
        ("TEXTCOLOR", (0, 2), (0, 2), colors.HexColor("#f39c12")),
        ("TEXTCOLOR", (0, 3), (0, 3), colors.HexColor("#e74c3c")),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ]))
    elements.append(cal_table)
    elements.append(Spacer(1, 4))
    return elements


def _embed_image(
    image_path: str,
    max_width: float = 14 * cm,
    max_height: float = 9 * cm,
) -> Optional[RLImage]:
    """Embed an image scaled to fit within bounds, or return None if missing."""
    if not image_path or not os.path.exists(image_path):
        return None
    return RLImage(image_path, width=max_width, height=max_height, kind="proportional")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_pdf_report(
    session,
    output_path: str,
    institution_name: str = "AI Medical Diagnostics",
    include_gradcam: bool = True,
    include_3d_map: bool = True,
    include_trends: bool = True,
) -> str:
    """Generate a formal multi-section PDF radiology report.

    Parameters
    ----------
    session : PatientSession
        Fully populated session object.
    output_path : str
        Destination path for the PDF file.
    institution_name : str
        Name shown in the report header.
    include_gradcam : bool
        Whether to embed Grad-CAM overlay images.
    include_3d_map : bool
        Whether to embed the 3D anatomical map screenshot.
    include_trends : bool
        Whether to include longitudinal trend data and chart.

    Returns
    -------
    str
        Path to the generated PDF.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    styles = _get_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    story: list = []

    # ---- Header ----
    story.append(Paragraph(institution_name, styles["ReportTitle"]))
    story.append(Paragraph("Automated Chest X-Ray Diagnostic Report", styles["Heading3"]))
    story.append(Spacer(1, 4 * mm))

    # ---- Device warning banner (Task 7a) ----
    if getattr(session, "device_detected", False):
        story.extend(_build_device_warning_banner(styles))
        story.append(Spacer(1, 2 * mm))

    # ---- Patient info ----
    story.append(Paragraph("Patient Information", styles["SectionHeading"]))
    info_data = [
        ["Patient ID", session.patient_id],
        ["Session ID", session.session_id[:8] + "..."],
        ["Timestamp", session.timestamp],
        ["Image", os.path.basename(session.image_path)],
    ]
    if session.metadata:
        if "age" in session.metadata:
            info_data.append(["Age", str(session.metadata["age"])])
        if "gender" in session.metadata:
            info_data.append(["Gender", session.metadata["gender"]])
        if "symptoms" in session.metadata:
            info_data.append(["Symptoms", ", ".join(session.metadata["symptoms"])])

    info_table = Table(info_data, colWidths=[4 * cm, 12 * cm])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 4 * mm))

    # ---- Findings table (Task 7b: includes Localisation column) ----
    story.append(Paragraph("Diagnostic Findings", styles["SectionHeading"]))
    if session.prob_dict:
        loc_scores = getattr(session, "localization_scores", None) or None
        story.append(_build_findings_table(session.prob_dict, loc_scores))
    else:
        story.append(Paragraph("No findings available.", styles["BodyText2"]))
    story.append(Spacer(1, 4 * mm))

    # ---- Clinical impression ----
    story.append(Paragraph("Clinical Impression", styles["SectionHeading"]))
    if session.impression_lines:
        for line in session.impression_lines:
            story.append(Paragraph(line, styles["BodyText2"]))
    else:
        story.append(Paragraph("No clinical impression generated.", styles["BodyText2"]))
    story.append(Spacer(1, 4 * mm))

    # ---- Confidence calibration guide (Task 7c) ----
    story.extend(_build_calibration_section(styles))

    # ---- Grad-CAM overlays ----
    if include_gradcam and session.overlay_paths:
        story.append(Paragraph("Explainability — Grad-CAM Heatmaps", styles["SectionHeading"]))
        story.append(Paragraph(
            "The heatmaps below highlight regions the model focused on "
            "when predicting each pathology. Warmer colours indicate higher "
            "activation.",
            styles["BodyText2"],
        ))
        story.append(Spacer(1, 2 * mm))
        loc_scores = getattr(session, "localization_scores", {}) or {}
        for label, path in session.overlay_paths.items():
            img = _embed_image(path, max_width=12 * cm, max_height=8 * cm)
            if img:
                prob = session.prob_dict.get(label, 0)
                loc_score = loc_scores.get(label)
                if loc_score is not None:
                    loc_text = (
                        f"  |  Localisation: {_localization_label(loc_score)} "
                        f"({loc_score*100:.0f}%)"
                    )
                    if loc_score < 0.4:
                        loc_text += " ⚠ activation outside expected anatomy"
                else:
                    loc_text = ""
                story.append(Paragraph(
                    f"<b>{label}</b> — {prob*100:.1f}% confidence{loc_text}",
                    styles["BodyText2"],
                ))
                story.append(img)
                story.append(Spacer(1, 3 * mm))

    # ---- 3D anatomical map ----
    if include_3d_map and session.findings_3d:
        story.append(Paragraph("3D Anatomical Mapping", styles["SectionHeading"]))
        story.append(Paragraph(
            "Findings are mapped onto a simplified 3D lung model.  Marker "
            "colour indicates severity (green = low, orange = moderate, "
            "red = high).",
            styles["BodyText2"],
        ))
        # Look for the 3D render image in the output directory
        out_dir = os.path.dirname(output_path)
        render_path = os.path.join(out_dir, "3d_findings.png")
        img = _embed_image(render_path)
        if img:
            story.append(img)
        story.append(Spacer(1, 4 * mm))

    # ---- Longitudinal trends ----
    if include_trends and session.trend_data:
        story.append(Paragraph("Longitudinal Trend Analysis", styles["SectionHeading"]))
        td = session.trend_data
        rpi = td.get("rpi", 0)
        direction = td.get("direction", "stable")
        days = td.get("days_between")

        summary = f"Recovery/Progression Index (RPI): <b>{rpi:+.3f}</b> — "
        summary += f"Overall direction: <b>{direction.upper()}</b>."
        if days is not None:
            summary += f"  ({days} days since previous session)"
        story.append(Paragraph(summary, styles["BodyText2"]))

        # Per-pathology deltas
        deltas = td.get("deltas", {})
        if deltas:
            delta_data = [["Pathology", "Change"]]
            for label, delta in sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True):
                sign = "+" if delta > 0 else ""
                delta_data.append([label, f"{sign}{delta*100:.1f}%"])
            dt = Table(delta_data, colWidths=[6 * cm, 4 * cm])
            dt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(dt)

        # Trend chart image
        out_dir = os.path.dirname(output_path)
        chart_path = os.path.join(out_dir, f"{session.patient_id}_trend.png")
        img = _embed_image(chart_path)
        if img:
            story.append(Spacer(1, 2 * mm))
            story.append(img)

        story.append(Spacer(1, 4 * mm))

    # ---- Disclaimer ----
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "DISCLAIMER: This report was generated by an AI-assisted diagnostic "
        "system and is intended for informational purposes only.  It does not "
        "constitute a medical diagnosis.  All findings must be reviewed and "
        "confirmed by a qualified radiologist or physician.",
        styles["Disclaimer"],
    ))

    doc.build(story)
    return output_path
