"""
NIH ChestX-ray14 inference pipeline with integrated diagnostic suite.

Features
--------
1. **GradCAM++** — tighter explainability heatmaps with anatomical localisation scoring.
2. **Medical Device Detection** — flags catheter/drain/lead artifacts and adjusts
   Effusion/Atelectasis confidence scores when heatmaps overlap device regions.
3. **Test-Time Augmentation** — averages 6 augmented passes for ~2% AUC gain.
4. **Anatomical Sanity Checks** — per-pathology localisation quality score (0-1).
5. **3D Anatomical Mapping** — projects findings onto a 3D lung model.
6. **Longitudinal Trend Analysis** — tracks patient history and computes
   a Recovery/Progression Index.
7. **Automated PDF Report** — device warning banner, localisation quality column,
   and confidence calibration guide.

All advanced features degrade gracefully if their dependencies are not installed.
"""

import argparse
import json
import os
import re
import sys

import torch
from PIL import Image
from torchvision import transforms

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.image_only_model import ImageOnlyModel
from utils.clinical_impression import generate_impression
from utils.patient_session import PatientSession


LABEL_NAMES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]
NUM_CLASSES = len(LABEL_NAMES)

# Pre-processing transform (must match val_transform from training)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(
    image_path: str,
    model_path: str = "results/best_model.pth",
    patient_id: str = "UNKNOWN",
    metadata_json: str | None = None,
    no_gradcam: bool = False,
    no_3d: bool = False,
    no_pdf: bool = False,
    no_tta: bool = False,
    history_dir: str = "patient_history",
):
    """Run the full diagnostic pipeline on a single chest X-ray.

    Parameters
    ----------
    image_path : str
        Path to the chest X-ray image.
    model_path : str
        Path to the saved ``ImageOnlyModel`` weights.
    patient_id : str
        Patient identifier for longitudinal tracking.
    metadata_json : str, optional
        JSON string or path to a JSON file with patient metadata
        (keys: ``age``, ``gender``, ``symptoms``).
    no_gradcam : bool
        Skip GradCAM++ heatmap generation.
    no_3d : bool
        Skip 3D anatomical mapping.
    no_pdf : bool
        Skip PDF report generation.
    no_tta : bool
        Skip test-time augmentation (use single forward pass instead).
    history_dir : str
        Directory for per-patient history JSON files.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{os.path.basename(image_path)}'")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found (path: '{os.path.basename(model_path)}'). Please train the model first.")
        return

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading model on {DEVICE}...")
    model = ImageOnlyModel(num_classes=NUM_CLASSES, use_pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # ------------------------------------------------------------------
    # 2. Load image & detect medical devices (Task 2 / 5a)
    # ------------------------------------------------------------------
    try:
        image = Image.open(image_path).convert("RGB")
        img_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Failed to load and transform image: {e}")
        return

    import numpy as np
    image_array = np.array(image)
    device_mask = None
    device_detected = False
    try:
        from utils.device_detector import detect_medical_devices, flag_device_presence
        device_mask = detect_medical_devices(image_array)
        device_detected = flag_device_presence(image_array)
        if device_detected:
            print(
                "\n[WARNING] Medical devices detected in image. "
                "Effusion/Atelectasis predictions will be adjusted."
            )
    except ImportError:
        print("  Device detection skipped (opencv-python not available)")
    except Exception as exc:
        print(f"  Device detection failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Inference (TTA or single-pass)
    # ------------------------------------------------------------------
    print(f"\nAnalyzing '{image_path}'...")
    prob_dict_raw: dict[str, float] = {}

    if not no_tta:
        try:
            from utils.inference_utils import predict_with_tta
            prob_dict_raw = predict_with_tta(
                model, image, INFERENCE_TRANSFORM, LABEL_NAMES,
                n_augments=5, device=DEVICE,
            )
            print(f"  TTA: averaged 6 views")
        except Exception as exc:
            print(f"  TTA failed ({exc}), using single-pass inference")

    if not prob_dict_raw:
        with torch.no_grad(), torch.amp.autocast("cuda" if DEVICE == "cuda" else "cpu"):
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().float().cpu().numpy()
        prob_dict_raw = {name: float(p) for name, p in zip(LABEL_NAMES, probs)}

    results = sorted(prob_dict_raw.items(), key=lambda x: x[1], reverse=True)
    results_filtered = [(name, prob) for name, prob in results if prob * 100 > 10]
    display = results_filtered if results_filtered else results

    print("-" * 30)
    print(f"{'Pathology':<15} | {'Risk / Confidence'}")
    print("-" * 30)
    for name, prob in display:
        print(f"{name:<15} | {prob*100:.2f}%")
    print("-" * 30)

    # ------------------------------------------------------------------
    # 4. Clinical impression (on raw probabilities before device adjustment)
    # ------------------------------------------------------------------
    prob_dict = dict(prob_dict_raw)
    impression_lines = generate_impression(prob_dict)

    print("\n--- CLINICAL IMPRESSION ---")
    for line in impression_lines:
        print(line)
    print("-" * 30)

    # ------------------------------------------------------------------
    # 5. Parse metadata (if provided)
    # ------------------------------------------------------------------
    def _validate_metadata(raw: dict) -> dict:
        validated = {}
        age = raw.get("age")
        if age is not None:
            if not isinstance(age, (int, float)) or not (0 <= age <= 150):
                raise ValueError(f"Invalid age: {age!r}")
            validated["age"] = age
        gender = raw.get("gender")
        if gender is not None:
            if not isinstance(gender, str) or gender not in ("M", "F", "Other", "Unknown"):
                raise ValueError(f"Invalid gender: {gender!r}")
            validated["gender"] = gender
        symptoms = raw.get("symptoms")
        if symptoms is not None:
            if not isinstance(symptoms, list):
                raise ValueError("symptoms must be a list")
            if len(symptoms) > 20:
                raise ValueError("Too many symptoms (max 20)")
            validated["symptoms"] = [str(s)[:100] for s in symptoms if isinstance(s, str)]
        return validated

    metadata = None
    if metadata_json:
        try:
            if os.path.isfile(metadata_json):
                with open(metadata_json) as f:
                    raw_meta = json.load(f)
            else:
                raw_meta = json.loads(metadata_json)
            metadata = _validate_metadata(raw_meta)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"Warning: Could not parse or validate metadata: {e}")

    # ------------------------------------------------------------------
    # 6. Create PatientSession
    # ------------------------------------------------------------------
    session = PatientSession(
        patient_id=patient_id,
        image_path=image_path,
        metadata=metadata,
        prob_dict=prob_dict,
        impression_lines=impression_lines,
        device_detected=device_detected,
    )

    file_prefix = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(os.path.dirname(model_path), f"{file_prefix}_inference")
    os.makedirs(out_dir, mode=0o750, exist_ok=True)

    # ------------------------------------------------------------------
    # 7. GradCAM++ (Task 6) + device-aware confidence adjustment (Task 5a)
    # ------------------------------------------------------------------
    if not no_gradcam:
        try:
            from utils.visualizer import generate_all_heatmaps

            print("\nGenerating GradCAM++ heatmaps...")
            cam_results = generate_all_heatmaps(
                model=model,
                image_path=image_path,
                prob_dict=prob_dict,
                label_names=LABEL_NAMES,
                transform=INFERENCE_TRANSFORM,
                threshold=0.10,
                device=DEVICE,
                output_dir=out_dir,
            )
            session.heatmaps = cam_results["heatmaps"]
            session.overlay_paths = cam_results["overlay_paths"]
            session.peak_coordinates = cam_results["peaks"]
            session.localization_scores = cam_results.get("localization_scores", {})

            # Log localization quality
            for lbl, score in session.localization_scores.items():
                status = "Good" if score >= 0.6 else ("Uncertain" if score >= 0.4 else "MISLOCALIZED")
                print(f"  {lbl}: localisation={score:.2f} ({status})")

            # Device-aware confidence adjustment (Task 5a)
            if device_mask is not None and device_detected:
                from utils.inference_utils import adjust_confidence_for_devices
                session.prob_dict = adjust_confidence_for_devices(
                    session.prob_dict, device_mask, session.heatmaps
                )
                # Rebuild display with adjusted probs
                results = sorted(session.prob_dict.items(), key=lambda x: x[1], reverse=True)
                results_filtered = [(n, p) for n, p in results if p * 100 > 10]
                display = results_filtered if results_filtered else results
                print("  Confidence scores adjusted for device artifacts.")

            print(f"  Saved {len(cam_results['overlay_paths'])} heatmap(s) to {out_dir}/")
        except ImportError:
            print("  Skipping GradCAM++ (missing dependency: opencv-python or grad-cam)")
        except Exception as e:
            print(f"  GradCAM++ failed: {e}")

    # ------------------------------------------------------------------
    # 8. 3D Anatomical Mapping (Feature 3)
    # ------------------------------------------------------------------
    if not no_3d and session.peak_coordinates:
        try:
            from utils.anatomical_mapper import map_2d_to_3d, visualize_3d_findings

            print("\nMapping findings to 3D...")
            orig_size = (image.height, image.width)
            for label, peak in session.peak_coordinates.items():
                session.findings_3d[label] = map_2d_to_3d(peak, orig_size, label)

            render_path = os.path.join(out_dir, "3d_findings.png")
            visualize_3d_findings(
                session.findings_3d,
                prob_dict,
                output_path=render_path,
                interactive=False,
            )
            print(f"  Saved 3D render to {render_path}")
        except ImportError:
            print("  Skipping 3D mapping (missing dependency: pyvista)")
        except Exception as e:
            print(f"  3D mapping failed: {e}")

    # ------------------------------------------------------------------
    # 9. Longitudinal Trend Analysis (Feature 5)
    # ------------------------------------------------------------------
    if patient_id != "UNKNOWN":
        try:
            from utils.history_tracker import (
                generate_trend_chart,
                get_latest_comparison,
                save_session_to_history,
            )

            trend = get_latest_comparison(session, history_dir)
            if trend:
                session.trend_data = trend
                rpi = trend["rpi"]
                direction = trend["direction"]
                print(f"\n--- LONGITUDINAL TREND ---")
                print(f"  RPI: {rpi:+.4f} ({direction.upper()})")
                for label, delta in trend.get("deltas", {}).items():
                    sign = "+" if delta > 0 else ""
                    print(f"  {label}: {sign}{delta*100:.1f}%")

            save_session_to_history(session, history_dir)
            print(f"  Session saved to patient history ({history_dir}/{patient_id}.json)")

            if trend:
                chart_path = generate_trend_chart(
                    patient_id,
                    history_dir,
                    os.path.join(out_dir, f"{patient_id}_trend.png"),
                )
                if chart_path:
                    print(f"  Saved trend chart to {chart_path}")
        except Exception as e:
            print(f"  Trend analysis failed: {e}")

    # ------------------------------------------------------------------
    # 10. PDF Report (Feature 4)
    # ------------------------------------------------------------------
    if not no_pdf:
        try:
            from utils.report_generator import generate_pdf_report

            pdf_path = os.path.join(out_dir, f"{file_prefix}_report.pdf")
            generate_pdf_report(session, pdf_path)
            session.report_path = pdf_path
            print(f"\nSaved PDF report to {pdf_path}")
        except ImportError:
            print("  Skipping PDF report (missing dependency: reportlab)")
        except Exception as e:
            print(f"  PDF report generation failed: {e}")

    # ------------------------------------------------------------------
    # 11. Text report & bar chart (original outputs)
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    report_lines = [
        "========================================",
        f"| Inference Report for {os.path.basename(image_path)}",
        "========================================",
        "",
    ]
    for name, prob in display:
        report_lines.append(f"{name:<20} | {prob*100:.2f}%")

    report_lines.append("")
    report_lines.append("--- CLINICAL IMPRESSION ---")
    report_lines.extend(impression_lines)
    report_lines.append("-" * 30)

    if session.trend_data:
        report_lines.append("")
        report_lines.append("--- LONGITUDINAL TREND ---")
        report_lines.append(f"RPI: {session.trend_data['rpi']:+.4f} ({session.trend_data['direction']})")
        for label, delta in session.trend_data.get("deltas", {}).items():
            sign = "+" if delta > 0 else ""
            report_lines.append(f"  {label}: {sign}{delta*100:.1f}%")

    txt_path = os.path.join(out_dir, f"{file_prefix}_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))

    plt.figure(figsize=(10, 6))
    sorted_names = [n for n, _ in display]
    sorted_probs = [p for _, p in display]
    bars = plt.bar(sorted_names, [p * 100 for p in sorted_probs], color="skyblue")
    plt.ylabel("Confidence (%)")
    plt.title(f"Diagnostic Confidence: {os.path.basename(image_path)}")
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%",
                 ha="center", va="bottom")
    plot_path = os.path.join(out_dir, f"{file_prefix}_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\nSaved inference report and plot to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict diseases from a chest X-ray image.",
    )
    parser.add_argument("image_path", type=str,
                        help="Path to the chest X-ray image (.png/.jpg)")
    parser.add_argument("--model", type=str, default="results/best_model.pth",
                        help="Path to the saved model weights")
    parser.add_argument("--patient-id", type=str, default="UNKNOWN",
                        help="Patient identifier for longitudinal tracking")
    parser.add_argument("--metadata", type=str, default=None,
                        help="JSON string or path to JSON file with patient metadata")
    parser.add_argument("--no-gradcam", action="store_true",
                        help="Skip Grad-CAM heatmap generation")
    parser.add_argument("--no-3d", action="store_true",
                        help="Skip 3D anatomical mapping")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF report generation")
    parser.add_argument("--no-tta", action="store_true",
                        help="Skip test-time augmentation (faster, slightly lower accuracy)")
    parser.add_argument("--history-dir", type=str, default="patient_history",
                        help="Directory for per-patient history files")
    args = parser.parse_args()

    _PATIENT_ID_RE = re.compile(r'^[A-Za-z0-9_\-]{1,64}$')
    if args.patient_id != "UNKNOWN" and not _PATIENT_ID_RE.match(args.patient_id):
        print(f"Error: --patient-id contains invalid characters: {args.patient_id!r}")
        sys.exit(1)
    args.history_dir = os.path.realpath(os.path.abspath(args.history_dir))

    predict(
        image_path=args.image_path,
        model_path=args.model,
        patient_id=args.patient_id,
        metadata_json=args.metadata,
        no_gradcam=args.no_gradcam,
        no_3d=args.no_3d,
        no_pdf=args.no_pdf,
        no_tta=args.no_tta,
        history_dir=args.history_dir,
    )
