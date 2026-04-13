"""
3D anatomical mapping of 2D chest X-ray findings.

Maps Grad-CAM peak activation coordinates from a 2D posteroanterior (PA)
chest X-ray onto a simplified 3D lung model and renders the result using
PyVista.

Coordinate conventions
----------------------
- **2D image space**: origin at top-left, ``(row, col)`` where row
  increases downward and col increases rightward.
- **3D lung space**: right-handed coordinate system centred on the thorax.
  *x* = patient-left (+) / patient-right (-),
  *y* = superior (+) / inferior (-),
  *z* = anterior (+) / posterior (-).
- **Radiological convention**: the *left* side of a PA chest X-ray
  corresponds to the *patient's right* side.

Depth priors
------------
A single 2D projection cannot determine anterior–posterior depth.
Pathology-specific depth priors are applied instead:

============================  =====  ======================================
Pathology                     z      Rationale
============================  =====  ======================================
Atelectasis                    0.0   Variable location
Cardiomegaly                  −0.1   Anterior / medial structure
Consolidation                  0.1   Can occur anywhere; slight posterior
Edema                          0.0   Diffuse, no clear depth preference
Effusion                       0.2   Gravitationally dependent, posterior
============================  =====  ======================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Depth priors per pathology (z-axis, normalised to [-1, 1])
DEPTH_PRIORS: dict[str, float] = {
    "Atelectasis": 0.0,
    "Cardiomegaly": -0.1,
    "Consolidation": 0.1,
    "Edema": 0.0,
    "Effusion": 0.2,
}


def map_2d_to_3d(
    peak_2d: tuple[int, int],
    image_size: tuple[int, int],
    pathology: str,
) -> tuple[float, float, float]:
    """Map a 2D finding coordinate to simplified 3D lung space.

    Parameters
    ----------
    peak_2d : (row, col)
        Peak activation in original-image pixel coordinates.
    image_size : (height, width)
        Dimensions of the original chest X-ray.
    pathology : str
        Pathology name (used for depth prior lookup).

    Returns
    -------
    (x, y, z)
        Normalised 3D coordinates in the lung coordinate system.
    """
    row, col = peak_2d
    h, w = image_size

    # Normalise to [0, 1]
    norm_x = col / max(w - 1, 1)
    norm_y = row / max(h - 1, 1)

    # Map x: image-left → patient-right (radiological convention)
    # Scale to [-0.8, 0.8]
    x_3d = -(norm_x * 1.6 - 0.8)

    # Map y: top of image → superior (+0.7), bottom → inferior (-0.7)
    y_3d = 0.7 - norm_y * 1.4

    # Depth from pathology prior
    z_3d = DEPTH_PRIORS.get(pathology, 0.0)

    return (float(x_3d), float(y_3d), float(z_3d))


def create_lung_mesh():
    """Create a pair of semi-transparent ellipsoid meshes approximating left and right lungs.

    Returns
    -------
    left_lung, right_lung : pyvista.PolyData
        PyVista meshes positioned in the 3D lung coordinate system.
    """
    import pyvista as pv

    # Right lung (patient right → negative x)
    right_lung = pv.ParametricEllipsoid(0.35, 0.6, 0.25)
    right_lung.translate((-0.45, 0.0, 0.0), inplace=True)

    # Left lung (patient left → positive x), slightly smaller due to cardiac notch
    left_lung = pv.ParametricEllipsoid(0.30, 0.55, 0.25)
    left_lung.translate((0.45, 0.0, 0.0), inplace=True)

    return left_lung, right_lung


def _severity_color(prob: float) -> str:
    """Map a probability to a severity colour."""
    if prob >= 0.6:
        return "red"
    if prob >= 0.3:
        return "orange"
    return "green"


def visualize_3d_findings(
    findings_3d: dict[str, tuple],
    prob_dict: dict[str, float],
    output_path: Optional[str] = None,
    interactive: bool = False,
) -> Optional[str]:
    """Render 3D lung model with finding markers.

    Parameters
    ----------
    findings_3d : dict
        ``{pathology: (x, y, z)}`` from :func:`map_2d_to_3d`.
    prob_dict : dict
        ``{pathology: probability}`` for colour-coding markers.
    output_path : str, optional
        If provided, save a screenshot to this path.
    interactive : bool
        If *True*, open an interactive 3D viewer.

    Returns
    -------
    str or None
        Path to the saved screenshot, or *None* if not saved.
    """
    import pyvista as pv

    left_lung, right_lung = create_lung_mesh()

    plotter = pv.Plotter(off_screen=not interactive)
    plotter.set_background("white")

    # Add lungs
    plotter.add_mesh(left_lung, color="lightskyblue", opacity=0.25, label="Left Lung")
    plotter.add_mesh(right_lung, color="lightskyblue", opacity=0.25, label="Right Lung")

    # Add finding markers
    for pathology, (x, y, z) in findings_3d.items():
        prob = prob_dict.get(pathology, 0.0)
        colour = _severity_color(prob)
        marker = pv.Sphere(radius=0.06, center=(x, y, z))
        plotter.add_mesh(marker, color=colour, label=f"{pathology} ({prob*100:.0f}%)")
        plotter.add_point_labels(
            np.array([[x, y, z + 0.10]]),
            [pathology],
            font_size=10,
            text_color="black",
            shape=None,
        )

    plotter.add_legend()
    plotter.camera_position = "xz"
    plotter.view_isometric()

    saved_path = None
    if output_path:
        plotter.screenshot(output_path)
        saved_path = output_path

    if interactive:
        plotter.show()
    else:
        plotter.close()

    return saved_path
