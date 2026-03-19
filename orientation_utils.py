"""
orientation_utils.py
====================
Orientation / quaternion utilities for multi-phase field solidification.

Quaternion convention throughout: SciPy format (x, y, z, w), where w is
the scalar part.  This matches numba CUDA grain_n111 pre-computation and
the original notebook.

Euler angle convention: Rotation.from_euler("xyz", euler_deg, degrees=True)
  — rotate around x-axis first, then y-axis, then z-axis.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation


def build_quaternion_from_config(orientation_cfg: dict) -> np.ndarray:
    """Build a unit quaternion (x, y, z, w) from an orientation config dict.

    Supported orientation_type values
    ----------------------------------
    "euler"
        Euler angles in degrees, rotation order "xyz" (x → y → z).
        Key: ``euler_deg = [angle_x, angle_y, angle_z]`` (degrees).
    "quaternion"
        Direct quaternion components — normalised on input.
        Key: ``quaternion = [x, y, z, w]``.

    Parameters
    ----------
    orientation_cfg : dict
        Sub-dict from config.yaml containing at least ``orientation_type``.
        If the key is absent, "euler" with [0, 0, 0] is assumed (identity).

    Returns
    -------
    np.ndarray, shape (4,), dtype float64
        Unit quaternion in SciPy convention (x, y, z, w).
    """
    otype = orientation_cfg.get("orientation_type", "euler")

    if otype == "euler":
        euler_deg = orientation_cfg.get("euler_deg", [0.0, 0.0, 0.0])
        # Rotation order "xyz": first around x-axis, then y-axis, then z-axis
        rot = Rotation.from_euler("xyz", euler_deg, degrees=True)
        return rot.as_quat()  # (x, y, z, w)

    elif otype == "quaternion":
        q = np.array(orientation_cfg["quaternion"], dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / norm

    else:
        raise ValueError(
            f"Unknown orientation_type: '{otype}'. "
            "Expected 'euler' or 'quaternion'."
        )


def compute_rotated_n111(grain_quaternions: np.ndarray) -> np.ndarray:
    """Compute rotated {111} surface normals for each grain.

    The 8 base normals (±1, ±1, ±1)/√3 are each rotated by the grain's
    quaternion.  The result is used by the CUDA kernels to compute
    cos(θ) between the interface gradient and the nearest {111} facet.

    Parameters
    ----------
    grain_quaternions : np.ndarray, shape (N, 4)
        Unit quaternions in SciPy convention (x, y, z, w) for each grain.
        grain_quaternions[0] is the liquid dummy [0, 0, 0, 1].

    Returns
    -------
    np.ndarray, shape (N, 8, 3), dtype float32
        Rotated {111} normals for every grain.
    """
    N = grain_quaternions.shape[0]

    n111_base = np.array([
        [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [-1,  1,  1],
        [ 1, -1, -1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
    ], dtype=np.float32) / np.sqrt(3.0)

    grain_n111 = np.zeros((N, 8, 3), dtype=np.float32)
    for gid in range(N):
        Rmat = Rotation.from_quat(grain_quaternions[gid]).as_matrix().astype(np.float32)
        grain_n111[gid] = (Rmat @ n111_base.T).T  # shape (8, 3)

    return grain_n111
