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


def rgb_to_unit_quaternion(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB colour to a unit quaternion (notebook convention).

    Maps RGB [0, 255] -> vector part v in [-1, 1]^3, then computes
    w = sqrt(max(1 - |v|^2, 0)) so that (v, w) lies on the unit 3-sphere.

    Parameters
    ----------
    rgb : np.ndarray, shape (3,), dtype uint8 -- (R, G, B) values 0..255

    Returns
    -------
    np.ndarray, shape (4,), dtype float64 -- unit quaternion (x, y, z, w)
    """
    v  = (rgb.astype(np.float64) / 255.0) * 2.0 - 1.0   # [-1, 1]^3
    n2 = float(np.dot(v, v))
    if n2 > 1.0:
        v  /= math.sqrt(n2)
        n2  = 1.0
    w  = math.sqrt(max(1.0 - n2, 0.0))
    q  = np.array([v[0], v[1], v[2], w], dtype=np.float64)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / qn


def load_quaternions_from_csv(csv_path: str, n_solid: int) -> np.ndarray:
    """Load grain quaternions from a CSV file.

    File format: one row per grain, four comma-separated columns x y z w
    (no header, no index column).  Rows correspond to grain gid = 1, 2, ...,
    n_solid in order.  Each quaternion is normalised on load.

    Parameters
    ----------
    csv_path : path to the CSV file
    n_solid  : expected number of solid grains

    Returns
    -------
    np.ndarray, shape (n_solid, 4), dtype float64 -- unit quaternions (x,y,z,w)
    """
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[0] < n_solid:
        raise ValueError(
            f"CSV '{csv_path}' has {data.shape[0]} rows but n_solid={n_solid}.")
    if data.shape[1] != 4:
        raise ValueError(
            f"CSV must have 4 columns (x,y,z,w), got {data.shape[1]}.")
    qs = data[:n_solid].copy()
    for i in range(n_solid):
        norm = np.linalg.norm(qs[i])
        qs[i] = qs[i] / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
    return qs


def assign_quaternions_to_grains(
        n_solid: int,
        mode: str = "random",
        orientation_seed: int = 42,
        csv_path: str = None,
        gid_to_rgb: dict = None,
) -> np.ndarray:
    """Assign unit quaternions to n_solid solid grains.

    Parameters
    ----------
    n_solid          : number of solid grains
    mode             : orientation assignment strategy --
        "random"  -- random unit quaternions seeded by orientation_seed
        "file"    -- load (x,y,z,w) rows from csv_path
        "rgb"     -- deterministic quaternion derived from the grain's RGB colour
    orientation_seed : integer RNG seed (used for "random" and "rgb" modes)
    csv_path         : CSV file path (required when mode="file")
    gid_to_rgb       : dict {gid: (R,G,B)} (required when mode="rgb")

    Returns
    -------
    grain_quaternions : np.ndarray, shape (n_solid + 1, 4), dtype float64
        grain_quaternions[0]      = liquid dummy [0, 0, 0, 1]
        grain_quaternions[1..N-1] = solid grain unit quaternions (x,y,z,w)
    """
    grain_quaternions = np.zeros((n_solid + 1, 4), dtype=np.float64)
    grain_quaternions[0] = np.array([0.0, 0.0, 0.0, 1.0])  # liquid dummy

    if mode == "random":
        rng = np.random.default_rng(orientation_seed)
        for gid in range(1, n_solid + 1):
            v = rng.standard_normal(4)
            grain_quaternions[gid] = v / np.linalg.norm(v)

    elif mode == "file":
        if csv_path is None:
            raise ValueError(
                "orientation_mode='file' requires orientation_csv to be set in config.")
        loaded = load_quaternions_from_csv(csv_path, n_solid)
        grain_quaternions[1:] = loaded

    elif mode == "rgb":
        if gid_to_rgb is None:
            raise ValueError(
                "orientation_mode='rgb' requires a gid_to_rgb colour mapping.")
        for gid in range(1, n_solid + 1):
            rgb = gid_to_rgb.get(gid, (0, 0, 0))
            grain_quaternions[gid] = rgb_to_unit_quaternion(
                np.array(rgb, dtype=np.uint8))

    else:
        raise ValueError(
            f"Unknown orientation_mode: '{mode}'. "
            "Expected 'random', 'file', or 'rgb'.")

    return grain_quaternions


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
