from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from numba import cuda
from scipy.spatial.transform import Rotation

from src.gpu_kernels import (
    kernel_update_nfmf,
    kernel_update_phasefield_active,
    kernel_update_temp,
)
from src.gpu_kernels_3d import (
    KMAX as KMAX_3D,
    kernel_update_nfmf_3d_checked,
    kernel_update_phasefield_active_3d_switchable,
    kernel_update_temp_3d,
)
from src.orientation_utils import assign_quaternions_to_grains, compute_rotated_n111
from src.plot_utils import save_phase_map
from src.plot_utils_3d import save_interface_position_3d, save_phase_map_slice_3d
from src.seed_modes import build_interaction_matrices
from src.seed_modes_3d import generate_random_grain_map_3d, init_phi_from_grain_map_3d


@dataclass(frozen=True)
class SimulationParams:
    nx: int
    ny: int
    nz: int
    dx: float
    dt: float
    nsteps: int
    T_melt: float
    G: float
    V_pulling: float
    Sf: float
    delta_factor: float
    gamma_100: float
    gamma_GB: float
    a0_deg: float
    delta_a: float
    mu_a: float
    p_round: float
    ksi: float
    omg_deg: float
    M_SL: float
    M_GB_ratio: float
    MAX_GRAINS: int
    KMAX: int
    threads_per_block: tuple[int, int, int]


@dataclass
class ToggleOverrides:
    anisotropy: int | None = None
    torque: int | None = None


@dataclass
class CaseOutcome:
    name: str
    passed: bool
    summary: str
    failure_reason: str
    metrics: dict[str, Any]
    extra: dict[str, Any]
    case_dir: str
    category: str = "quick"
    level: str = "L0"


class KMaxOverflowError(RuntimeError):
    """Raised when a cell needs more active phases than the kernel supports."""


def default_config_dict() -> dict[str, Any]:
    return {
        "grid": {
            "nx": 256,
            "ny": 256,
            "nz": 256,
            "dx": 1.0e-4,
            "dy": 1.0e-4,
            "dz": 1.0e-4,
            "dt": 5.0e-5,
            "nsteps": 20000,
        },
        "physical": {"T_melt": 1687.0, "G": 1.0e2, "V_pulling": 5.0e-2, "Sf": 2.12e4},
        "interface": {"delta_factor": 6.0, "gamma_100": 0.44, "gamma_GB": 0.60},
        "anisotropy": {
            "a0_deg": 54.7,
            "delta_a": 0.36,
            "mu_a": 0.6156,
            "p_round": 0.05,
            "ksi": 0.60,
            "omg_deg": 10.0,
        },
        "mobility": {"M_SL": 5.0e-5, "M_GB_ratio": 0.07},
        "gpu": {"MAX_GRAINS": 20, "KMAX": int(KMAX_3D), "threads_per_block": [4, 4, 4]},
    }


def load_reference_params() -> tuple[SimulationParams, list[str]]:
    cfg_path = Path("config_3d.yaml")
    notes: list[str] = []
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        notes.append(f"Loaded base parameters from {cfg_path}.")
    else:
        cfg = default_config_dict()
        notes.append("config_3d.yaml was not found; used built-in defaults.")

    config_kmax = int(cfg.get("gpu", {}).get("KMAX", KMAX_3D))
    if config_kmax != KMAX_3D:
        notes.append(
            f"config_3d.yaml gpu.KMAX={config_kmax} differs from compile-time "
            f"KMAX={KMAX_3D}; verification uses compile-time KMAX."
        )

    threads = tuple(int(v) for v in cfg.get("gpu", {}).get("threads_per_block", [4, 4, 4]))
    if len(threads) != 3:
        notes.append("gpu.threads_per_block was not 3D; fell back to (4, 4, 4).")
        threads = (4, 4, 4)

    params = SimulationParams(
        nx=int(cfg["grid"]["nx"]),
        ny=int(cfg["grid"]["ny"]),
        nz=int(cfg["grid"]["nz"]),
        dx=float(cfg["grid"]["dx"]),
        dt=float(cfg["grid"]["dt"]),
        nsteps=int(cfg["grid"]["nsteps"]),
        T_melt=float(cfg["physical"]["T_melt"]),
        G=float(cfg["physical"]["G"]),
        V_pulling=float(cfg["physical"]["V_pulling"]),
        Sf=float(cfg["physical"]["Sf"]),
        delta_factor=float(cfg["interface"]["delta_factor"]),
        gamma_100=float(cfg["interface"]["gamma_100"]),
        gamma_GB=float(cfg["interface"]["gamma_GB"]),
        a0_deg=float(cfg["anisotropy"]["a0_deg"]),
        delta_a=float(cfg["anisotropy"]["delta_a"]),
        mu_a=float(cfg["anisotropy"]["mu_a"]),
        p_round=float(cfg["anisotropy"]["p_round"]),
        ksi=float(cfg["anisotropy"]["ksi"]),
        omg_deg=float(cfg["anisotropy"]["omg_deg"]),
        M_SL=float(cfg["mobility"]["M_SL"]),
        M_GB_ratio=float(cfg["mobility"]["M_GB_ratio"]),
        MAX_GRAINS=int(cfg["gpu"]["MAX_GRAINS"]),
        KMAX=int(KMAX_3D),
        threads_per_block=threads,
    )

    if params.MAX_GRAINS < params.KMAX:
        notes.append(
            f"gpu.MAX_GRAINS={params.MAX_GRAINS} is smaller than compile-time "
            f"KMAX={params.KMAX}; verification allocates APT arrays with depth "
            "max(MAX_GRAINS, KMAX, number_of_grain)."
        )

    return params, notes


def derive_constants(params: SimulationParams) -> dict[str, float]:
    delta = params.delta_factor * params.dx
    a0 = math.radians(params.a0_deg)
    theta_c_rad = math.radians(params.omg_deg)

    def eps_from_gamma(gamma: float) -> float:
        return math.sqrt(8.0 * delta * gamma / (math.pi * math.pi))

    def w_from_gamma(gamma: float) -> float:
        return 4.0 * gamma / delta

    def mij_from_mobility(mobility: float) -> float:
        return (math.pi * math.pi / (8.0 * delta)) * mobility

    m_gb = params.M_SL * params.M_GB_ratio
    return {
        "delta": delta,
        "a0": a0,
        "theta_c_rad": theta_c_rad,
        "eps0_sl": eps_from_gamma(params.gamma_100),
        "w0_sl": w_from_gamma(params.gamma_100),
        "m_sl_phi": mij_from_mobility(params.M_SL),
        "eps_gb": eps_from_gamma(params.gamma_GB),
        "w_gb": w_from_gamma(params.gamma_GB),
        "m_gb_phi": mij_from_mobility(m_gb),
        "g2_floor": (0.1 / params.dx) ** 2,
        "cooling_rate": params.G * params.V_pulling * params.dt,
    }


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, ensure_ascii=False, indent=2)


def constant_temp_field_3d(nx: int, ny: int, nz: int, value: float) -> np.ndarray:
    temp = np.empty((nx, ny, nz), dtype=np.float32)
    temp[:, :, :] = np.float32(value)
    return temp


def constant_temp_field_2d(nx: int, ny: int, value: float) -> np.ndarray:
    temp = np.empty((nx, ny), dtype=np.float32)
    temp[:, :] = np.float32(value)
    return temp


def flat_single_phi_3d(nx: int, ny: int, nz: int, dx: float, delta: float, seed_height: int) -> np.ndarray:
    phi = np.zeros((2, nx, ny, nz), dtype=np.float32)
    factor = np.float32(2.2 / delta)
    z = np.arange(nz, dtype=np.float64)
    dist = (z - seed_height) * dx
    solid_1d = (0.5 * (1.0 - np.tanh(factor * dist))).astype(np.float32)
    phi[1] = solid_1d[np.newaxis, np.newaxis, :]
    phi[0] = 1.0 - phi[1]
    return phi


def wavy_single_phi_3d(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    delta: float,
    base_height: float,
    amplitude_cells: float,
    phase_shift: float = 0.0,
) -> np.ndarray:
    phi = np.zeros((2, nx, ny, nz), dtype=np.float32)
    factor = np.float32(2.2 / delta)
    x = np.arange(nx, dtype=np.float64).reshape(nx, 1, 1)
    z = np.arange(nz, dtype=np.float64).reshape(1, 1, nz)
    height = base_height + amplitude_cells * np.cos(2.0 * math.pi * x / max(nx, 1) + phase_shift)
    dist = (z - height) * dx
    solid = (0.5 * (1.0 - np.tanh(factor * dist))).astype(np.float32)
    phi[1] = np.broadcast_to(solid, (nx, ny, nz))
    phi[0] = 1.0 - phi[1]
    return phi


def wavy_single_phi_2d(
    nx: int,
    ny: int,
    dx: float,
    delta: float,
    base_height: float,
    amplitude_cells: float,
    phase_shift: float = 0.0,
) -> np.ndarray:
    phi = np.zeros((2, nx, ny), dtype=np.float32)
    factor = np.float32(2.2 / delta)
    x = np.arange(nx, dtype=np.float64).reshape(nx, 1)
    y = np.arange(ny, dtype=np.float64).reshape(1, ny)
    height = base_height + amplitude_cells * np.cos(2.0 * math.pi * x / max(nx, 1) + phase_shift)
    dist = (y - height) * dx
    solid = (0.5 * (1.0 - np.tanh(factor * dist))).astype(np.float32)
    phi[1] = solid
    phi[0] = 1.0 - phi[1]
    return phi


def stripe_grain_map(nx: int, ny: int, n_solid: int) -> np.ndarray:
    grain_map = np.zeros((nx, ny), dtype=np.int32)
    edges = np.linspace(0, nx, n_solid + 1, dtype=int)
    for gid in range(1, n_solid + 1):
        grain_map[edges[gid - 1]:edges[gid], :] = gid
    return grain_map


def quaternions_from_rotations(rotations: list[Rotation]) -> np.ndarray:
    q = np.zeros((len(rotations) + 1, 4), dtype=np.float64)
    q[0] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for gid, rot in enumerate(rotations, start=1):
        q[gid] = rot.as_quat()
    return q


def rotation_align_111_to_z() -> Rotation:
    target = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    source = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    rot, _ = Rotation.align_vectors(target, source)
    return rot


def save_scalar_map(path: Path, data: np.ndarray, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data.T, origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_slices_3d(phi: np.ndarray, out_dir: Path, prefix: str, number_of_grain: int) -> None:
    for axis in ("xy", "xz", "yz"):
        save_phase_map_slice_3d(
            phi,
            str(out_dir),
            f"{prefix}_{axis}.png",
            number_of_grain,
            axis=axis,
            title=f"{prefix} ({axis})",
        )
    save_interface_position_3d(phi, str(out_dir), f"{prefix}_interface.png", title=f"{prefix} interface")


def idx_periodic_p(i: int, n: int) -> int:
    return i + 1 if i < n - 1 else 0


def idx_periodic_m(i: int, n: int) -> int:
    return i - 1 if i > 0 else n - 1


def idx_mirror_p(i: int, n: int) -> int:
    return i + 1 if i < n - 1 else n - 1


def idx_mirror_m(i: int, n: int) -> int:
    return i - 1 if i > 0 else 0


def active_phase_ids_at(phi: np.ndarray, l: int, m: int, k: int, number_of_grain: int) -> list[int]:
    nx, ny, nz = phi.shape[1:]
    lp = idx_periodic_p(l, nx)
    lm = idx_periodic_m(l, nx)
    mp = idx_periodic_p(m, ny)
    mm = idx_periodic_m(m, ny)
    kp = idx_mirror_p(k, nz)
    km = idx_mirror_m(k, nz)
    active: list[int] = []
    for gid in range(number_of_grain):
        if (phi[gid, l, m, k] > 0.0) or (
            phi[gid, l, m, k] == 0.0 and (
                (phi[gid, lp, m, k] > 0.0) or
                (phi[gid, lm, m, k] > 0.0) or
                (phi[gid, l, mp, k] > 0.0) or
                (phi[gid, l, mm, k] > 0.0) or
                (phi[gid, l, m, kp] > 0.0) or
                (phi[gid, l, m, km] > 0.0)
            )
        ):
            active.append(gid)
    return active


def cpu_active_count_map_3d(phi: np.ndarray, number_of_grain: int) -> np.ndarray:
    nx, ny, nz = phi.shape[1:]
    nf = np.zeros((nx, ny, nz), dtype=np.int32)
    for l in range(nx):
        for m in range(ny):
            for k in range(nz):
                nf[l, m, k] = len(active_phase_ids_at(phi, l, m, k, number_of_grain))
    return nf


def find_kmax_exceedance_3d(phi: np.ndarray, number_of_grain: int, kmax: int) -> dict[str, Any] | None:
    nx, ny, nz = phi.shape[1:]
    for l in range(nx):
        for m in range(ny):
            for k in range(nz):
                active = active_phase_ids_at(phi, l, m, k, number_of_grain)
                if len(active) > kmax:
                    return {
                        "location": [int(l), int(m), int(k)],
                        "count": int(len(active)),
                        "active_ids_preview": active[: min(len(active), 16)],
                    }
    return None


def interface_positions_3d(phi: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    solid = 1.0 - phi[0]
    mask = solid > threshold
    nx, ny, nz = solid.shape
    has_solid = mask.any(axis=2)
    reverse_idx = np.argmax(mask[:, :, ::-1], axis=2)
    pos = np.where(has_solid, nz - 1 - reverse_idx, np.nan)
    return pos.astype(np.float64)


def interface_profile_xz(phi: np.ndarray, y_index: int | None = None, threshold: float = 0.5) -> np.ndarray:
    solid = 1.0 - phi[0]
    ny = solid.shape[1]
    y_idx = ny // 2 if y_index is None else int(np.clip(y_index, 0, ny - 1))
    slc = solid[:, y_idx, :]
    mask = slc > threshold
    nx, nz = slc.shape
    has_solid = mask.any(axis=1)
    reverse_idx = np.argmax(mask[:, ::-1], axis=1)
    pos = np.where(has_solid, nz - 1 - reverse_idx, np.nan)
    return pos.astype(np.float64)


def groove_angle_from_profile(profile: np.ndarray, dx: float, center_idx: int) -> float:
    if center_idx <= 0 or center_idx >= profile.size - 1:
        return float("nan")
    left = profile[center_idx] - profile[center_idx - 1]
    right = profile[center_idx + 1] - profile[center_idx]
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    left_angle = math.degrees(math.atan(abs(left) / max(dx, 1.0e-30)))
    right_angle = math.degrees(math.atan(abs(right) / max(dx, 1.0e-30)))
    return 0.5 * (left_angle + right_angle)


def interface_curve_from_field_xz(field_xz: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_idx, z_interface) from a 2D field using linear phi=threshold crossing."""
    nx, nz = field_xz.shape
    x_coords = np.arange(nx, dtype=np.float64)
    z_curve = np.full(nx, np.nan, dtype=np.float64)
    for x in range(nx):
        col = field_xz[x]
        above = np.where(col >= threshold)[0]
        if above.size == 0:
            continue
        k0 = int(above[-1])
        if k0 >= nz - 1:
            z_curve[x] = float(nz - 1)
            continue
        v0 = float(col[k0])
        v1 = float(col[k0 + 1])
        if abs(v1 - v0) < 1.0e-12:
            z_curve[x] = float(k0)
        else:
            frac = (threshold - v0) / (v1 - v0)
            z_curve[x] = float(k0) + frac
    return x_coords, z_curve


def solid_interface_curve_xz(phi: np.ndarray, y_index: int | None = None, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    solid = 1.0 - phi[0]
    ny = solid.shape[1]
    y_idx = ny // 2 if y_index is None else int(np.clip(y_index, 0, ny - 1))
    return interface_curve_from_field_xz(solid[:, y_idx, :], threshold=threshold)


def sample_column_at_fractional_z(col: np.ndarray, z_value: float) -> float:
    if not np.isfinite(z_value):
        return float("nan")
    nz = col.shape[0]
    z0 = int(np.floor(z_value))
    z0 = int(np.clip(z0, 0, nz - 1))
    z1 = min(z0 + 1, nz - 1)
    frac = float(z_value - z0)
    return float((1.0 - frac) * col[z0] + frac * col[z1])


def boundary_delta_profile_x(phi: np.ndarray, gid_left: int, gid_right: int, y_index: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_coords, z_curve = solid_interface_curve_xz(phi, y_index=y_index)
    ny = phi.shape[2]
    y_idx = ny // 2 if y_index is None else int(np.clip(y_index, 0, ny - 1))
    delta = np.full_like(x_coords, np.nan, dtype=np.float64)
    for x in range(phi.shape[1]):
        delta[x] = sample_column_at_fractional_z(
            phi[gid_left, x, y_idx, :] - phi[gid_right, x, y_idx, :],
            z_curve[x],
        )
    return x_coords, z_curve, delta


def interpolate_root_from_sign_change(
    x_coords: np.ndarray,
    z_curve: np.ndarray,
    delta: np.ndarray,
    x_guess: float | None = None,
) -> tuple[float, float]:
    valid_pairs: list[tuple[float, float]] = []
    for i in range(len(x_coords) - 1):
        d0 = delta[i]
        d1 = delta[i + 1]
        if not (np.isfinite(d0) and np.isfinite(d1)):
            continue
        if d0 == 0.0:
            xr = float(x_coords[i])
            zr = float(z_curve[i])
            valid_pairs.append((xr, zr))
            continue
        if d0 * d1 > 0.0:
            continue
        frac = abs(d0) / max(abs(d0) + abs(d1), 1.0e-12)
        xr = float(x_coords[i] + frac * (x_coords[i + 1] - x_coords[i]))
        z0 = z_curve[i]
        z1 = z_curve[i + 1]
        zr = float((1.0 - frac) * z0 + frac * z1) if (np.isfinite(z0) and np.isfinite(z1)) else float("nan")
        valid_pairs.append((xr, zr))
    if not valid_pairs:
        return float("nan"), float("nan")
    if x_guess is None:
        return valid_pairs[len(valid_pairs) // 2]
    return min(valid_pairs, key=lambda pair: abs(pair[0] - x_guess))


def grain_boundary_root_and_trijunction(
    phi: np.ndarray,
    gid_left: int = 1,
    gid_right: int = 2,
    y_index: int | None = None,
    x_guess: float | None = None,
) -> dict[str, float]:
    x_coords, z_curve, delta = boundary_delta_profile_x(phi, gid_left, gid_right, y_index=y_index)
    root_x, root_z = interpolate_root_from_sign_change(x_coords, z_curve, delta, x_guess=x_guess)
    return {
        "root_x_cells": float(root_x),
        "root_z_cells": float(root_z),
        "trijunction_x_cells": float(root_x),
        "trijunction_z_cells": float(root_z),
    }


def fit_interface_line(
    x_coords: np.ndarray,
    z_curve: np.ndarray,
    x_min: float,
    x_max: float,
) -> dict[str, float]:
    mask = np.isfinite(z_curve) & (x_coords >= x_min) & (x_coords <= x_max)
    xs = x_coords[mask]
    zs = z_curve[mask]
    if xs.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "angle_deg": float("nan"), "rmse": float("nan")}
    slope, intercept = np.polyfit(xs, zs, deg=1)
    fit = slope * xs + intercept
    rmse = float(np.sqrt(np.mean((fit - zs) ** 2)))
    angle_deg = float(math.degrees(math.atan(float(slope))))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "angle_deg": angle_deg,
        "rmse": rmse,
    }


def robust_groove_metrics(
    phi: np.ndarray,
    root_x_guess: float,
    y_index: int | None = None,
    fit_half_width: float = 4.0,
    exclude_half_width: float = 1.0,
) -> dict[str, float]:
    x_coords, z_curve = solid_interface_curve_xz(phi, y_index=y_index)
    root = grain_boundary_root_and_trijunction(phi, 1, 2, y_index=y_index, x_guess=root_x_guess)
    root_x = root["root_x_cells"]
    root_z = root["root_z_cells"]
    if not np.isfinite(root_x):
        root_x = float(root_x_guess)
        if 0 <= int(np.clip(round(root_x), 0, len(z_curve) - 1)) < len(z_curve):
            root_z = float(z_curve[int(np.clip(round(root_x), 0, len(z_curve) - 1))])

    left_fit = fit_interface_line(x_coords, z_curve, root_x - fit_half_width, root_x - exclude_half_width)
    right_fit = fit_interface_line(x_coords, z_curve, root_x + exclude_half_width, root_x + fit_half_width)

    far_mask = np.isfinite(z_curve) & ((x_coords <= root_x - fit_half_width) | (x_coords >= root_x + fit_half_width))
    far_field = float(np.nanmean(z_curve[far_mask])) if np.any(far_mask) else float("nan")
    groove_depth_cells = float(max(far_field - root_z, 0.0)) if (np.isfinite(far_field) and np.isfinite(root_z)) else float("nan")
    if np.isfinite(left_fit["angle_deg"]) and np.isfinite(right_fit["angle_deg"]):
        dihedral = float(abs(right_fit["angle_deg"] - left_fit["angle_deg"]))
    else:
        dihedral = float("nan")

    return {
        "root_x_cells": float(root_x),
        "root_z_cells": float(root_z),
        "trijunction_x_cells": float(root_x),
        "trijunction_z_cells": float(root_z),
        "left_angle_deg": left_fit["angle_deg"],
        "right_angle_deg": right_fit["angle_deg"],
        "left_fit_rmse": left_fit["rmse"],
        "right_fit_rmse": right_fit["rmse"],
        "groove_depth_cells": groove_depth_cells,
        "dihedral_angle_like_deg": dihedral,
    }


def compute_metrics_3d(
    phi: np.ndarray,
    nf: np.ndarray,
    dx: float,
    runtime_s: float,
    interface_angle_deg: float = float("nan"),
) -> dict[str, Any]:
    phi_sum = phi.sum(axis=0)
    grain_volumes = [float(phi[gid].sum() * (dx ** 3)) for gid in range(1, phi.shape[0])]
    volume_threshold = dx ** 3
    interface = interface_positions_3d(phi)
    finite = interface[np.isfinite(interface)]
    if finite.size:
        mean_z = float(np.mean(finite) * dx)
        std_z = float(np.std(finite) * dx)
    else:
        mean_z = float("nan")
        std_z = float("nan")
    return {
        "phi_sum_error_max": float(np.max(np.abs(phi_sum - 1.0))),
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "nf_max": int(np.max(nf)) if nf.size else 0,
        "nan_count": int(np.isnan(phi).sum()),
        "solid_fraction": float(np.mean(1.0 - phi[0])),
        "solid_volume": float(np.sum(1.0 - phi[0]) * (dx ** 3)),
        "grain_volumes": grain_volumes,
        "surviving_grains": int(sum(v > volume_threshold for v in grain_volumes)),
        "interface_position_mean": mean_z,
        "interface_position_max": float(np.nanmax(finite) * dx) if finite.size else float("nan"),
        "interface_position_std": std_z,
        "interface_roughness": std_z,
        "interface_angle_deg": float(interface_angle_deg),
        "solid_volume_growth": float("nan"),
        "winner_loser_volume_ratio": float("nan"),
        "grain_boundary_root_shift": float("nan"),
        "trijunction_x": float("nan"),
        "trijunction_z": float("nan"),
        "dihedral_angle_like_metric": float("nan"),
        "angle_deviation_from_final_state": float("nan"),
        "front_velocity": float("nan"),
        "runtime_s": float(runtime_s),
    }


def finalize_metrics(
    metrics: dict[str, Any],
    phi_initial: np.ndarray,
    phi_final: np.ndarray,
    dx: float,
    total_time: float,
    root_initial_cells: float | None = None,
    trijunction_final: dict[str, float] | None = None,
    groove_metrics: dict[str, float] | None = None,
    angle_history_deg: list[float] | None = None,
) -> dict[str, Any]:
    initial_interface = interface_positions_3d(phi_initial)
    initial_finite = initial_interface[np.isfinite(initial_interface)]
    initial_mean = float(np.mean(initial_finite) * dx) if initial_finite.size else float("nan")
    metrics["solid_volume_growth"] = float(np.sum((1.0 - phi_final[0]) - (1.0 - phi_initial[0])) * (dx ** 3))
    if np.isfinite(metrics.get("interface_position_mean", float("nan"))) and np.isfinite(initial_mean):
        metrics["front_velocity"] = float((metrics["interface_position_mean"] - initial_mean) / max(total_time, 1.0e-30))

    positive = [float(v) for v in metrics.get("grain_volumes", []) if float(v) > 0.5 * (dx ** 3)]
    if len(positive) >= 2:
        metrics["winner_loser_volume_ratio"] = float(max(positive) / max(min(positive), 1.0e-30))

    if trijunction_final is not None:
        metrics["trijunction_x"] = float(trijunction_final.get("trijunction_x_cells", float("nan")) * dx)
        metrics["trijunction_z"] = float(trijunction_final.get("trijunction_z_cells", float("nan")) * dx)
        if root_initial_cells is not None and np.isfinite(root_initial_cells):
            metrics["grain_boundary_root_shift"] = float(
                (trijunction_final.get("root_x_cells", float("nan")) - root_initial_cells) * dx
            )

    if groove_metrics is not None:
        metrics["interface_angle_deg"] = float(groove_metrics.get("dihedral_angle_like_deg", float("nan")))
        metrics["dihedral_angle_like_metric"] = float(groove_metrics.get("dihedral_angle_like_deg", float("nan")))
        metrics["trijunction_x"] = float(groove_metrics.get("trijunction_x_cells", float("nan")) * dx)
        metrics["trijunction_z"] = float(groove_metrics.get("trijunction_z_cells", float("nan")) * dx)
        if root_initial_cells is not None and np.isfinite(root_initial_cells):
            metrics["grain_boundary_root_shift"] = float(
                (groove_metrics.get("root_x_cells", float("nan")) - root_initial_cells) * dx
            )

    if angle_history_deg:
        finite_angles = [float(a) for a in angle_history_deg if np.isfinite(a)]
        if finite_angles and np.isfinite(metrics.get("interface_angle_deg", float("nan"))):
            final_angle = float(metrics["interface_angle_deg"])
            metrics["angle_deviation_from_final_state"] = float(
                np.mean(np.abs(np.array(finite_angles, dtype=np.float64) - final_angle))
            )

    return metrics


def blank_metrics() -> dict[str, Any]:
    return {
        "phi_sum_error_max": float("nan"),
        "phi_min": float("nan"),
        "phi_max": float("nan"),
        "nf_max": -1,
        "nan_count": -1,
        "solid_fraction": float("nan"),
        "solid_volume": float("nan"),
        "grain_volumes": [],
        "surviving_grains": -1,
        "interface_position_mean": float("nan"),
        "interface_position_max": float("nan"),
        "interface_position_std": float("nan"),
        "interface_roughness": float("nan"),
        "interface_angle_deg": float("nan"),
        "solid_volume_growth": float("nan"),
        "winner_loser_volume_ratio": float("nan"),
        "grain_boundary_root_shift": float("nan"),
        "trijunction_x": float("nan"),
        "trijunction_z": float("nan"),
        "dihedral_angle_like_metric": float("nan"),
        "angle_deviation_from_final_state": float("nan"),
        "front_velocity": float("nan"),
        "runtime_s": float("nan"),
    }


def run_checked_apt_only(label: str, out_dir: Path, params: SimulationParams, phi: np.ndarray) -> dict[str, Any]:
    number_of_grain = phi.shape[0]
    nx, ny, nz = phi.shape[1:]
    threads = params.threads_per_block
    blocks = (
        math.ceil(nx / threads[0]),
        math.ceil(ny / threads[1]),
        math.ceil(nz / threads[2]),
    )
    apt_depth = max(params.MAX_GRAINS, params.KMAX, number_of_grain)

    save_slices_3d(phi, out_dir, f"{label}_field", number_of_grain)

    d_phi = cuda.to_device(phi.astype(np.float32))
    d_mf = cuda.to_device(np.zeros((apt_depth, nx, ny, nz), dtype=np.int32))
    d_nf = cuda.to_device(np.zeros((nx, ny, nz), dtype=np.int32))
    d_status = cuda.to_device(np.zeros(2, dtype=np.int32))

    t0 = time.perf_counter()
    kernel_update_nfmf_3d_checked[blocks, threads](d_phi, d_mf, d_nf, d_status, nx, ny, nz, number_of_grain)
    cuda.synchronize()
    runtime_s = time.perf_counter() - t0

    mf = d_mf.copy_to_host()
    nf = d_nf.copy_to_host()
    status = d_status.copy_to_host()
    metrics = compute_metrics_3d(phi, nf, params.dx, runtime_s)
    metrics["nf_max"] = max(metrics["nf_max"], int(status[1]))
    write_json(
        out_dir / f"{label}_apt.json",
        {"status": status, "metrics": metrics, "number_of_grain": number_of_grain},
    )
    return {"mf": mf, "nf": nf, "status": status, "metrics": metrics}


def run_3d_simulation(
    label: str,
    out_dir: Path,
    params: SimulationParams,
    phi0: np.ndarray,
    temp0: np.ndarray,
    grain_quaternions: np.ndarray,
    enable_anisotropy: bool,
    enable_torque: bool,
    sample_every: int | None = None,
    observer: Callable[[np.ndarray, int], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    number_of_grain = phi0.shape[0]
    nx, ny, nz = phi0.shape[1:]
    if number_of_grain != grain_quaternions.shape[0]:
        raise ValueError("phi0 phase count and quaternion count do not match.")

    derived = derive_constants(params)
    grain_n111 = compute_rotated_n111(grain_quaternions)
    wij, aij, mij = build_interaction_matrices(
        number_of_grain,
        derived["eps0_sl"],
        derived["w0_sl"],
        derived["m_sl_phi"],
        derived["eps_gb"],
        derived["w_gb"],
        derived["m_gb_phi"],
    )

    threads = params.threads_per_block
    blocks = (
        math.ceil(nx / threads[0]),
        math.ceil(ny / threads[1]),
        math.ceil(nz / threads[2]),
    )
    apt_depth = max(params.MAX_GRAINS, params.KMAX, number_of_grain)

    save_slices_3d(phi0, out_dir, f"{label}_initial", number_of_grain)
    initial_nf = cpu_active_count_map_3d(phi0, number_of_grain)
    initial_metrics = compute_metrics_3d(phi0, initial_nf, params.dx, 0.0)

    d_phi = cuda.to_device(phi0.astype(np.float32))
    d_phi_new = cuda.to_device(phi0.astype(np.float32).copy())
    d_temp = cuda.to_device(temp0.astype(np.float32))
    d_mf = cuda.to_device(np.zeros((apt_depth, nx, ny, nz), dtype=np.int32))
    d_nf = cuda.to_device(np.zeros((nx, ny, nz), dtype=np.int32))
    d_status = cuda.to_device(np.zeros(2, dtype=np.int32))
    d_wij = cuda.to_device(wij.astype(np.float32))
    d_aij = cuda.to_device(aij.astype(np.float32))
    d_mij = cuda.to_device(mij.astype(np.float32))
    d_n111 = cuda.to_device(grain_n111.astype(np.float32))

    nf_max_seen = 0
    t0 = time.perf_counter()
    samples: list[dict[str, Any]] = []

    for step in range(1, params.nsteps + 1):
        if derived["cooling_rate"] != 0.0:
            kernel_update_temp_3d[blocks, threads](
                d_temp,
                np.float32(derived["cooling_rate"]),
                nx,
                ny,
                nz,
            )

        d_status.copy_to_device(np.zeros(2, dtype=np.int32))
        kernel_update_nfmf_3d_checked[blocks, threads](
            d_phi, d_mf, d_nf, d_status, nx, ny, nz, number_of_grain
        )
        cuda.synchronize()
        status = d_status.copy_to_host()
        nf_max_seen = max(nf_max_seen, int(status[1]))
        if int(status[0]) != 0:
            phi_host = d_phi.copy_to_host()
            info = find_kmax_exceedance_3d(phi_host, number_of_grain, params.KMAX)
            raise KMaxOverflowError(
                f"KMAX overflow in {label}: max active phases {int(status[1])}, details={info}"
            )

        kernel_update_phasefield_active_3d_switchable[blocks, threads](
            d_phi,
            d_phi_new,
            d_temp,
            d_mf,
            d_nf,
            d_wij,
            d_aij,
            d_mij,
            d_n111,
            nx,
            ny,
            nz,
            number_of_grain,
            np.float32(params.dx),
            np.float32(params.dt),
            np.float32(params.T_melt),
            np.float32(params.Sf),
            np.float32(derived["eps0_sl"]),
            np.float32(derived["w0_sl"]),
            np.float32(derived["a0"]),
            np.float32(params.delta_a),
            np.float32(params.mu_a),
            np.float32(params.p_round),
            np.float32(derived["g2_floor"]),
            np.float32(params.ksi),
            np.float32(derived["theta_c_rad"]),
            np.int32(1 if enable_anisotropy else 0),
            np.int32(1 if enable_torque else 0),
        )
        cuda.synchronize()
        d_phi, d_phi_new = d_phi_new, d_phi

        if observer is not None and sample_every is not None and (step % sample_every == 0 or step == params.nsteps):
            phi_sample = d_phi.copy_to_host()
            obs = observer(phi_sample, step)
            if obs is not None:
                sample_record = {"step": int(step)}
                sample_record.update(obs)
                samples.append(sample_record)

    d_status.copy_to_device(np.zeros(2, dtype=np.int32))
    kernel_update_nfmf_3d_checked[blocks, threads](d_phi, d_mf, d_nf, d_status, nx, ny, nz, number_of_grain)
    cuda.synchronize()
    status = d_status.copy_to_host()
    nf_max_seen = max(nf_max_seen, int(status[1]))
    if int(status[0]) != 0:
        phi_host = d_phi.copy_to_host()
        info = find_kmax_exceedance_3d(phi_host, number_of_grain, params.KMAX)
        raise KMaxOverflowError(
            f"KMAX overflow in final APT of {label}: max active phases {int(status[1])}, details={info}"
        )

    runtime_s = time.perf_counter() - t0
    phi = d_phi.copy_to_host()
    temp = d_temp.copy_to_host()
    nf = d_nf.copy_to_host()

    save_slices_3d(phi, out_dir, f"{label}_final", number_of_grain)
    metrics = compute_metrics_3d(phi, nf, params.dx, runtime_s)
    metrics["nf_max"] = max(metrics["nf_max"], nf_max_seen)

    write_json(
        out_dir / f"{label}_run.json",
        {
            "label": label,
            "params": asdict(params),
            "enable_anisotropy": bool(enable_anisotropy),
            "enable_torque": bool(enable_torque),
            "initial_metrics": initial_metrics,
            "final_metrics": metrics,
            "grain_quaternions": grain_quaternions.tolist(),
        },
    )
    return {
        "phi": phi,
        "temp": temp,
        "nf": nf,
        "metrics": metrics,
        "initial_metrics": initial_metrics,
        "number_of_grain": number_of_grain,
        "samples": samples,
    }


def run_2d_simulation(
    label: str,
    out_dir: Path,
    params: SimulationParams,
    phi0: np.ndarray,
    temp0: np.ndarray,
    grain_quaternions: np.ndarray,
) -> dict[str, Any]:
    number_of_grain = phi0.shape[0]
    nx, ny = phi0.shape[1:]
    derived = derive_constants(params)
    grain_n111 = compute_rotated_n111(grain_quaternions)
    wij, aij, mij = build_interaction_matrices(
        number_of_grain,
        derived["eps0_sl"],
        derived["w0_sl"],
        derived["m_sl_phi"],
        derived["eps_gb"],
        derived["w_gb"],
        derived["m_gb_phi"],
    )

    threads = (max(1, params.threads_per_block[0]), max(1, params.threads_per_block[1]))
    blocks = (math.ceil(nx / threads[0]), math.ceil(ny / threads[1]))
    apt_depth = max(params.MAX_GRAINS, number_of_grain, 18)

    save_phase_map(phi0, str(out_dir), f"{label}_initial.png", number_of_grain, title=f"{label} initial")

    d_phi = cuda.to_device(phi0.astype(np.float32))
    d_phi_new = cuda.to_device(phi0.astype(np.float32).copy())
    d_temp = cuda.to_device(temp0.astype(np.float32))
    d_mf = cuda.to_device(np.zeros((apt_depth, nx, ny), dtype=np.int32))
    d_nf = cuda.to_device(np.zeros((nx, ny), dtype=np.int32))
    d_wij = cuda.to_device(wij.astype(np.float32))
    d_aij = cuda.to_device(aij.astype(np.float32))
    d_mij = cuda.to_device(mij.astype(np.float32))
    d_n111 = cuda.to_device(grain_n111.astype(np.float32))

    t0 = time.perf_counter()
    for _ in range(params.nsteps):
        if derived["cooling_rate"] != 0.0:
            kernel_update_temp[blocks, threads](d_temp, np.float32(derived["cooling_rate"]), nx, ny)
        kernel_update_nfmf[blocks, threads](d_phi, d_mf, d_nf, nx, ny, number_of_grain)
        kernel_update_phasefield_active[blocks, threads](
            d_phi,
            d_phi_new,
            d_temp,
            d_mf,
            d_nf,
            d_wij,
            d_aij,
            d_mij,
            d_n111,
            nx,
            ny,
            number_of_grain,
            np.float32(params.dx),
            np.float32(params.dt),
            np.float32(params.T_melt),
            np.float32(params.Sf),
            np.float32(derived["eps0_sl"]),
            np.float32(derived["w0_sl"]),
            np.float32(derived["a0"]),
            np.float32(params.delta_a),
            np.float32(params.mu_a),
            np.float32(params.p_round),
            np.float32(derived["g2_floor"]),
            np.float32(params.ksi),
            np.float32(derived["theta_c_rad"]),
        )
        cuda.synchronize()
        d_phi, d_phi_new = d_phi_new, d_phi

    runtime_s = time.perf_counter() - t0
    phi = d_phi.copy_to_host()
    save_phase_map(phi, str(out_dir), f"{label}_final.png", number_of_grain, title=f"{label} final")
    write_json(
        out_dir / f"{label}_run.json",
        {"params": asdict(params), "runtime_s": runtime_s, "grain_quaternions": grain_quaternions.tolist()},
    )
    return {"phi": phi, "runtime_s": runtime_s}


def resolve_toggles(default_anisotropy: bool, default_torque: bool, overrides: ToggleOverrides) -> tuple[bool, bool]:
    anisotropy = default_anisotropy if overrides.anisotropy is None else bool(overrides.anisotropy)
    torque = default_torque if overrides.torque is None else bool(overrides.torque)
    if not anisotropy:
        torque = False
    return anisotropy, torque


def classify_signal_level(signal: float, l1_threshold: float, l2_threshold: float, passed: bool) -> str:
    if not passed or not np.isfinite(signal):
        return "L0"
    if signal >= l2_threshold:
        return "L2"
    if signal >= l1_threshold:
        return "L1"
    return "L0"


def write_case_csv(case_dir: Path, payload: dict[str, Any]) -> None:
    with (case_dir / "case_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in payload.items():
            writer.writerow([key, json.dumps(to_serializable(value), ensure_ascii=False)])


def write_case_markdown(
    case_dir: Path,
    name: str,
    category: str,
    level: str,
    passed: bool,
    summary: str,
    failure_reason: str,
    metrics: dict[str, Any],
    extra: dict[str, Any],
) -> None:
    lines = [
        f"# {name}",
        "",
        f"- Category: {category}",
        f"- Status: {'PASS' if passed else 'FAIL'}",
        f"- Physical level: {level}",
        f"- Summary: {summary}",
    ]
    if failure_reason:
        lines.append(f"- Failure reason: {failure_reason}")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
        ]
    )
    for key, value in metrics.items():
        lines.append(f"- {key}: `{json.dumps(to_serializable(value), ensure_ascii=False)}`")
    if extra:
        lines.extend(["", "## Extra", ""])
        for key, value in extra.items():
            lines.append(f"- {key}: `{json.dumps(to_serializable(value), ensure_ascii=False)}`")
    (case_dir / "case_report.md").write_text("\n".join(lines), encoding="utf-8")


def bicrystal_phi_3d(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    delta: float,
    seed_height: float,
    split_index: int,
    left_gid: int = 1,
    right_gid: int = 2,
    left_offset: float = 0.0,
    right_offset: float = 0.0,
    waviness: float = 0.0,
) -> np.ndarray:
    phi = np.zeros((3, nx, ny, nz), dtype=np.float32)
    factor = np.float32(2.2 / delta)
    x = np.arange(nx, dtype=np.float64).reshape(nx, 1, 1)
    z = np.arange(nz, dtype=np.float64).reshape(1, 1, nz)
    base = seed_height + waviness * np.cos(2.0 * math.pi * x / max(nx, 1))
    left_height = base + left_offset
    right_height = base + right_offset
    left_solid = (0.5 * (1.0 - np.tanh(factor * ((z - left_height) * dx)))).astype(np.float32)
    right_solid = (0.5 * (1.0 - np.tanh(factor * ((z - right_height) * dx)))).astype(np.float32)
    phi[left_gid, :split_index, :, :] = np.broadcast_to(left_solid[:split_index], (split_index, ny, nz))
    phi[right_gid, split_index:, :, :] = np.broadcast_to(right_solid[split_index:], (nx - split_index, ny, nz))
    phi[0] = np.clip(1.0 - phi[1] - phi[2], 0.0, 1.0)
    return phi


def alternating_multigrain_phi_3d(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    delta: float,
    seed_height: float,
    ngrains: int,
    waviness: float = 0.0,
) -> np.ndarray:
    grain_map = stripe_grain_map(nx, ny, ngrains)
    phi = init_phi_from_grain_map_3d(grain_map, ngrains, nx, ny, nz, dx, delta, int(seed_height))
    if waviness == 0.0:
        return phi
    factor = np.float32(2.2 / delta)
    x = np.arange(nx, dtype=np.float64).reshape(nx, 1, 1)
    z = np.arange(nz, dtype=np.float64).reshape(1, 1, nz)
    base = seed_height + waviness * np.cos(2.0 * math.pi * x / max(nx, 1))
    solid_profile = (0.5 * (1.0 - np.tanh(factor * ((z - base) * dx)))).astype(np.float32)
    for gid in range(1, ngrains + 1):
        mask = grain_map == gid
        phi[gid] = np.where(mask[:, :, np.newaxis], solid_profile, 0.0)
    phi[0] = np.clip(1.0 - phi[1:].sum(axis=0), 0.0, 1.0)
    return phi


def orientation_library() -> dict[str, Rotation]:
    return {
        "identity": Rotation.identity(),
        "y_15": Rotation.from_euler("y", 15.0, degrees=True),
        "y_30": Rotation.from_euler("y", 30.0, degrees=True),
        "y_45": Rotation.from_euler("y", 45.0, degrees=True),
        "x_30": Rotation.from_euler("x", 30.0, degrees=True),
        "tilt_xyz": Rotation.from_euler("xyz", [20.0, 35.0, 10.0], degrees=True),
        "aligned_111_to_z": rotation_align_111_to_z(),
    }


SCAN_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def simulation_signature(params: SimulationParams, extra: tuple[Any, ...] = ()) -> tuple[Any, ...]:
    return tuple(asdict(params).values()) + extra


def build_directional_scan_case(
    base: SimulationParams,
    delta_a: float,
    ksi: float,
    nsteps: int,
) -> SimulationParams:
    return replace(base, nx=16, ny=6, nz=20, nsteps=nsteps, delta_a=delta_a, ksi=ksi, p_round=0.02, G=0.0, V_pulling=0.0)


def single_grain_benchmark_phi(params: SimulationParams) -> np.ndarray:
    derived = derive_constants(params)
    return wavy_single_phi_3d(params.nx, params.ny, params.nz, params.dx, derived["delta"], 6.5, 1.8, phase_shift=0.3)


def run_directional_scan(
    case_dir: Path,
    params: SimulationParams,
    undercoolings: dict[str, float],
    enable_anisotropy: bool,
    enable_torque: bool,
    include_isotropic_baseline: bool = False,
    artifact_prefix: str | None = None,
) -> dict[str, Any]:
    cache_key = simulation_signature(params, extra=(tuple(sorted(undercoolings.items())), enable_anisotropy, enable_torque, include_isotropic_baseline))
    if cache_key in SCAN_CACHE:
        result = SCAN_CACHE[cache_key]
    else:
        candidates = orientation_library()
        rows: list[dict[str, Any]] = []
        phi0 = single_grain_benchmark_phi(params)
        for driving_label, undercooling in undercoolings.items():
            temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt - undercooling)
            for orientation_name, rot in candidates.items():
                quats = quaternions_from_rotations([rot])
                run = run_3d_simulation(
                    f"scan_{driving_label}_{orientation_name}",
                    case_dir,
                    params,
                    phi0,
                    temp0,
                    quats,
                    enable_anisotropy,
                    enable_torque,
                )
                metrics = finalize_metrics(
                    dict(run["metrics"]),
                    phi0,
                    run["phi"],
                    params.dx,
                    params.dt * params.nsteps,
                )
                rows.append(
                    {
                        "orientation": orientation_name,
                        "driving_label": driving_label,
                        "undercooling": undercooling,
                        "interface_mean": metrics["interface_position_mean"],
                        "interface_max": metrics["interface_position_max"],
                        "solid_volume_growth": metrics["solid_volume_growth"],
                        "front_velocity": metrics["front_velocity"],
                        "runtime_s": metrics["runtime_s"],
                    }
                )

            if include_isotropic_baseline:
                for orientation_name, rot in candidates.items():
                    quats = quaternions_from_rotations([rot])
                    run = run_3d_simulation(
                        f"scan_iso_{driving_label}_{orientation_name}",
                        case_dir,
                        replace(params, delta_a=0.0, ksi=1.0),
                        phi0,
                        temp0,
                        quats,
                        False,
                        False,
                    )
                    metrics = finalize_metrics(
                        dict(run["metrics"]),
                        phi0,
                        run["phi"],
                        params.dx,
                        params.dt * params.nsteps,
                    )
                    rows.append(
                        {
                            "orientation": orientation_name,
                            "driving_label": f"{driving_label}_iso",
                            "undercooling": undercooling,
                            "interface_mean": metrics["interface_position_mean"],
                            "interface_max": metrics["interface_position_max"],
                            "solid_volume_growth": metrics["solid_volume_growth"],
                            "front_velocity": metrics["front_velocity"],
                            "runtime_s": metrics["runtime_s"],
                        }
                    )

        result = {"rows": rows}
        SCAN_CACHE[cache_key] = result

    rows = result["rows"]
    prefix = f"{artifact_prefix}_" if artifact_prefix else ""
    with (case_dir / f"{prefix}directional_scan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = list(undercoolings.keys())
    orientation_names = list(orientation_library().keys())
    heatmap = np.full((len(labels), len(orientation_names)), np.nan, dtype=np.float64)
    for i, label in enumerate(labels):
        for j, orientation_name in enumerate(orientation_names):
            match = [
                row["front_velocity"]
                for row in rows
                if row["driving_label"] == label and row["orientation"] == orientation_name
            ]
            if match:
                heatmap[i, j] = float(match[0])
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(orientation_names)))
    ax.set_xticklabels(orientation_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Directional preference map (front velocity)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="front velocity [m/s]")
    plt.tight_layout()
    plt.savefig(case_dir / f"{prefix}directional_preference_map.png", dpi=150)
    plt.close(fig)
    return result


def directional_rows(rows: list[dict[str, Any]], driving_label: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["driving_label"] == driving_label]


def directional_stats(rows: list[dict[str, Any]], driving_label: str) -> dict[str, Any]:
    subset = directional_rows(rows, driving_label)
    if not subset:
        raise RuntimeError(f"No directional-scan rows were found for driving label '{driving_label}'.")
    best = max(subset, key=lambda row: row["front_velocity"])
    worst = min(subset, key=lambda row: row["front_velocity"])
    return {
        "rows": subset,
        "best": best,
        "worst": worst,
        "spread": float(best["front_velocity"] - worst["front_velocity"]),
    }


def directional_row_by_orientation(
    rows: list[dict[str, Any]],
    driving_label: str,
    orientation_name: str,
) -> dict[str, Any]:
    for row in rows:
        if row["driving_label"] == driving_label and row["orientation"] == orientation_name:
            return row
    raise RuntimeError(
        f"No directional-scan row found for driving label '{driving_label}' and orientation '{orientation_name}'."
    )


def select_switch_pair(scan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = {}
    for row in scan_rows:
        label = row["driving_label"]
        if label.endswith("_iso"):
            continue
        grouped.setdefault(row["orientation"], {})[label] = float(row["front_velocity"])
    names = sorted(grouped.keys())
    best_pair: dict[str, Any] | None = None
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            if "low" not in grouped[name_a] or "high" not in grouped[name_a]:
                continue
            if "low" not in grouped[name_b] or "high" not in grouped[name_b]:
                continue
            low_diff = grouped[name_a]["low"] - grouped[name_b]["low"]
            high_diff = grouped[name_a]["high"] - grouped[name_b]["high"]
            switch_strength = abs(low_diff) + abs(high_diff)
            switched = low_diff * high_diff < 0.0
            candidate = {
                "orientation_a": name_a,
                "orientation_b": name_b,
                "low_diff": low_diff,
                "high_diff": high_diff,
                "switch_strength": switch_strength,
                "switched": switched,
            }
            if best_pair is None:
                best_pair = candidate
            elif (candidate["switched"], candidate["switch_strength"]) > (best_pair["switched"], best_pair["switch_strength"]):
                best_pair = candidate
    if best_pair is None:
        raise RuntimeError("Could not select an orientation pair from the directional scan.")

    best_pair["low_winner"] = best_pair["orientation_a"] if best_pair["low_diff"] > 0.0 else best_pair["orientation_b"]
    best_pair["high_winner"] = best_pair["orientation_a"] if best_pair["high_diff"] > 0.0 else best_pair["orientation_b"]
    return best_pair


def orientation_quaternion_by_name(name: str) -> np.ndarray:
    return quaternions_from_rotations([orientation_library()[name]])


def bicrystal_group_metrics(phi: np.ndarray, family_a: list[int], family_b: list[int], dx: float) -> dict[str, float]:
    vols = np.array([float(phi[gid].sum() * (dx ** 3)) for gid in range(1, phi.shape[0])], dtype=np.float64)
    vol_a = float(np.sum([vols[gid - 1] for gid in family_a]))
    vol_b = float(np.sum([vols[gid - 1] for gid in family_b]))
    if vol_a >= vol_b:
        winner = "A"
        ratio = vol_a / max(vol_b, 1.0e-30)
    else:
        winner = "B"
        ratio = vol_b / max(vol_a, 1.0e-30)
    return {
        "volume_a": vol_a,
        "volume_b": vol_b,
        "winner_loser_volume_ratio": float(ratio),
        "winner_family": winner,
    }


def make_case_result(
    name: str,
    passed: bool,
    summary: str,
    failure_reason: str,
    metrics: dict[str, Any],
    extra: dict[str, Any],
    case_dir: Path,
    category: str = "quick",
    level: str = "L0",
) -> CaseOutcome:
    write_case_csv(
        case_dir,
        {
            "name": name,
            "category": category,
            "passed": passed,
            "level": level,
            **metrics,
            **{f"extra_{k}": v for k, v in extra.items()},
        },
    )
    write_json(
        case_dir / "case_result.json",
        {
            "name": name,
            "passed": passed,
            "category": category,
            "level": level,
            "summary": summary,
            "failure_reason": failure_reason,
            "metrics": metrics,
            "extra": extra,
        },
    )
    write_case_markdown(case_dir, name, category, level, passed, summary, failure_reason, metrics, extra)
    return CaseOutcome(
        name=name,
        passed=passed,
        summary=summary,
        failure_reason=failure_reason,
        metrics=metrics,
        extra=extra,
        case_dir=str(case_dir),
        category=category,
        level=level,
    )


def case_basic_constraints(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = replace(base, nx=12, ny=8, nz=14, nsteps=6)
    derived = derive_constants(params)
    grain_map = generate_random_grain_map_3d(params.nx, params.ny, 4, random_seed=17)
    phi0 = init_phi_from_grain_map_3d(grain_map, 4, params.nx, params.ny, params.nz, params.dx, derived["delta"], 4)
    temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt - 5.0)
    quats = assign_quaternions_to_grains(4, mode="random", orientation_seed=23)
    enable_anisotropy, enable_torque = resolve_toggles(True, True, overrides)
    run = run_3d_simulation("basic_constraints", case_dir, params, phi0, temp0, quats, enable_anisotropy, enable_torque)
    metrics = run["metrics"]
    passed = (
        metrics["nan_count"] == 0 and
        metrics["phi_min"] >= -1.0e-5 and
        metrics["phi_max"] <= 1.0 + 1.0e-5 and
        metrics["phi_sum_error_max"] <= 5.0e-4 and
        metrics["nf_max"] <= params.KMAX
    )
    summary = (
        f"Short multi-grain run stayed bounded with phi_sum_error_max={metrics['phi_sum_error_max']:.2e} "
        f"and nf_max={metrics['nf_max']}."
    )
    failure_reason = "" if passed else "Constraint violation detected in short random-mode run."
    extra = {"enable_anisotropy": enable_anisotropy, "enable_torque": enable_torque}
    return make_case_result("basic_constraints", passed, summary, failure_reason, metrics, extra, case_dir)


def case_kmax_overflow_detection(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    _ = overrides
    params = replace(base, nx=4, ny=4, nz=4, nsteps=0)
    number_of_grain = params.KMAX + 2
    phi = np.zeros((number_of_grain, params.nx, params.ny, params.nz), dtype=np.float32)
    phi[:, 1, 1, 1] = np.float32(1.0 / number_of_grain)
    phi[0] = np.clip(1.0 - phi[1:].sum(axis=0), 0.0, 1.0)
    apt = run_checked_apt_only("kmax_overflow", case_dir, params, phi)
    overflow_info = find_kmax_exceedance_3d(phi, number_of_grain, params.KMAX)
    detected = int(apt["status"][0]) == 1 and overflow_info is not None
    metrics = apt["metrics"]
    summary = (
        f"KMAX guard {'detected' if detected else 'missed'} an overflow request of "
        f"{overflow_info['count'] if overflow_info else 'unknown'} active phases."
    )
    failure_reason = "" if detected else "Checked APT kernel did not report the crafted KMAX overflow."
    extra = {
        "status": apt["status"].tolist(),
        "overflow_info": overflow_info,
        "requested_number_of_grain": number_of_grain,
    }
    return make_case_result("kmax_overflow_detection", detected, summary, failure_reason, metrics, extra, case_dir)


def case_static_flat_interface(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = replace(base, nx=12, ny=6, nz=14, nsteps=6)
    derived = derive_constants(params)
    phi0 = flat_single_phi_3d(params.nx, params.ny, params.nz, params.dx, derived["delta"], 5)
    temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt)
    quats = quaternions_from_rotations([Rotation.identity()])
    enable_anisotropy, enable_torque = resolve_toggles(False, False, overrides)
    run = run_3d_simulation("static_flat", case_dir, params, phi0, temp0, quats, enable_anisotropy, enable_torque)
    metrics = run["metrics"]
    initial = run["initial_metrics"]
    mean_shift = abs(metrics["interface_position_mean"] - initial["interface_position_mean"])
    roughness = metrics["interface_roughness"]
    passed = mean_shift <= 1.05 * params.dx and roughness <= 0.15 * params.dx
    summary = (
        f"Flat interface mean shift={mean_shift:.2e} m, roughness={roughness:.2e} m "
        f"with anisotropy={enable_anisotropy}, torque={enable_torque}."
    )
    failure_reason = "" if passed else "Planar interface drifted or roughened more than the static threshold."
    extra = {
        "initial_interface_mean": initial["interface_position_mean"],
        "final_interface_mean": metrics["interface_position_mean"],
        "mean_shift": mean_shift,
    }
    return make_case_result("static_flat_interface", passed, summary, failure_reason, metrics, extra, case_dir)


def case_two_d_limit_consistency(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params3d = replace(base, nx=12, ny=4, nz=14, nsteps=6, delta_a=0.0, ksi=1.0)
    derived = derive_constants(params3d)
    phi3d_0 = wavy_single_phi_3d(params3d.nx, params3d.ny, params3d.nz, params3d.dx, derived["delta"], 4.8, 1.2)
    temp3d_0 = constant_temp_field_3d(params3d.nx, params3d.ny, params3d.nz, params3d.T_melt - 4.0)
    quats = quaternions_from_rotations([Rotation.identity()])
    enable_anisotropy, enable_torque = resolve_toggles(False, False, overrides)
    run3d = run_3d_simulation("two_d_limit_3d", case_dir, params3d, phi3d_0, temp3d_0, quats, enable_anisotropy, enable_torque)

    phi2d_0 = wavy_single_phi_2d(params3d.nx, params3d.nz, params3d.dx, derived["delta"], 4.8, 1.2)
    temp2d_0 = constant_temp_field_2d(params3d.nx, params3d.nz, params3d.T_melt - 4.0)
    run2d = run_2d_simulation("two_d_limit_2d", case_dir, params3d, phi2d_0, temp2d_0, quats)

    y_mid = params3d.ny // 2
    phi3d_slice = run3d["phi"][1, :, y_mid, :]
    phi2d = run2d["phi"][1]
    diff = phi3d_slice - phi2d
    linf = float(np.max(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff * diff)))
    save_scalar_map(case_dir / "two_d_limit_difference.png", diff, "3D mid-y minus 2D solid fraction", cmap="coolwarm")

    metrics = run3d["metrics"]
    passed = linf <= 2.0e-3 and l2 <= 8.0e-4
    summary = f"3D mid-y slice matched 2D reference with L_inf={linf:.2e}, L2={l2:.2e}."
    failure_reason = "" if passed else "3D y-uniform evolution diverged from the 2D reference beyond tolerance."
    extra = {"phi_linf_vs_2d": linf, "phi_l2_vs_2d": l2}
    return make_case_result("two_d_limit_consistency", passed, summary, failure_reason, metrics, extra, case_dir)


def case_boundary_conditions(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    _ = overrides
    params = replace(base, nx=6, ny=5, nz=5, nsteps=0)
    phi = np.zeros((4, params.nx, params.ny, params.nz), dtype=np.float32)
    phi[1, 0, 2, 2] = 1.0
    phi[2, 3, 0, 2] = 1.0
    phi[3, 3, 2, 0] = 1.0
    phi[0] = np.clip(1.0 - phi[1:].sum(axis=0), 0.0, 1.0)
    apt = run_checked_apt_only("boundary_conditions", case_dir, params, phi)
    mf = apt["mf"]
    nf = apt["nf"]

    def has_phase(cell: tuple[int, int, int], gid: int) -> bool:
        l, m, k = cell
        return gid in mf[:nf[l, m, k], l, m, k].tolist()

    checks = {
        "x_periodic_wrap": has_phase((params.nx - 1, 2, 2), 1),
        "y_periodic_wrap": has_phase((3, params.ny - 1, 2), 2),
        "z_mirror_adjacent": has_phase((3, 2, 1), 3),
        "z_no_wrap": not has_phase((3, 2, params.nz - 1), 3),
    }
    metrics = apt["metrics"]
    passed = all(checks.values())
    summary = "Boundary-condition probes: " + ", ".join(f"{k}={v}" for k, v in checks.items())
    failure_reason = "" if passed else "At least one x/y periodic or z-Neumann boundary probe failed."
    extra = {"checks": checks}
    return make_case_result("boundary_conditions", passed, summary, failure_reason, metrics, extra, case_dir)


def case_isotropic_orientation_independence(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = replace(base, nx=12, ny=6, nz=14, nsteps=6, delta_a=0.0, ksi=1.0)
    derived = derive_constants(params)
    phi0 = flat_single_phi_3d(params.nx, params.ny, params.nz, params.dx, derived["delta"], 5)
    temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt - 6.0)
    enable_anisotropy, enable_torque = resolve_toggles(False, False, overrides)

    q_a = quaternions_from_rotations([Rotation.identity()])
    q_b = quaternions_from_rotations([rotation_align_111_to_z()])
    run_a = run_3d_simulation("isotropic_orientation_a", case_dir, params, phi0, temp0, q_a, enable_anisotropy, enable_torque)
    run_b = run_3d_simulation("isotropic_orientation_b", case_dir, params, phi0, temp0, q_b, enable_anisotropy, enable_torque)

    mean_diff = abs(run_a["metrics"]["interface_position_mean"] - run_b["metrics"]["interface_position_mean"])
    solid_diff = abs(run_a["metrics"]["solid_fraction"] - run_b["metrics"]["solid_fraction"])
    phi_diff = run_a["phi"][1] - run_b["phi"][1]
    l2 = float(np.sqrt(np.mean(phi_diff * phi_diff)))
    save_scalar_map(case_dir / "isotropic_orientation_difference_xz.png", phi_diff[:, params.ny // 2, :], "Isotropic orientation diff", cmap="coolwarm")

    metrics = run_a["metrics"]
    passed = mean_diff <= 0.15 * params.dx and solid_diff <= 2.0e-4 and l2 <= 2.0e-3
    summary = (
        f"Isotropic orientation sensitivity stayed small: mean_diff={mean_diff:.2e} m, "
        f"solid_diff={solid_diff:.2e}, L2={l2:.2e}."
    )
    failure_reason = "" if passed else "Orientation changed the isotropic solution more than the allowed tolerance."
    extra = {"interface_mean_diff": mean_diff, "solid_fraction_diff": solid_diff, "phi_l2_diff": l2}
    return make_case_result("isotropic_orientation_independence", passed, summary, failure_reason, metrics, extra, case_dir)


def case_anisotropic_preferred_growth(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = replace(base, nx=16, ny=6, nz=18, nsteps=12, delta_a=0.55, p_round=0.02)
    derived = derive_constants(params)
    phi0 = wavy_single_phi_3d(params.nx, params.ny, params.nz, params.dx, derived["delta"], 6.0, 2.0, phase_shift=0.2)
    temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt)
    enable_anisotropy, enable_torque = resolve_toggles(True, False, overrides)

    q_identity = quaternions_from_rotations([Rotation.identity()])
    q_aligned = quaternions_from_rotations([rotation_align_111_to_z()])
    run_identity = run_3d_simulation("anisotropic_growth_identity", case_dir, params, phi0, temp0, q_identity, enable_anisotropy, enable_torque)
    run_aligned = run_3d_simulation("anisotropic_growth_aligned111", case_dir, params, phi0, temp0, q_aligned, enable_anisotropy, enable_torque)

    mean_identity = run_identity["metrics"]["interface_position_mean"]
    mean_aligned = run_aligned["metrics"]["interface_position_mean"]
    delta_mean = abs(mean_identity - mean_aligned)
    roughness_diff = abs(run_identity["metrics"]["interface_roughness"] - run_aligned["metrics"]["interface_roughness"])
    phi_diff = run_identity["phi"][1] - run_aligned["phi"][1]
    l2 = float(np.sqrt(np.mean(phi_diff * phi_diff)))
    save_scalar_map(case_dir / "anisotropic_orientation_difference_xz.png", phi_diff[:, params.ny // 2, :], "Identity - aligned111", cmap="coolwarm")
    preferred = "identity" if mean_identity > mean_aligned else "aligned111"

    metrics = run_identity["metrics"]
    passed = delta_mean >= 0.10 * params.dx or roughness_diff >= 0.05 * params.dx or l2 >= 5.0e-5
    summary = (
        f"Anisotropy created orientation-dependent evolution: |delta mean|={delta_mean:.2e} m, "
        f"roughness_diff={roughness_diff:.2e} m, L2={l2:.2e}; {preferred} advanced further."
    )
    failure_reason = "" if passed else "Orientation dependence was too weak to confirm anisotropic preference."
    extra = {
        "interface_mean_identity": mean_identity,
        "interface_mean_aligned111": mean_aligned,
        "interface_mean_abs_diff": delta_mean,
        "roughness_diff": roughness_diff,
        "phi_l2_diff": l2,
        "preferred_orientation": preferred,
        "enable_torque": enable_torque,
    }
    return make_case_result("anisotropic_preferred_growth", passed, summary, failure_reason, metrics, extra, case_dir)


def case_torque_term_contribution(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = replace(base, nx=16, ny=6, nz=18, nsteps=12, delta_a=0.55, p_round=0.02)
    derived = derive_constants(params)
    phi0 = wavy_single_phi_3d(params.nx, params.ny, params.nz, params.dx, derived["delta"], 6.0, 2.0, phase_shift=0.3)
    temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt)
    rotation = Rotation.from_euler("xyz", [20.0, 35.0, 10.0], degrees=True)
    quats = quaternions_from_rotations([rotation])

    aniso_on, _ = resolve_toggles(True, False, overrides)
    torque_on = True if overrides.torque is None else bool(overrides.torque)
    run_off = run_3d_simulation("torque_off", case_dir, params, phi0, temp0, quats, aniso_on, False)
    run_on = run_3d_simulation("torque_on", case_dir, params, phi0, temp0, quats, aniso_on, torque_on)

    diff = run_on["phi"][1] - run_off["phi"][1]
    l2 = float(np.sqrt(np.mean(diff * diff)))
    roughness_diff = abs(run_on["metrics"]["interface_roughness"] - run_off["metrics"]["interface_roughness"])
    save_scalar_map(case_dir / "torque_difference_xz.png", diff[:, params.ny // 2, :], "Torque on - off", cmap="coolwarm")

    metrics = run_on["metrics"]
    passed = l2 >= 5.0e-5 or roughness_diff >= 0.05 * params.dx
    summary = f"Torque changed the solution by L2={l2:.2e}, roughness_diff={roughness_diff:.2e} m."
    failure_reason = "" if passed else "Torque on/off produced no measurable difference in the chosen anisotropic case."
    extra = {"phi_l2_diff": l2, "roughness_diff": roughness_diff, "anisotropy_enabled": aniso_on}
    return make_case_result("torque_term_contribution", passed, summary, failure_reason, metrics, extra, case_dir)


def case_grain_competition(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = replace(base, nx=12, ny=12, nz=16, nsteps=8)
    derived = derive_constants(params)
    grain_map = generate_random_grain_map_3d(params.nx, params.ny, 6, random_seed=31)
    phi0 = init_phi_from_grain_map_3d(grain_map, 6, params.nx, params.ny, params.nz, params.dx, derived["delta"], 5)
    temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt - 5.0)
    quats = assign_quaternions_to_grains(6, mode="random", orientation_seed=11)
    enable_anisotropy, enable_torque = resolve_toggles(True, True, overrides)
    run = run_3d_simulation("grain_competition", case_dir, params, phi0, temp0, quats, enable_anisotropy, enable_torque)

    initial_volumes = np.array(run["initial_metrics"]["grain_volumes"], dtype=np.float64)
    final_volumes = np.array(run["metrics"]["grain_volumes"], dtype=np.float64)
    volume_change = float(np.max(np.abs(final_volumes - initial_volumes)))
    top_phase_initial = np.unique(np.argmax(phi0, axis=0)[:, :, -1])
    top_phase_final = np.unique(np.argmax(run["phi"], axis=0)[:, :, -1])
    surviving_initial = run["initial_metrics"]["surviving_grains"]
    surviving_final = run["metrics"]["surviving_grains"]

    metrics = run["metrics"]
    passed = surviving_final <= surviving_initial and volume_change >= 2.0e-14
    summary = (
        f"Competition changed grain volumes by max {volume_change:.2e} m^3 and "
        f"surviving grains {surviving_initial}->{surviving_final}."
    )
    failure_reason = "" if passed else "Short multi-grain run did not show enough measurable competition."
    extra = {
        "max_volume_change": volume_change,
        "top_surface_phase_count_initial": int(np.sum(top_phase_initial > 0)),
        "top_surface_phase_count_final": int(np.sum(top_phase_final > 0)),
    }
    return make_case_result("grain_competition", passed, summary, failure_reason, metrics, extra, case_dir)


def case_grain_boundary_groove(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = replace(base, nx=18, ny=6, nz=18, nsteps=10, delta_a=0.0, ksi=1.0)
    derived = derive_constants(params)
    grain_map = stripe_grain_map(params.nx, params.ny, 3)
    phi0 = init_phi_from_grain_map_3d(grain_map, 3, params.nx, params.ny, params.nz, params.dx, derived["delta"], 6)
    temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt)
    quats = quaternions_from_rotations([Rotation.identity(), Rotation.identity(), Rotation.identity()])
    enable_anisotropy, enable_torque = resolve_toggles(False, False, overrides)
    run = run_3d_simulation("grain_boundary_groove", case_dir, params, phi0, temp0, quats, enable_anisotropy, enable_torque)

    profile = interface_profile_xz(run["phi"])
    split1 = params.nx // 3
    split2 = 2 * params.nx // 3
    far_field = np.nanmean(np.concatenate([profile[1:split1 - 1], profile[split1 + 1:split2 - 1], profile[split2 + 1:-1]]))
    local1 = np.nanmin(profile[max(split1 - 1, 0):min(split1 + 2, profile.size)])
    local2 = np.nanmin(profile[max(split2 - 1, 0):min(split2 + 2, profile.size)])
    groove_depth = float(max(far_field - local1, far_field - local2, 0.0) * params.dx)
    angle1 = groove_angle_from_profile(profile, params.dx, split1)
    angle2 = groove_angle_from_profile(profile, params.dx, split2)
    groove_angle = float(np.nanmean([angle1, angle2]))
    metrics = compute_metrics_3d(run["phi"], run["nf"], params.dx, run["metrics"]["runtime_s"], interface_angle_deg=groove_angle)
    save_scalar_map(case_dir / "grain_boundary_groove_profile.png", profile[:, np.newaxis], "Interface z(x) profile")

    passed = groove_depth >= 0.15 * params.dx and np.isfinite(groove_angle)
    summary = f"GB groove depth={groove_depth:.2e} m, groove angle={groove_angle:.2f} deg."
    failure_reason = "" if passed else "Capillary relaxation did not produce a measurable groove at the trijunctions."
    extra = {"groove_depth": groove_depth, "angle_split1_deg": angle1, "angle_split2_deg": angle2}
    return make_case_result("grain_boundary_groove_trijunction", passed, summary, failure_reason, metrics, extra, case_dir)


def case_convergence_sweep(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    _ = overrides
    coarse = replace(base, nx=12, ny=4, nz=16, dx=1.0e-4, dt=5.0e-5, nsteps=6, delta_a=0.0, ksi=1.0)
    dt_half = replace(coarse, dt=2.5e-5, nsteps=12)
    delta_wide = replace(coarse, delta_factor=8.0)
    fine = replace(base, nx=24, ny=4, nz=32, dx=5.0e-5, dt=1.25e-5, nsteps=24, delta_a=0.0, ksi=1.0)

    def build_case_phi(params: SimulationParams) -> tuple[np.ndarray, np.ndarray]:
        derived = derive_constants(params)
        base_height = 0.35 * params.nz
        amplitude = 0.08 * params.nx
        phi = wavy_single_phi_3d(params.nx, params.ny, params.nz, params.dx, derived["delta"], base_height, amplitude, phase_shift=0.25)
        temp = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt)
        return phi, temp

    quats = quaternions_from_rotations([Rotation.identity()])
    run_coarse = run_3d_simulation("convergence_coarse", case_dir, coarse, *build_case_phi(coarse), quats, False, False)
    run_dt = run_3d_simulation("convergence_dt_half", case_dir, dt_half, *build_case_phi(dt_half), quats, False, False)
    run_delta = run_3d_simulation("convergence_delta_wide", case_dir, delta_wide, *build_case_phi(delta_wide), quats, False, False)
    run_fine = run_3d_simulation("convergence_fine", case_dir, fine, *build_case_phi(fine), quats, False, False)

    y_c = coarse.ny // 2
    y_f = fine.ny // 2
    coarse_slice = run_coarse["phi"][1, :, y_c, :]
    dt_slice = run_dt["phi"][1, :, y_c, :]
    delta_slice = run_delta["phi"][1, :, y_c, :]
    fine_slice = run_fine["phi"][1, ::2, y_f, ::2]

    grid_error = float(np.sqrt(np.mean((coarse_slice - fine_slice) ** 2)))
    dt_error = float(np.sqrt(np.mean((dt_slice - fine_slice) ** 2)))
    delta_sensitivity = float(np.sqrt(np.mean((delta_slice - coarse_slice) ** 2)))
    grid_interface_diff = abs(run_coarse["metrics"]["interface_position_mean"] - run_fine["metrics"]["interface_position_mean"])
    dt_interface_diff = abs(run_coarse["metrics"]["interface_position_mean"] - run_dt["metrics"]["interface_position_mean"])
    delta_interface_diff = abs(run_coarse["metrics"]["interface_position_mean"] - run_delta["metrics"]["interface_position_mean"])
    save_scalar_map(case_dir / "convergence_coarse_minus_fine.png", coarse_slice - fine_slice, "Coarse - fine/2", cmap="coolwarm")
    save_scalar_map(case_dir / "convergence_dt_half_minus_fine.png", dt_slice - fine_slice, "dt/2 - fine/2", cmap="coolwarm")

    metrics = run_coarse["metrics"]
    passed = (
        grid_interface_diff <= 2.5 * coarse.dx and
        dt_interface_diff <= 1.5 * coarse.dx and
        delta_interface_diff <= 1.5 * coarse.dx
    )
    summary = (
        f"Convergence sanity check: grid_error={grid_error:.2e}, dt_error={dt_error:.2e}, "
        f"delta_sensitivity={delta_sensitivity:.2e}, interface_diffs="
        f"({grid_interface_diff:.2e}, {dt_interface_diff:.2e}, {delta_interface_diff:.2e})."
    )
    failure_reason = "" if passed else "Grid, dt, or interface-width sensitivity exceeded the small-run sanity thresholds."
    extra = {
        "grid_error": grid_error,
        "dt_error": dt_error,
        "delta_sensitivity": delta_sensitivity,
        "grid_interface_diff": grid_interface_diff,
        "dt_interface_diff": dt_interface_diff,
        "delta_interface_diff": delta_interface_diff,
        "fine_runtime_s": run_fine["metrics"]["runtime_s"],
    }
    return make_case_result("convergence_grid_dt_delta", passed, summary, failure_reason, metrics, extra, case_dir)


def main_directional_low_params(base: SimulationParams) -> SimulationParams:
    params = build_directional_scan_case(base, delta_a=0.5, ksi=0.05, nsteps=8)
    return replace(params, nx=16, ny=6, nz=24)


def main_directional_high_params(base: SimulationParams) -> SimulationParams:
    params = build_directional_scan_case(base, delta_a=0.5, ksi=0.05, nsteps=8)
    return replace(params, nx=16, ny=6, nz=24)


def main_bicrystal_switch_params(base: SimulationParams) -> SimulationParams:
    return replace(
        base,
        nx=18,
        ny=6,
        nz=28,
        nsteps=8,
        delta_a=0.5,
        ksi=0.05,
        p_round=0.02,
        G=0.0,
        V_pulling=0.0,
    )


def main_bicrystal_energy_params(base: SimulationParams) -> SimulationParams:
    return replace(
        base,
        nx=18,
        ny=6,
        nz=28,
        nsteps=14,
        delta_a=0.2,
        ksi=0.01,
        p_round=0.02,
        G=0.0,
        V_pulling=0.0,
    )


def main_bicrystal_kinetic_params(base: SimulationParams) -> SimulationParams:
    return replace(
        base,
        nx=18,
        ny=6,
        nz=28,
        nsteps=10,
        delta_a=0.35,
        ksi=0.01,
        p_round=0.02,
        G=0.0,
        V_pulling=0.0,
    )


def main_multigrain_low_params(base: SimulationParams) -> SimulationParams:
    return replace(
        base,
        nx=24,
        ny=10,
        nz=30,
        nsteps=22,
        delta_a=0.2,
        ksi=0.01,
        p_round=0.02,
        G=0.0,
        V_pulling=0.0,
    )


def main_multigrain_high_params(base: SimulationParams) -> SimulationParams:
    return replace(
        base,
        nx=24,
        ny=10,
        nz=28,
        nsteps=10,
        delta_a=0.35,
        ksi=0.01,
        p_round=0.02,
        G=0.0,
        V_pulling=0.0,
    )


def main_directional_schedule() -> dict[str, float]:
    return {"low": 0.25, "high": 3.0}


def main_bicrystal_pair() -> tuple[str, str]:
    return "aligned_111_to_z", "identity"


def groove_observer(split_index: int) -> Callable[[np.ndarray, int], dict[str, Any]]:
    def _observe(phi: np.ndarray, step: int) -> dict[str, Any]:
        groove = robust_groove_metrics(phi, root_x_guess=float(split_index))
        return {
            "step": int(step),
            "dihedral_angle_like_deg": groove["dihedral_angle_like_deg"],
            "trijunction_x_cells": groove["trijunction_x_cells"],
            "trijunction_z_cells": groove["trijunction_z_cells"],
        }

    return _observe


def run_multigrain_family_competition(
    label: str,
    case_dir: Path,
    params: SimulationParams,
    family_a_orientation: str,
    family_b_orientation: str,
    undercooling: float,
    ngrains: int,
    waviness: float,
    enable_anisotropy: bool,
    enable_torque: bool,
) -> dict[str, Any]:
    derived = derive_constants(params)
    phi0 = alternating_multigrain_phi_3d(
        params.nx,
        params.ny,
        params.nz,
        params.dx,
        derived["delta"],
        7.0,
        ngrains=ngrains,
        waviness=waviness,
    )
    family_a = list(range(1, ngrains + 1, 2))
    family_b = list(range(2, ngrains + 1, 2))
    quats = np.zeros((ngrains + 1, 4), dtype=np.float64)
    quats[0] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rot_a = orientation_library()[family_a_orientation]
    rot_b = orientation_library()[family_b_orientation]
    for gid in family_a:
        quats[gid] = rot_a.as_quat()
    for gid in family_b:
        quats[gid] = rot_b.as_quat()

    run = run_3d_simulation(
        label,
        case_dir,
        params,
        phi0,
        constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt - undercooling),
        quats,
        enable_anisotropy,
        enable_torque,
    )
    metrics = finalize_metrics(dict(run["metrics"]), phi0, run["phi"], params.dx, params.dt * params.nsteps)
    group = bicrystal_group_metrics(run["phi"], family_a, family_b, params.dx)
    metrics["winner_loser_volume_ratio"] = group["winner_loser_volume_ratio"]
    return {
        "phi0": phi0,
        "run": run,
        "metrics": metrics,
        "group": group,
        "winner_orientation": family_a_orientation if group["winner_family"] == "A" else family_b_orientation,
        "loser_orientation": family_b_orientation if group["winner_family"] == "A" else family_a_orientation,
        "ngrains": ngrains,
    }


def run_bicrystal_competition(
    label: str,
    case_dir: Path,
    params: SimulationParams,
    left_orientation: str,
    right_orientation: str,
    undercooling: float,
    enable_anisotropy: bool,
    enable_torque: bool,
) -> dict[str, Any]:
    derived = derive_constants(params)
    split_index = params.nx // 2
    phi0 = bicrystal_phi_3d(
        params.nx,
        params.ny,
        params.nz,
        params.dx,
        derived["delta"],
        seed_height=7.0,
        split_index=split_index,
        waviness=0.8,
    )
    temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt - undercooling)
    quats = quaternions_from_rotations([orientation_library()[left_orientation], orientation_library()[right_orientation]])
    root_initial = grain_boundary_root_and_trijunction(phi0, 1, 2, x_guess=split_index)
    run = run_3d_simulation(
        label,
        case_dir,
        params,
        phi0,
        temp0,
        quats,
        enable_anisotropy,
        enable_torque,
        sample_every=max(2, params.nsteps // 4),
        observer=groove_observer(split_index),
    )
    groove = robust_groove_metrics(run["phi"], root_x_guess=float(split_index))
    angle_history = [sample.get("dihedral_angle_like_deg", float("nan")) for sample in run.get("samples", [])]
    metrics = finalize_metrics(
        dict(run["metrics"]),
        phi0,
        run["phi"],
        params.dx,
        params.dt * params.nsteps,
        root_initial_cells=root_initial["root_x_cells"],
        trijunction_final=grain_boundary_root_and_trijunction(run["phi"], 1, 2, x_guess=split_index),
        groove_metrics=groove,
        angle_history_deg=angle_history,
    )
    family = bicrystal_group_metrics(run["phi"], [1], [2], params.dx)
    metrics["winner_loser_volume_ratio"] = family["winner_loser_volume_ratio"]
    return {
        "phi0": phi0,
        "run": run,
        "metrics": metrics,
        "winner_gid": 1 if family["winner_family"] == "A" else 2,
        "winner_orientation": left_orientation if family["winner_family"] == "A" else right_orientation,
        "loser_orientation": right_orientation if family["winner_family"] == "A" else left_orientation,
        "family_metrics": family,
        "root_initial_cells": root_initial["root_x_cells"],
        "split_index": split_index,
    }


def case_single_grain_preferred_growth_benchmark(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = main_directional_high_params(base)
    enable_anisotropy, enable_torque = resolve_toggles(True, False, overrides)
    scan = run_directional_scan(
        case_dir,
        params,
        {"high": main_directional_schedule()["high"]},
        enable_anisotropy,
        enable_torque,
        include_isotropic_baseline=True,
        artifact_prefix="benchmark",
    )
    rows = scan["rows"]
    on_stats = directional_stats(rows, "high")
    off_stats = directional_stats(rows, "high_iso")
    aligned = directional_row_by_orientation(rows, "high", "aligned_111_to_z")
    spread_on = on_stats["spread"]
    spread_off = off_stats["spread"]
    aligned_penalty = float(on_stats["best"]["front_velocity"] - aligned["front_velocity"])
    passed = spread_on >= 5.0 * max(spread_off, 1.0e-12) and aligned_penalty >= 5.0e-2
    level = classify_signal_level(spread_on, 3.0e-2, 8.0e-2, passed)
    metrics = blank_metrics()
    metrics["interface_position_mean"] = float(on_stats["best"]["interface_mean"])
    metrics["interface_position_max"] = float(on_stats["best"]["interface_max"])
    metrics["solid_volume_growth"] = float(on_stats["best"]["solid_volume_growth"])
    metrics["front_velocity"] = float(on_stats["best"]["front_velocity"])
    summary = (
        f"Single-grain preferred growth spread_on={spread_on:.2e} m/s vs isotropic spread_off={spread_off:.2e} m/s; "
        f"best={on_stats['best']['orientation']}, worst={on_stats['worst']['orientation']}, "
        f"aligned penalty={aligned_penalty:.2e} m/s."
    )
    extra = {
        "spread_on": spread_on,
        "spread_off": spread_off,
        "best_orientation": on_stats["best"]["orientation"],
        "worst_orientation": on_stats["worst"]["orientation"],
        "best_front_velocity": on_stats["best"]["front_velocity"],
        "worst_front_velocity": on_stats["worst"]["front_velocity"],
        "aligned_front_velocity": aligned["front_velocity"],
        "aligned_penalty": aligned_penalty,
    }
    return make_case_result(
        "single_grain_preferred_growth_benchmark",
        passed,
        summary,
        "" if passed else "Preferred growth did not separate enough from the isotropic baseline.",
        metrics,
        extra,
        case_dir,
        category="main",
        level=level,
    )


def case_directional_preference_map(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    enable_anisotropy, enable_torque = resolve_toggles(True, False, overrides)
    schedule = main_directional_schedule()
    scan_low = run_directional_scan(
        case_dir,
        main_directional_low_params(base),
        {"low": schedule["low"]},
        enable_anisotropy,
        enable_torque,
        include_isotropic_baseline=True,
        artifact_prefix="low",
    )
    scan_high = run_directional_scan(
        case_dir,
        main_directional_high_params(base),
        {"high": schedule["high"]},
        enable_anisotropy,
        enable_torque,
        include_isotropic_baseline=True,
        artifact_prefix="high",
    )
    low_rows = scan_low["rows"]
    high_rows = scan_high["rows"]
    low_stats = directional_stats(low_rows, "low")
    high_stats = directional_stats(high_rows, "high")
    low_iso_stats = directional_stats(low_rows, "low_iso")
    high_iso_stats = directional_stats(high_rows, "high_iso")
    low_aligned = directional_row_by_orientation(low_rows, "low", "aligned_111_to_z")
    high_aligned = directional_row_by_orientation(high_rows, "high", "aligned_111_to_z")
    aligned_penalty_low = float(low_stats["best"]["front_velocity"] - low_aligned["front_velocity"])
    aligned_penalty_high = float(high_stats["best"]["front_velocity"] - high_aligned["front_velocity"])
    passed = (
        low_stats["spread"] >= 1.5e-2 and
        high_stats["spread"] >= 5.0e-2 and
        high_stats["spread"] > low_stats["spread"] and
        low_iso_stats["spread"] <= 1.0e-10 and
        high_iso_stats["spread"] <= 1.0e-10 and
        aligned_penalty_high > aligned_penalty_low + 2.0e-2
    )
    level = classify_signal_level(high_stats["spread"], 4.0e-2, 8.0e-2, passed)
    metrics = blank_metrics()
    metrics["interface_position_mean"] = float(high_stats["best"]["interface_mean"])
    metrics["interface_position_max"] = float(high_stats["best"]["interface_max"])
    metrics["solid_volume_growth"] = float(high_stats["best"]["solid_volume_growth"])
    metrics["front_velocity"] = float(high_stats["best"]["front_velocity"])
    summary = (
        f"Directional map spreads: low={low_stats['spread']:.2e}, high={high_stats['spread']:.2e}; "
        f"aligned penalty grew {aligned_penalty_low:.2e} -> {aligned_penalty_high:.2e} m/s."
    )
    extra = {
        "low": {
            "best_orientation": low_stats["best"]["orientation"],
            "worst_orientation": low_stats["worst"]["orientation"],
            "spread": low_stats["spread"],
            "iso_spread": low_iso_stats["spread"],
            "aligned_penalty": aligned_penalty_low,
        },
        "high": {
            "best_orientation": high_stats["best"]["orientation"],
            "worst_orientation": high_stats["worst"]["orientation"],
            "spread": high_stats["spread"],
            "iso_spread": high_iso_stats["spread"],
            "aligned_penalty": aligned_penalty_high,
        },
    }
    return make_case_result(
        "directional_preference_map",
        passed,
        summary,
        "" if passed else "Directional preference map did not show a clear anisotropic spread.",
        metrics,
        extra,
        case_dir,
        category="main",
        level=level,
    )


def case_anisotropy_threshold_test(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    enable_anisotropy, _ = resolve_toggles(True, False, overrides)
    delta_values = [0.0, 0.1, 0.2, 0.35, 0.5]
    rows: list[dict[str, Any]] = []
    for delta_a in delta_values:
        params = replace(build_directional_scan_case(base, delta_a=delta_a, ksi=0.05, nsteps=8), nx=16, ny=6, nz=24)
        phi0 = single_grain_benchmark_phi(params)
        temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt - 3.0)
        q1 = quaternions_from_rotations([Rotation.identity()])
        q2 = quaternions_from_rotations([rotation_align_111_to_z()])
        run_a = run_3d_simulation(f"threshold_identity_{delta_a:.2f}", case_dir, params, phi0, temp0, q1, enable_anisotropy and delta_a > 0.0, False)
        run_b = run_3d_simulation(f"threshold_aligned_{delta_a:.2f}", case_dir, params, phi0, temp0, q2, enable_anisotropy and delta_a > 0.0, False)
        metrics_a = finalize_metrics(dict(run_a["metrics"]), phi0, run_a["phi"], params.dx, params.dt * params.nsteps)
        metrics_b = finalize_metrics(dict(run_b["metrics"]), phi0, run_b["phi"], params.dx, params.dt * params.nsteps)
        diff = abs(metrics_a["front_velocity"] - metrics_b["front_velocity"])
        rows.append({"delta_a": delta_a, "front_velocity_diff": diff})

    with (case_dir / "anisotropy_threshold.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["delta_a"] for row in rows], [row["front_velocity_diff"] for row in rows], marker="o")
    ax.set_xlabel("delta_a")
    ax.set_ylabel("|delta front velocity| [m/s]")
    ax.set_title("Anisotropy threshold response")
    plt.tight_layout()
    plt.savefig(case_dir / "anisotropy_threshold.png", dpi=150)
    plt.close(fig)

    monotonic = all(rows[i + 1]["front_velocity_diff"] >= rows[i]["front_velocity_diff"] - 2.0e-2 for i in range(len(rows) - 1))
    passed = rows[0]["front_velocity_diff"] <= 5.0e-4 and rows[-1]["front_velocity_diff"] >= 7.0e-2 and monotonic
    level = classify_signal_level(rows[-1]["front_velocity_diff"], 7.0e-2, 9.0e-2, passed)
    metrics = blank_metrics()
    metrics["front_velocity"] = float(rows[-1]["front_velocity_diff"])
    summary = (
        f"Directional contrast grew from {rows[0]['front_velocity_diff']:.2e} to "
        f"{rows[-1]['front_velocity_diff']:.2e} as delta_a increased."
    )
    extra = {"rows": rows, "monotonic_non_decreasing": monotonic}
    return make_case_result(
        "anisotropy_threshold_test",
        passed,
        summary,
        "" if passed else "Directional contrast did not grow cleanly with anisotropy strength.",
        metrics,
        extra,
        case_dir,
        category="main",
        level=level,
    )


def case_bicrystal_competition_low_high_driving(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = main_bicrystal_switch_params(base)
    schedule = main_directional_schedule()
    enable_anisotropy, enable_torque = resolve_toggles(True, False, overrides)
    low_orientation, high_orientation = main_bicrystal_pair()
    low_run = run_bicrystal_competition(
        "bicrystal_low",
        case_dir,
        params,
        low_orientation,
        high_orientation,
        schedule["low"],
        enable_anisotropy,
        enable_torque,
    )
    high_run = run_bicrystal_competition(
        "bicrystal_high",
        case_dir,
        params,
        low_orientation,
        high_orientation,
        schedule["high"],
        enable_anisotropy,
        enable_torque,
    )
    winner_switched = (
        low_run["winner_orientation"] == low_orientation and
        high_run["winner_orientation"] == high_orientation
    )
    signal = min(low_run["metrics"]["winner_loser_volume_ratio"], high_run["metrics"]["winner_loser_volume_ratio"])
    passed = winner_switched and signal >= 1.01
    level = classify_signal_level(signal, 1.01, 1.015, passed)
    metrics = dict(high_run["metrics"])
    summary = (
        f"Bicrystal winner switched {low_run['winner_orientation']} -> {high_run['winner_orientation']} "
        f"between low/high driving."
    )
    extra = {
        "pair": {"low_orientation": low_orientation, "high_orientation": high_orientation},
        "low_winner": low_run["winner_orientation"],
        "high_winner": high_run["winner_orientation"],
        "low_ratio": low_run["metrics"]["winner_loser_volume_ratio"],
        "high_ratio": high_run["metrics"]["winner_loser_volume_ratio"],
        "low_root_shift": low_run["metrics"]["grain_boundary_root_shift"],
        "high_root_shift": high_run["metrics"]["grain_boundary_root_shift"],
    }
    return make_case_result(
        "bicrystal_competition_low_high_driving",
        passed,
        summary,
        "" if passed else "Low/high driving did not produce the expected bicrystal winner switch.",
        metrics,
        extra,
        case_dir,
        category="main",
        level=level,
    )


def case_bicrystal_interfacial_energy_dominated_regime(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = main_bicrystal_energy_params(base)
    enable_anisotropy, enable_torque = resolve_toggles(True, False, overrides)
    aligned_orientation, identity_orientation = main_bicrystal_pair()
    low_undercooling = 0.01
    low_on = run_bicrystal_competition(
        "bicrystal_low_on",
        case_dir,
        params,
        aligned_orientation,
        identity_orientation,
        low_undercooling,
        enable_anisotropy,
        enable_torque,
    )
    low_off = run_bicrystal_competition(
        "bicrystal_low_off",
        case_dir,
        replace(params, delta_a=0.0, ksi=1.0),
        aligned_orientation,
        identity_orientation,
        low_undercooling,
        False,
        False,
    )
    ratio_gain = low_on["metrics"]["winner_loser_volume_ratio"] - low_off["metrics"]["winner_loser_volume_ratio"]
    velocity_gain = low_on["metrics"]["front_velocity"] - low_off["metrics"]["front_velocity"]
    root_shift_gain = abs(low_on["metrics"]["grain_boundary_root_shift"]) - abs(low_off["metrics"]["grain_boundary_root_shift"])
    passed = (
        low_on["winner_orientation"] == aligned_orientation and
        velocity_gain > 5.0e-3 and
        ratio_gain > 3.0e-3 and
        root_shift_gain > 5.0e-7
    )
    level = classify_signal_level(velocity_gain, 5.0e-3, 1.2e-2, passed)
    metrics = dict(low_on["metrics"])
    summary = (
        f"Low-driving bicrystal favored {low_on['winner_orientation']} with front-velocity gain {velocity_gain:.2e} m/s "
        f"and ratio gain {ratio_gain:.2e} over anisotropy-off."
    )
    extra = {
        "expected_low_winner": aligned_orientation,
        "winner_on": low_on["winner_orientation"],
        "winner_off": low_off["winner_orientation"],
        "ratio_on": low_on["metrics"]["winner_loser_volume_ratio"],
        "ratio_off": low_off["metrics"]["winner_loser_volume_ratio"],
        "front_velocity_on": low_on["metrics"]["front_velocity"],
        "front_velocity_off": low_off["metrics"]["front_velocity"],
        "front_velocity_gain": velocity_gain,
        "root_shift_on": low_on["metrics"]["grain_boundary_root_shift"],
        "root_shift_off": low_off["metrics"]["grain_boundary_root_shift"],
        "root_shift_gain": root_shift_gain,
    }
    return make_case_result(
        "bicrystal_interfacial_energy_dominated_regime_test",
        passed,
        summary,
        "" if passed else "Low-driving bicrystal did not show a clean aligned-facet preference over the anisotropy-off baseline.",
        metrics,
        extra,
        case_dir,
        category="main",
        level=level,
    )


def case_bicrystal_kinetic_dominated_regime(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    params = main_bicrystal_kinetic_params(base)
    enable_anisotropy, enable_torque = resolve_toggles(True, False, overrides)
    aligned_orientation, identity_orientation = main_bicrystal_pair()
    high_undercooling = 6.0
    high_on = run_bicrystal_competition(
        "bicrystal_high_on",
        case_dir,
        params,
        aligned_orientation,
        identity_orientation,
        high_undercooling,
        enable_anisotropy,
        enable_torque,
    )
    high_off = run_bicrystal_competition(
        "bicrystal_high_off",
        case_dir,
        replace(params, delta_a=0.0, ksi=1.0),
        aligned_orientation,
        identity_orientation,
        high_undercooling,
        False,
        False,
    )
    ratio_gain = high_on["metrics"]["winner_loser_volume_ratio"] - high_off["metrics"]["winner_loser_volume_ratio"]
    root_switch = high_on["metrics"]["grain_boundary_root_shift"] * high_off["metrics"]["grain_boundary_root_shift"] < 0.0
    passed = (
        high_on["winner_orientation"] == identity_orientation and
        high_off["winner_orientation"] == aligned_orientation and
        ratio_gain > 0.02 and
        root_switch
    )
    level = classify_signal_level(ratio_gain, 0.02, 0.05, passed)
    metrics = dict(high_on["metrics"])
    summary = (
        f"High-driving bicrystal favored {high_on['winner_orientation']} with ratio gain {ratio_gain:.2e} "
        f"over the anisotropy-off baseline."
    )
    extra = {
        "expected_high_winner": identity_orientation,
        "winner_on": high_on["winner_orientation"],
        "winner_off": high_off["winner_orientation"],
        "ratio_on": high_on["metrics"]["winner_loser_volume_ratio"],
        "ratio_off": high_off["metrics"]["winner_loser_volume_ratio"],
        "root_shift_on": high_on["metrics"]["grain_boundary_root_shift"],
        "root_shift_off": high_off["metrics"]["grain_boundary_root_shift"],
        "root_switch": root_switch,
    }
    return make_case_result(
        "bicrystal_kinetic_dominated_regime_test",
        passed,
        summary,
        "" if passed else "High-driving bicrystal did not show the expected kinetic-dominated winner.",
        metrics,
        extra,
        case_dir,
        category="main",
        level=level,
    )


def case_multigrain_competition_extension(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    enable_anisotropy, enable_torque = resolve_toggles(True, False, overrides)
    aligned_orientation, identity_orientation = main_bicrystal_pair()
    low_run = run_multigrain_family_competition(
        "multigrain_low",
        case_dir,
        main_multigrain_low_params(base),
        aligned_orientation,
        identity_orientation,
        undercooling=0.01,
        ngrains=6,
        waviness=1.5,
        enable_anisotropy=enable_anisotropy,
        enable_torque=enable_torque,
    )
    high_run = run_multigrain_family_competition(
        "multigrain_high",
        case_dir,
        main_multigrain_high_params(base),
        aligned_orientation,
        identity_orientation,
        undercooling=6.0,
        ngrains=6,
        waviness=1.0,
        enable_anisotropy=enable_anisotropy,
        enable_torque=enable_torque,
    )
    low_family_name = low_run["winner_orientation"]
    high_family_name = high_run["winner_orientation"]
    low_ratio = low_run["group"]["winner_loser_volume_ratio"]
    high_ratio = high_run["group"]["winner_loser_volume_ratio"]
    passed = (
        low_family_name == aligned_orientation and
        high_family_name == identity_orientation and
        high_ratio > 1.05 and
        (high_ratio - 1.0) > 10.0 * max(low_ratio - 1.0, 1.0e-12)
    )
    signal = min(low_ratio, high_ratio)
    level = classify_signal_level(signal, 1.005, 1.02, passed)
    metrics = dict(high_run["metrics"])
    summary = (
        f"Multigrain extension switched dominant family {low_family_name} -> {high_family_name}; "
        f"winner ratios were {low_ratio:.4f} and {high_ratio:.4f}."
    )
    extra = {
        "low_family_name": low_family_name,
        "high_family_name": high_family_name,
        "low_group": low_run["group"],
        "high_group": high_run["group"],
        "low_surviving_grains": low_run["metrics"]["surviving_grains"],
        "high_surviving_grains": high_run["metrics"]["surviving_grains"],
    }
    return make_case_result(
        "multigrain_competition_extension",
        passed,
        summary,
        "" if passed else "Multigrain extension did not preserve the bicrystal family switch trend strongly enough.",
        metrics,
        extra,
        case_dir,
        category="main",
        level=level,
    )


def case_groove_depth_only(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    _ = overrides
    params = replace(base, nx=20, ny=6, nz=24, nsteps=20, delta_a=0.0, ksi=1.0, G=0.0, V_pulling=0.0)
    derived = derive_constants(params)
    split_index = params.nx // 2
    phi0 = bicrystal_phi_3d(params.nx, params.ny, params.nz, params.dx, derived["delta"], 7.0, split_index, waviness=0.6)
    temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt)
    quats = quaternions_from_rotations([Rotation.identity(), Rotation.identity()])
    root_initial = grain_boundary_root_and_trijunction(phi0, 1, 2, x_guess=split_index)
    run = run_3d_simulation("groove_depth_only", case_dir, params, phi0, temp0, quats, False, False)
    groove = robust_groove_metrics(run["phi"], root_x_guess=float(split_index))
    metrics = finalize_metrics(
        dict(run["metrics"]),
        phi0,
        run["phi"],
        params.dx,
        params.dt * params.nsteps,
        root_initial_cells=root_initial["root_x_cells"],
        groove_metrics=groove,
    )
    groove_depth = groove["groove_depth_cells"] * params.dx
    passed = np.isfinite(groove_depth) and groove_depth >= 0.10 * params.dx
    level = classify_signal_level(groove_depth / params.dx if np.isfinite(groove_depth) else float("nan"), 0.10, 0.25, passed)
    summary = f"Groove depth benchmark gave depth={groove_depth:.2e} m."
    extra = {"groove_depth": groove_depth, "groove_metrics": groove}
    return make_case_result(
        "groove_depth_only",
        passed,
        summary,
        "" if passed else "Groove depth remained too small for a reliable capillarity benchmark.",
        metrics,
        extra,
        case_dir,
        category="main",
        level=level,
    )


def case_robust_groove_angle_estimation(case_dir: Path, base: SimulationParams, overrides: ToggleOverrides) -> CaseOutcome:
    _ = overrides
    params = replace(base, nx=20, ny=6, nz=24, nsteps=20, delta_a=0.0, ksi=1.0, G=0.0, V_pulling=0.0)
    derived = derive_constants(params)
    split_index = params.nx // 2
    phi0 = bicrystal_phi_3d(params.nx, params.ny, params.nz, params.dx, derived["delta"], 7.0, split_index, waviness=0.6)
    temp0 = constant_temp_field_3d(params.nx, params.ny, params.nz, params.T_melt)
    quats = quaternions_from_rotations([Rotation.identity(), Rotation.identity()])
    root_initial = grain_boundary_root_and_trijunction(phi0, 1, 2, x_guess=split_index)
    run = run_3d_simulation(
        "robust_groove_angle",
        case_dir,
        params,
        phi0,
        temp0,
        quats,
        False,
        False,
        sample_every=2,
        observer=groove_observer(split_index),
    )
    groove = robust_groove_metrics(run["phi"], root_x_guess=float(split_index))
    angle_history = [sample.get("dihedral_angle_like_deg", float("nan")) for sample in run.get("samples", [])]
    metrics = finalize_metrics(
        dict(run["metrics"]),
        phi0,
        run["phi"],
        params.dx,
        params.dt * params.nsteps,
        root_initial_cells=root_initial["root_x_cells"],
        groove_metrics=groove,
        angle_history_deg=angle_history,
    )
    fit_rmse = max(groove["left_fit_rmse"], groove["right_fit_rmse"])
    passed = np.isfinite(metrics["interface_angle_deg"]) and fit_rmse <= 0.6
    level = classify_signal_level(metrics["interface_angle_deg"], 2.0, 5.0, passed)
    summary = (
        f"Robust groove fit produced dihedral-like angle={metrics['interface_angle_deg']:.2f} deg "
        f"with fit_rmse={fit_rmse:.2e} cells."
    )
    extra = {"groove_metrics": groove, "angle_history_deg": angle_history, "fit_rmse": fit_rmse}
    return make_case_result(
        "robust_groove_angle_estimation",
        passed,
        summary,
        "" if passed else "Robust groove-angle fit remained numerically unstable.",
        metrics,
        extra,
        case_dir,
        category="main",
        level=level,
    )


CASE_BUILDERS: dict[str, Callable[[Path, SimulationParams, ToggleOverrides], CaseOutcome]] = {
    "basic_constraints": case_basic_constraints,
    "kmax_overflow_detection": case_kmax_overflow_detection,
    "static_flat_interface": case_static_flat_interface,
    "two_d_limit_consistency": case_two_d_limit_consistency,
    "boundary_conditions": case_boundary_conditions,
    "isotropic_orientation_independence": case_isotropic_orientation_independence,
    "anisotropic_preferred_growth": case_anisotropic_preferred_growth,
    "torque_term_contribution": case_torque_term_contribution,
    "grain_competition": case_grain_competition,
    "grain_boundary_groove_trijunction": case_grain_boundary_groove,
    "convergence_grid_dt_delta": case_convergence_sweep,
    "single_grain_preferred_growth_benchmark": case_single_grain_preferred_growth_benchmark,
    "directional_preference_map": case_directional_preference_map,
    "anisotropy_threshold_test": case_anisotropy_threshold_test,
    "bicrystal_competition_low_high_driving": case_bicrystal_competition_low_high_driving,
    "bicrystal_interfacial_energy_dominated_regime_test": case_bicrystal_interfacial_energy_dominated_regime,
    "bicrystal_kinetic_dominated_regime_test": case_bicrystal_kinetic_dominated_regime,
    "multigrain_competition_extension": case_multigrain_competition_extension,
    "groove_depth_only": case_groove_depth_only,
    "robust_groove_angle_estimation": case_robust_groove_angle_estimation,
}

MAIN_CASE_NAMES = {
    "single_grain_preferred_growth_benchmark",
    "directional_preference_map",
    "anisotropy_threshold_test",
    "bicrystal_competition_low_high_driving",
    "bicrystal_interfacial_energy_dominated_regime_test",
    "bicrystal_kinetic_dominated_regime_test",
    "multigrain_competition_extension",
    "groove_depth_only",
    "robust_groove_angle_estimation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3D multiphase-field verification cases.")
    parser.add_argument("--out-dir", default="tests/output/latest", help="Directory for case artifacts.")
    parser.add_argument("--report-path", default="verification_report.md", help="Markdown report path.")
    parser.add_argument("--csv-path", default=None, help="CSV output path. Default: <out-dir>/verification_results.csv")
    parser.add_argument("--cases", nargs="*", default=None, help="Subset of case names to run.")
    parser.add_argument("--list-cases", action="store_true", help="List case names and exit.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failing case.")
    parser.add_argument("--anisotropy", choices=["default", "on", "off"], default="default", help="Override anisotropy toggle.")
    parser.add_argument("--torque", choices=["default", "on", "off"], default="default", help="Override torque toggle.")
    return parser.parse_args()


def parse_toggle_overrides(args: argparse.Namespace) -> ToggleOverrides:
    def parse_flag(value: str) -> int | None:
        if value == "default":
            return None
        if value == "on":
            return 1
        return 0

    return ToggleOverrides(anisotropy=parse_flag(args.anisotropy), torque=parse_flag(args.torque))


def execute_case(
    name: str,
    builder: Callable[[Path, SimulationParams, ToggleOverrides], CaseOutcome],
    out_root: Path,
    base: SimulationParams,
    overrides: ToggleOverrides,
) -> CaseOutcome:
    case_dir = out_root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        return builder(case_dir, base, overrides)
    except Exception as exc:  # noqa: BLE001
        failure_reason = f"{type(exc).__name__}: {exc}"
        trace = traceback.format_exc(limit=4)
        write_json(case_dir / "case_error.json", {"error": failure_reason, "traceback": trace})
        return make_case_result(
            name=name,
            passed=False,
            summary="Case crashed before completing.",
            failure_reason=failure_reason,
            metrics=blank_metrics(),
            extra={"traceback": trace},
            case_dir=case_dir,
            category="main" if name in MAIN_CASE_NAMES else "quick",
            level="L0",
        )


def flatten_outcome(outcome: CaseOutcome) -> dict[str, Any]:
    metrics = dict(outcome.metrics)
    return {
        "name": outcome.name,
        "category": outcome.category,
        "passed": int(outcome.passed),
        "level": outcome.level,
        "summary": outcome.summary,
        "failure_reason": outcome.failure_reason,
        "case_dir": outcome.case_dir,
        "phi_sum_error_max": metrics.get("phi_sum_error_max"),
        "phi_min": metrics.get("phi_min"),
        "phi_max": metrics.get("phi_max"),
        "nf_max": metrics.get("nf_max"),
        "nan_count": metrics.get("nan_count"),
        "solid_fraction": metrics.get("solid_fraction"),
        "solid_volume": metrics.get("solid_volume"),
        "solid_volume_growth": metrics.get("solid_volume_growth"),
        "grain_volumes": json.dumps(to_serializable(metrics.get("grain_volumes", [])), ensure_ascii=False),
        "surviving_grains": metrics.get("surviving_grains"),
        "interface_position_mean": metrics.get("interface_position_mean"),
        "interface_position_max": metrics.get("interface_position_max"),
        "interface_position_std": metrics.get("interface_position_std"),
        "interface_roughness": metrics.get("interface_roughness"),
        "winner_loser_volume_ratio": metrics.get("winner_loser_volume_ratio"),
        "grain_boundary_root_shift": metrics.get("grain_boundary_root_shift"),
        "trijunction_x": metrics.get("trijunction_x"),
        "trijunction_z": metrics.get("trijunction_z"),
        "dihedral_angle_like_metric": metrics.get("dihedral_angle_like_metric"),
        "interface_angle_deg": metrics.get("interface_angle_deg"),
        "angle_deviation_from_final_state": metrics.get("angle_deviation_from_final_state"),
        "front_velocity": metrics.get("front_velocity"),
        "runtime_s": metrics.get("runtime_s"),
        "extra": json.dumps(to_serializable(outcome.extra), ensure_ascii=False),
    }


def write_csv(path: Path, outcomes: list[CaseOutcome]) -> None:
    rows = [flatten_outcome(outcome) for outcome in outcomes]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "-"
    return f"{value:.3e}"


def write_report(
    report_path: Path,
    csv_path: Path,
    out_root: Path,
    base: SimulationParams,
    notes: list[str],
    outcomes: list[CaseOutcome],
) -> None:
    passed = sum(outcome.passed for outcome in outcomes)
    total = len(outcomes)
    main_cases = [outcome for outcome in outcomes if outcome.category == "main"]
    quick_cases = [outcome for outcome in outcomes if outcome.category != "main"]
    lines: list[str] = []
    lines.append("# 3D Multiphase-Field Verification Report")
    lines.append("")
    lines.append(f"- Summary: {passed}/{total} cases passed")
    lines.append(f"- Main physics: {sum(outcome.passed for outcome in main_cases)}/{len(main_cases)} passed")
    lines.append(f"- Quick sanity: {sum(outcome.passed for outcome in quick_cases)}/{len(quick_cases)} passed")
    lines.append(f"- Artifact root: `{out_root}`")
    lines.append(f"- CSV: `{csv_path}`")
    lines.append(f"- Compile-time KMAX: `{base.KMAX}`")
    lines.append(f"- Threads per block: `{base.threads_per_block}`")
    lines.append("")
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    for section_name, section_outcomes in (("Main Physics", main_cases), ("Quick Sanity", quick_cases)):
        if not section_outcomes:
            continue
        lines.append(f"## {section_name}")
        lines.append("")
        lines.append("| Case | Status | Level | Interface mean | Interface max | Volume growth | Winner/Loser | Root shift | Front velocity |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for outcome in section_outcomes:
            metrics = outcome.metrics
            lines.append(
                "| "
                f"[{outcome.name}]({Path(outcome.case_dir).as_posix()}) | "
                f"{'PASS' if outcome.passed else 'FAIL'} | "
                f"{outcome.level} | "
                f"{format_float(metrics.get('interface_position_mean'))} | "
                f"{format_float(metrics.get('interface_position_max'))} | "
                f"{format_float(metrics.get('solid_volume_growth'))} | "
                f"{format_float(metrics.get('winner_loser_volume_ratio'))} | "
                f"{format_float(metrics.get('grain_boundary_root_shift'))} | "
                f"{format_float(metrics.get('front_velocity'))} |"
            )
        lines.append("")
    lines.append("## Case Notes")
    lines.append("")
    for outcome in main_cases + quick_cases:
        lines.append(f"### {outcome.name}")
        lines.append("")
        lines.append(f"- Category: {outcome.category}")
        lines.append(f"- Status: {'PASS' if outcome.passed else 'FAIL'}")
        lines.append(f"- Physical level: {outcome.level}")
        lines.append(f"- Summary: {outcome.summary}")
        if outcome.failure_reason:
            lines.append(f"- Failure reason: {outcome.failure_reason}")
        lines.append(f"- Artifacts: `{outcome.case_dir}`")
        lines.append(
            f"- Key metrics: phi_sum_error_max={format_float(outcome.metrics.get('phi_sum_error_max'))}, "
            f"solid_fraction={format_float(outcome.metrics.get('solid_fraction'))}, "
            f"surviving_grains={outcome.metrics.get('surviving_grains', '-')}, "
            f"interface_mean={format_float(outcome.metrics.get('interface_position_mean'))}, "
            f"interface_max={format_float(outcome.metrics.get('interface_position_max'))}, "
            f"winner_ratio={format_float(outcome.metrics.get('winner_loser_volume_ratio'))}, "
            f"root_shift={format_float(outcome.metrics.get('grain_boundary_root_shift'))}, "
            f"interface_angle={format_float(outcome.metrics.get('interface_angle_deg'))}, "
            f"front_velocity={format_float(outcome.metrics.get('front_velocity'))}"
        )
        if outcome.extra:
            lines.append(f"- Extra: `{json.dumps(to_serializable(outcome.extra), ensure_ascii=False)}`")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for name in CASE_BUILDERS:
            print(name)
        return 0

    if not cuda.is_available():
        raise RuntimeError("CUDA is required for verification runs.")

    base, notes = load_reference_params()
    overrides = parse_toggle_overrides(args)

    selected = list(CASE_BUILDERS.keys()) if not args.cases else args.cases
    unknown = [name for name in selected if name not in CASE_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path) if args.csv_path else out_root / "verification_results.csv"
    report_path = Path(args.report_path)

    write_json(
        out_root / "verification_context.json",
        {
            "base_params": asdict(base),
            "notes": notes,
            "selected_cases": selected,
            "toggle_overrides": asdict(overrides),
        },
    )

    outcomes: list[CaseOutcome] = []
    for name in selected:
        print(f"[verification] running {name}")
        outcome = execute_case(name, CASE_BUILDERS[name], out_root, base, overrides)
        outcomes.append(outcome)
        print(f"[verification] {name}: {'PASS' if outcome.passed else 'FAIL'} | {outcome.summary}")
        if args.stop_on_fail and not outcome.passed:
            break

    write_csv(csv_path, outcomes)
    write_report(report_path, csv_path, out_root, base, notes, outcomes)
    print(f"[verification] wrote CSV -> {csv_path}")
    print(f"[verification] wrote report -> {report_path}")
    return 0 if all(outcome.passed for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
