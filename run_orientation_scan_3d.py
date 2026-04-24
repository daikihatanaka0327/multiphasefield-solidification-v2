"""
run_orientation_scan_3d.py
==========================
3D single-crystal isothermal orientation scan.

Goal
----
Quantify orientation-dependent growth-rate differences without requiring
faceted shape reproduction. This script runs several single-grain
orientations, measures volume/equivalent radius over time, and aggregates
growth-rate statistics.

Outputs
-------
For each orientation case:
  - timeseries.csv
  - xy_XXXX.png, xz_XXXX.png, yz_XXXX.png
  - final_xy.png, final_xz.png, final_yz.png

For the full scan:
  - summary.csv
  - growth_rate_vs_orientation.png
  - equivalent_radius_vs_time.png
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from numba import cuda
from scipy.spatial.transform import Rotation

from src.gpu_kernels_3d import (
    KMAX as KERNEL_KMAX,
    kernel_update_nfmf_3d,
    kernel_update_phasefield_active_3d_switchable,
)
from src.orientation_utils import build_quaternion_from_config, compute_rotated_n111
from src.plot_utils import save_run_config
from src.seed_modes_3d import build_interaction_matrices, init_singlemode_sphere_3d


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D orientation scan for single-crystal isothermal growth.")
    parser.add_argument("--config", default="config_3d.yaml", help="Path to config YAML.")
    parser.add_argument("--out-dir", default=None, help="Override output directory.")
    parser.add_argument("--nx", type=int, default=None, help="Override grid nx.")
    parser.add_argument("--ny", type=int, default=None, help="Override grid ny.")
    parser.add_argument("--nz", type=int, default=None, help="Override grid nz.")
    parser.add_argument("--nsteps", type=int, default=None, help="Override total time steps.")
    parser.add_argument("--save-every", type=int, default=None, help="Override save interval.")
    parser.add_argument(
        "--initial-radius-cells",
        type=float,
        default=None,
        help="Override initial sphere radius [cells].",
    )
    parser.add_argument(
        "--undercooling-k",
        type=float,
        default=None,
        help="Override uniform undercooling [K].",
    )
    parser.add_argument(
        "--aniso-mode",
        choices=["full", "isotropic", "energetic_only", "kinetic_only"],
        default=None,
        help="Runtime anisotropy toggle mode.",
    )
    parser.add_argument(
        "--fit-end-time-s",
        type=float,
        default=None,
        help="Optional upper cutoff time [s] for growth-rate fitting.",
    )
    return parser.parse_args()


def eps_from_gamma(delta, gamma):
    return math.sqrt(8.0 * delta * gamma / (math.pi * math.pi))


def w_from_gamma(delta, gamma):
    return 4.0 * gamma / delta


def mij_from_M(delta, mobility):
    return (math.pi * math.pi / (8.0 * delta)) * mobility


def pick_override(arg_value, cfg_value, default_value):
    if arg_value is not None:
        return arg_value
    if cfg_value is not None:
        return cfg_value
    return default_value


def safe_case_name(label: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    return name.strip("_") or "case"


def align_direction_to_z_quaternion(direction_xyz) -> list[float]:
    src = np.array([direction_xyz], dtype=np.float64)
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    tgt = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    rot, _ = Rotation.align_vectors(tgt, src)
    return rot.as_quat().astype(np.float64).tolist()


def default_orientation_cases() -> list[dict]:
    return [
        {
            "label": "{100}",
            "orientation_type": "euler",
            "euler_deg": [0.0, 0.0, 0.0],
        },
        {
            "label": "{110}",
            "orientation_type": "euler",
            "euler_deg": [0.0, 45.0, 0.0],
        },
        {
            "label": "{111}",
            "orientation_type": "quaternion",
            "quaternion": align_direction_to_z_quaternion([1.0, 1.0, 1.0]),
        },
        {
            "label": "mid1",
            "orientation_type": "euler",
            "euler_deg": [0.0, 15.0, 0.0],
        },
        {
            "label": "mid2",
            "orientation_type": "euler",
            "euler_deg": [20.0, 35.0, 10.0],
        },
    ]


def resolve_orientation_cases(scan_cfg: dict) -> list[dict]:
    raw_cases = scan_cfg.get("orientation_cases")
    if not raw_cases:
        raw_cases = default_orientation_cases()

    cases: list[dict] = []
    for idx, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"orientation_cases[{idx-1}] must be a dict.")

        case_cfg = copy.deepcopy(raw)
        label = str(case_cfg.get("label", f"case_{idx:02d}"))
        quat = build_quaternion_from_config(case_cfg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            euler_deg = Rotation.from_quat(quat).as_euler("xyz", degrees=True)

        cases.append(
            {
                "case_index": idx,
                "label": label,
                "orientation_type": str(case_cfg.get("orientation_type", "euler")),
                "quaternion": [float(v) for v in quat],
                "euler_deg": [float(v) for v in euler_deg],
            }
        )

    return cases


def resolve_anisotropy_mode(mode: str, delta_a_base: float, ksi_base: float):
    mode = mode.lower()
    if mode == "full":
        return 1, 1, float(delta_a_base), float(ksi_base)
    if mode == "isotropic":
        return 0, 0, 0.0, 1.0
    if mode == "energetic_only":
        return 1, 1, float(delta_a_base), 1.0
    if mode == "kinetic_only":
        return 1, 0, 0.0, float(ksi_base)
    raise ValueError(f"Unknown aniso_mode: {mode}")


def save_solid_slice(phi: np.ndarray, out_path: Path, axis: str, index: int | None, title: str, dpi: int = 150) -> None:
    solid = 1.0 - phi[0]
    nx, ny, nz = solid.shape

    if axis == "xy":
        idx = nz // 2 if index is None else int(np.clip(index, 0, nz - 1))
        slc = solid[:, :, idx]
    elif axis == "xz":
        idx = ny // 2 if index is None else int(np.clip(index, 0, ny - 1))
        slc = solid[:, idx, :]
    elif axis == "yz":
        idx = nx // 2 if index is None else int(np.clip(index, 0, nx - 1))
        slc = solid[idx, :, :]
    else:
        raise ValueError(f"Unsupported slice axis: {axis}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(slc.T, cmap="magma", origin="lower", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("phi_solid")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_step_slices(phi: np.ndarray, case_dir: Path, axes: list[str], step: int, step_width: int) -> None:
    for axis in axes:
        out_path = case_dir / f"{axis}_{step:0{step_width}d}.png"
        save_solid_slice(phi, out_path, axis=axis, index=None, title=f"{axis} step {step}")


def save_final_slices(phi: np.ndarray, case_dir: Path, axes: list[str]) -> None:
    for axis in axes:
        out_path = case_dir / f"final_{axis}.png"
        save_solid_slice(phi, out_path, axis=axis, index=None, title=f"{axis} final")


def trilinear_sample(field: np.ndarray, x: float, y: float, z: float) -> float:
    nx, ny, nz = field.shape
    if x < 0.0 or y < 0.0 or z < 0.0 or x > (nx - 1) or y > (ny - 1) or z > (nz - 1):
        return float("nan")

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    z0 = int(math.floor(z))
    x1 = min(x0 + 1, nx - 1)
    y1 = min(y0 + 1, ny - 1)
    z1 = min(z0 + 1, nz - 1)

    tx = x - x0
    ty = y - y0
    tz = z - z0

    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return float(c0 * (1.0 - tz) + c1 * tz)


def max_distance_to_boundary(center_cells: np.ndarray, direction: np.ndarray, shape: tuple[int, int, int]) -> float:
    nx, ny, nz = shape
    max_index = np.array([nx - 1.0, ny - 1.0, nz - 1.0], dtype=np.float64)
    limits = []

    for c, d, upper in zip(center_cells, direction, max_index):
        if abs(d) < 1.0e-12:
            continue
        if d > 0.0:
            lim = (upper - c) / d
        else:
            lim = (0.0 - c) / d
        if lim > 0.0:
            limits.append(float(lim))

    if not limits:
        return 0.0
    return min(limits)


def directional_radius_one_side(
    phi_solid: np.ndarray,
    center_cells: np.ndarray,
    direction: np.ndarray,
    threshold: float,
    step_cells: float,
) -> float:
    d = np.asarray(direction, dtype=np.float64)
    d_norm = np.linalg.norm(d)
    if d_norm < 1.0e-12:
        return float("nan")
    d /= d_norm

    max_s = max_distance_to_boundary(center_cells, d, phi_solid.shape)
    if max_s <= 0.0:
        return float("nan")

    v0 = trilinear_sample(phi_solid, center_cells[0], center_cells[1], center_cells[2])
    if not np.isfinite(v0) or v0 < threshold:
        return float("nan")

    prev_s = 0.0
    prev_v = v0
    n_steps = max(2, int(math.ceil(max_s / step_cells)) + 1)

    for i in range(1, n_steps + 1):
        s = min(i * step_cells, max_s)
        pos = center_cells + d * s
        v = trilinear_sample(phi_solid, pos[0], pos[1], pos[2])
        if not np.isfinite(v):
            break

        if prev_v >= threshold and v < threshold:
            dv = prev_v - v
            if abs(dv) < 1.0e-12:
                return float(s)
            frac = (prev_v - threshold) / dv
            frac = float(np.clip(frac, 0.0, 1.0))
            return float(prev_s + frac * (s - prev_s))

        prev_s = s
        prev_v = v

    return float("nan")


def directional_radius_symmetric(
    phi_solid: np.ndarray,
    center_cells: np.ndarray,
    direction: np.ndarray,
    threshold: float = 0.5,
    step_cells: float = 0.5,
) -> float:
    rp = directional_radius_one_side(phi_solid, center_cells, direction, threshold, step_cells)
    rm = directional_radius_one_side(phi_solid, center_cells, -np.asarray(direction), threshold, step_cells)

    vals = [v for v in (rp, rm) if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def measure_observables(
    phi: np.ndarray,
    step: int,
    dt: float,
    dx: float,
    center_cells: np.ndarray,
    probe_step_cells: float,
) -> dict:
    phi_solid = phi[1].astype(np.float64)
    volume = float(np.sum(phi_solid) * (dx ** 3))

    if volume > 0.0:
        equivalent_radius = float((3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0))
    else:
        equivalent_radius = 0.0

    dirs = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "diag111": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }

    row = {
        "step": int(step),
        "time": float(step * dt),
        "volume": volume,
        "equivalent_radius": equivalent_radius,
    }

    for name, vec in dirs.items():
        r_cells = directional_radius_symmetric(
            phi_solid,
            center_cells=center_cells,
            direction=vec,
            threshold=0.5,
            step_cells=probe_step_cells,
        )
        row[f"radius_{name}"] = float(r_cells * dx) if np.isfinite(r_cells) else float("nan")

    return row


def fit_growth_rate(
    timeseries: list[dict],
    fit_fraction: float,
    fit_end_time_s: float | None = None,
) -> tuple[float, int, int, int]:
    if len(timeseries) < 2:
        return float("nan"), 0, 0, 0

    times = np.array([float(r["time"]) for r in timeseries], dtype=np.float64)
    req = np.array([float(r["equivalent_radius"]) for r in timeseries], dtype=np.float64)
    indices = np.arange(len(timeseries), dtype=np.int32)
    valid_mask = np.isfinite(times) & np.isfinite(req)

    if fit_end_time_s is not None and fit_end_time_s > 0.0:
        cutoff_mask = valid_mask & (times <= fit_end_time_s + 1.0e-15)
        if np.count_nonzero(cutoff_mask) >= 2:
            sel_idx = indices[cutoff_mask]
            x = times[cutoff_mask]
            y = req[cutoff_mask]
            slope = float(np.polyfit(x, y, 1)[0])
            return slope, int(sel_idx[0]), int(sel_idx[-1]), int(x.size)

    valid_indices = indices[valid_mask]
    valid_times = times[valid_mask]
    valid_req = req[valid_mask]

    if valid_times.size < 2:
        return float("nan"), 0, 0, 0

    n_use = max(2, int(math.ceil(valid_times.size * fit_fraction)))
    x = valid_times[-n_use:]
    y = valid_req[-n_use:]
    sel_idx = valid_indices[-n_use:]

    slope = float(np.polyfit(x, y, 1)[0])
    return slope, int(sel_idx[0]), int(sel_idx[-1]), int(x.size)


def anisotropy_index_from_row(row: dict) -> float:
    keys = ("radius_x", "radius_y", "radius_z", "radius_diag111")
    vals = np.array([float(row.get(k, float("nan"))) for k in keys], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    mean_v = float(np.mean(vals))
    if abs(mean_v) < 1.0e-30:
        return float("nan")
    return float((np.max(vals) - np.min(vals)) / mean_v)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_growth_rate(summary_rows: list[dict], out_path: Path) -> None:
    labels = [str(r["orientation_label"]) for r in summary_rows]
    values = np.array([float(r["fitted_growth_rate"]) for r in summary_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 5))
    if values.size > 0:
        draw_values = np.nan_to_num(values, nan=0.0)
        bars = ax.bar(labels, draw_values, color="tab:blue")
        for bar, val in zip(bars, values):
            txt = "nan" if not np.isfinite(val) else f"{val:.3e}"
            ax.text(bar.get_x() + bar.get_width() * 0.5, bar.get_height(), txt, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("fitted growth rate [m/s]")
    ax.set_title("Growth Rate vs Orientation")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_req_timeseries(case_timeseries: dict[str, list[dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, rows in case_timeseries.items():
        t = np.array([float(r["time"]) for r in rows], dtype=np.float64)
        r = np.array([float(r["equivalent_radius"]) for r in rows], dtype=np.float64)
        mask = np.isfinite(t) & np.isfinite(r)
        if np.count_nonzero(mask) < 2:
            continue
        ax.plot(t[mask], r[mask], marker="o", ms=3, lw=1.5, label=label)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("equivalent radius [m]")
    ax.set_title("Equivalent Radius vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_runtime = copy.deepcopy(cfg)

    scan_cfg = cfg_runtime.get("orientation_scan_3d", {})

    nx = int(pick_override(args.nx, scan_cfg.get("nx"), 96))
    ny = int(pick_override(args.ny, scan_cfg.get("ny"), 96))
    nz = int(pick_override(args.nz, scan_cfg.get("nz"), 96))
    nsteps = int(pick_override(args.nsteps, scan_cfg.get("nsteps"), 150))
    save_every = int(pick_override(args.save_every, scan_cfg.get("save_every"), 10))
    initial_radius_cells = float(
        pick_override(args.initial_radius_cells, scan_cfg.get("initial_radius_cells"), 10.0)
    )
    undercooling_k = float(pick_override(args.undercooling_k, scan_cfg.get("undercooling_K"), 1.0))
    aniso_mode = str(pick_override(args.aniso_mode, scan_cfg.get("aniso_mode"), "full")).lower()
    fit_end_time_s_cfg = scan_cfg.get("fit_end_time_s", 0.015)
    fit_end_time_s = pick_override(args.fit_end_time_s, fit_end_time_s_cfg, 0.015)
    if fit_end_time_s is not None:
        fit_end_time_s = float(fit_end_time_s)
        if fit_end_time_s <= 0.0:
            fit_end_time_s = None

    if nsteps <= 0:
        raise ValueError("nsteps must be positive.")
    if save_every <= 0:
        raise ValueError("save_every must be positive.")

    dt_grid = float(cfg_runtime["grid"]["dt"])
    dx = float(cfg_runtime["grid"]["dx"])
    dy = float(cfg_runtime["grid"]["dy"])
    dz = float(cfg_runtime["grid"]["dz"])
    if (dx != dy) or (dx != dz):
        raise ValueError("3D kernels assume dx == dy == dz.")

    T_melt = float(cfg_runtime["physical"]["T_melt"])
    Sf = float(cfg_runtime["physical"]["Sf"])

    delta = float(cfg_runtime["interface"]["delta_factor"]) * dx
    gamma_100 = float(cfg_runtime["interface"]["gamma_100"])
    gamma_gb = float(cfg_runtime["interface"]["gamma_GB"])

    a0 = float(cfg_runtime["anisotropy"]["a0_deg"]) * math.pi / 180.0
    delta_a = float(cfg_runtime["anisotropy"]["delta_a"])
    mu_a = float(cfg_runtime["anisotropy"]["mu_a"])
    p_round = float(cfg_runtime["anisotropy"]["p_round"])
    ksi = float(cfg_runtime["anisotropy"]["ksi"])
    theta_c_rad = float(cfg_runtime["anisotropy"]["omg_deg"]) * math.pi / 180.0

    m_sl = float(cfg_runtime["mobility"]["M_SL"])
    m_gb = m_sl * float(cfg_runtime["mobility"]["M_GB_ratio"])

    max_grains = int(cfg_runtime["gpu"]["MAX_GRAINS"])
    config_kmax = int(cfg_runtime["gpu"]["KMAX"])
    threadsperblock = tuple(int(v) for v in cfg_runtime["gpu"]["threads_per_block"])

    if len(threadsperblock) != 3:
        raise ValueError("gpu.threads_per_block must have 3 entries for 3D.")
    if config_kmax != KERNEL_KMAX:
        raise ValueError(
            f"config_3d.yaml gpu.KMAX={config_kmax} does not match compile-time "
            f"KMAX={KERNEL_KMAX} in src/gpu_kernels_3d.py."
        )
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. A CUDA-capable GPU is required for 3D runs.")

    orientation_cases = resolve_orientation_cases(scan_cfg)
    if len(orientation_cases) < 1:
        raise ValueError("No orientation cases were resolved.")

    save_slices = list(scan_cfg.get("save_slices", ["xy", "xz", "yz"]))
    for axis in save_slices:
        if axis not in ("xy", "xz", "yz"):
            raise ValueError(f"Unsupported axis in save_slices: {axis}")

    fit_fraction = float(scan_cfg.get("fit_fraction", 0.5))
    fit_fraction = min(1.0, max(0.1, fit_fraction))
    probe_step_cells = float(scan_cfg.get("direction_probe_step_cells", 0.5))
    probe_step_cells = max(0.1, probe_step_cells)
    g2_floor_scale = float(scan_cfg.get("g2_floor_scale", 0.1))
    g2_floor_f = np.float32((g2_floor_scale / dx) ** 2)

    center_x_frac = float(scan_cfg.get("center_x_frac", 0.5))
    center_y_frac = float(scan_cfg.get("center_y_frac", 0.5))
    center_z_frac = float(scan_cfg.get("center_z_frac", 0.5))
    center_x = center_x_frac * nx * dx
    center_y = center_y_frac * ny * dy
    center_z = center_z_frac * nz * dz
    center_cells = np.array([center_x / dx, center_y / dy, center_z / dz], dtype=np.float64)

    if args.out_dir:
        run_dir = Path(args.out_dir)
    else:
        out_base = Path(scan_cfg.get("outdir", "result/orientation_scan_3d"))
        run_dir = out_base / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    number_of_grain = 2
    if number_of_grain > max_grains:
        raise RuntimeError(f"number_of_grain={number_of_grain} exceeds MAX_GRAINS={max_grains}.")

    eps0_sl = eps_from_gamma(delta, gamma_100)
    w0_sl = w_from_gamma(delta, gamma_100)
    m_sl_phi = mij_from_M(delta, m_sl)
    eps_gb = eps_from_gamma(delta, gamma_gb)
    w_gb = w_from_gamma(delta, gamma_gb)
    m_gb_phi = mij_from_M(delta, m_gb)

    wij, aij, mij = build_interaction_matrices(
        number_of_grain,
        eps0_sl,
        w0_sl,
        m_sl_phi,
        eps_gb,
        w_gb,
        m_gb_phi,
    )

    enable_aniso, enable_torque, delta_a_runtime, ksi_runtime = resolve_anisotropy_mode(aniso_mode, delta_a, ksi)

    print("Mode                  : orientation_scan_3d")
    print(f"Config                : {args.config}")
    print(f"Grid                  : {nx}x{ny}x{nz}, dx={dx:.2e}, dt={dt_grid:.2e}, nsteps={nsteps}")
    print(f"Initial radius        : {initial_radius_cells:.2f} cells")
    print(f"Undercooling          : {undercooling_k:.3f} K")
    print(f"Anisotropy mode       : {aniso_mode}")
    print(f"  enable_anisotropy   : {bool(enable_aniso)}")
    print(f"  enable_torque       : {bool(enable_torque)}")
    print(f"  delta_a (runtime)   : {delta_a_runtime:.5f}")
    print(f"  ksi (runtime)       : {ksi_runtime:.5f}")
    if fit_end_time_s is None:
        print(f"Fit window            : trailing {fit_fraction:.2f} fraction of saved points")
    else:
        print(f"Fit window            : time <= {fit_end_time_s:.5f} s")
    print(f"Save interval         : every {save_every} steps")
    print(f"Output dir            : {run_dir}")
    print("Orientation cases:")
    for case in orientation_cases:
        e = case["euler_deg"]
        q = case["quaternion"]
        print(
            f"  - {case['label']}: "
            f"euler_deg=({e[0]:+.3f}, {e[1]:+.3f}, {e[2]:+.3f}), "
            f"quat=({q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f})"
        )

    save_run_config(
        str(run_dir),
        cfg_runtime,
        {
            "mode": "orientation_scan_3d",
            "config_path": args.config,
            "grid_runtime": {"nx": nx, "ny": ny, "nz": nz, "nsteps": nsteps, "save_every": save_every},
            "initial_radius_cells": initial_radius_cells,
            "undercooling_K": undercooling_k,
            "aniso_mode": aniso_mode,
            "enable_anisotropy": bool(enable_aniso),
            "enable_torque": bool(enable_torque),
            "delta_a_runtime": delta_a_runtime,
            "ksi_runtime": ksi_runtime,
            "fit_fraction": fit_fraction,
            "fit_end_time_s": fit_end_time_s,
            "probe_step_cells": probe_step_cells,
            "g2_floor_scale": g2_floor_scale,
            "save_slices": save_slices,
            "orientation_cases": orientation_cases,
        },
    )

    d_wij = cuda.to_device(wij.astype(np.float32))
    d_aij = cuda.to_device(aij.astype(np.float32))
    d_mij = cuda.to_device(mij.astype(np.float32))

    blockspergrid = (
        math.ceil(nx / threadsperblock[0]),
        math.ceil(ny / threadsperblock[1]),
        math.ceil(nz / threadsperblock[2]),
    )

    radius_m = initial_radius_cells * dx
    temp_iso = np.full((nx, ny, nz), T_melt - undercooling_k, dtype=np.float32)
    step_width = max(4, len(str(nsteps)))

    summary_rows: list[dict] = []
    series_map: dict[str, list[dict]] = {}

    print("\nStarting orientation scan ...")

    for case in orientation_cases:
        label = str(case["label"])
        case_name = f"{int(case['case_index']):02d}_{safe_case_name(label)}"
        case_dir = run_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        quat = np.array(case["quaternion"], dtype=np.float64)
        grain_quats = np.zeros((number_of_grain, 4), dtype=np.float64)
        grain_quats[0] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        grain_quats[1] = quat
        grain_n111 = compute_rotated_n111(grain_quats)

        phi0 = init_singlemode_sphere_3d(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            delta,
            radius_m,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            solid_gid=1,
        )

        timeseries: list[dict] = []
        timeseries.append(
            measure_observables(
                phi0,
                step=0,
                dt=dt_grid,
                dx=dx,
                center_cells=center_cells,
                probe_step_cells=probe_step_cells,
            )
        )
        save_step_slices(phi0, case_dir, save_slices, step=0, step_width=step_width)

        d_phi = cuda.to_device(phi0.astype(np.float32))
        d_phi_new = cuda.to_device(phi0.astype(np.float32).copy())
        d_temp = cuda.to_device(temp_iso.copy())
        d_mf = cuda.to_device(np.zeros((max_grains, nx, ny, nz), dtype=np.int32))
        d_nf = cuda.to_device(np.zeros((nx, ny, nz), dtype=np.int32))
        d_n111 = cuda.to_device(grain_n111.astype(np.float32))

        print(f"\nCase {case_name} ({label})")
        print("  Running time evolution ...")

        final_phi_host = phi0
        for nstep in range(1, nsteps + 1):
            kernel_update_nfmf_3d[blockspergrid, threadsperblock](
                d_phi, d_mf, d_nf, nx, ny, nz, number_of_grain
            )

            kernel_update_phasefield_active_3d_switchable[blockspergrid, threadsperblock](
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
                np.float32(dx),
                np.float32(dt_grid),
                np.float32(T_melt),
                np.float32(Sf),
                np.float32(eps0_sl),
                np.float32(w0_sl),
                np.float32(a0),
                np.float32(delta_a_runtime),
                np.float32(mu_a),
                np.float32(p_round),
                g2_floor_f,
                np.float32(ksi_runtime),
                np.float32(theta_c_rad),
                np.int32(enable_aniso),
                np.int32(enable_torque),
            )

            d_phi, d_phi_new = d_phi_new, d_phi

            if (nstep % save_every == 0) or (nstep == nsteps):
                final_phi_host = d_phi.copy_to_host()
                row = measure_observables(
                    final_phi_host,
                    step=nstep,
                    dt=dt_grid,
                    dx=dx,
                    center_cells=center_cells,
                    probe_step_cells=probe_step_cells,
                )
                timeseries.append(row)
                save_step_slices(final_phi_host, case_dir, save_slices, step=nstep, step_width=step_width)
                print(f"  saved step_{nstep}")

        save_final_slices(final_phi_host, case_dir, save_slices)

        timeseries_fields = [
            "step",
            "time",
            "volume",
            "equivalent_radius",
            "radius_x",
            "radius_y",
            "radius_z",
            "radius_diag111",
        ]
        write_csv(case_dir / "timeseries.csv", timeseries, timeseries_fields)

        growth_rate, fit_start, fit_end, fit_points = fit_growth_rate(
            timeseries,
            fit_fraction,
            fit_end_time_s=fit_end_time_s,
        )
        final_row = timeseries[-1]
        aniso_idx = anisotropy_index_from_row(final_row)

        euler = case["euler_deg"]
        q = case["quaternion"]
        summary_rows.append(
            {
                "case_name": case_name,
                "orientation_label": label,
                "orientation_type": case["orientation_type"],
                "euler_x_deg": float(euler[0]),
                "euler_y_deg": float(euler[1]),
                "euler_z_deg": float(euler[2]),
                "quat_x": float(q[0]),
                "quat_y": float(q[1]),
                "quat_z": float(q[2]),
                "quat_w": float(q[3]),
                "final_volume": float(final_row["volume"]),
                "final_equivalent_radius": float(final_row["equivalent_radius"]),
                "fitted_growth_rate": float(growth_rate),
                "anisotropy_index": float(aniso_idx),
                "fit_start_step": int(timeseries[fit_start]["step"] if fit_points > 0 else 0),
                "fit_end_step": int(timeseries[fit_end]["step"] if fit_points > 0 else 0),
                "fit_start_time": float(timeseries[fit_start]["time"] if fit_points > 0 else 0.0),
                "fit_end_time": float(timeseries[fit_end]["time"] if fit_points > 0 else 0.0),
                "fit_points": int(fit_points),
            }
        )

        series_map[label] = timeseries
        print(
            f"  done: final r_eq={final_row['equivalent_radius']:.4e} m, "
            f"fitted growth rate={growth_rate:.4e} m/s"
        )

    summary_fields = [
        "case_name",
        "orientation_label",
        "orientation_type",
        "euler_x_deg",
        "euler_y_deg",
        "euler_z_deg",
        "quat_x",
        "quat_y",
        "quat_z",
        "quat_w",
        "final_volume",
        "final_equivalent_radius",
        "fitted_growth_rate",
        "anisotropy_index",
        "fit_start_step",
        "fit_end_step",
        "fit_start_time",
        "fit_end_time",
        "fit_points",
    ]
    write_csv(run_dir / "summary.csv", summary_rows, summary_fields)
    plot_growth_rate(summary_rows, run_dir / "growth_rate_vs_orientation.png")
    plot_req_timeseries(series_map, run_dir / "equivalent_radius_vs_time.png")

    print("\nScan finished.")
    print(f"Summary CSV  : {run_dir / 'summary.csv'}")
    print(f"Growth figure: {run_dir / 'growth_rate_vs_orientation.png'}")
    print(f"Req figure   : {run_dir / 'equivalent_radius_vs_time.png'}")


if __name__ == "__main__":
    main()
