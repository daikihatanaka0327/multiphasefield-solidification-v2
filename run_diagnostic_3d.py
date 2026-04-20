"""
run_diagnostic_3d.py
====================
Single-crystal 3D isothermal growth diagnostic.

Purpose
-------
Place a small sphere-shaped solid grain at the domain center in an
undercooled 3D melt (isothermal, no cooling) and run the 3D phase
field evolution. The purpose is to verify whether anisotropic
surface energy and kinetic anisotropy produce the expected faceted
growth morphology (octahedral shape in 3D, reflecting {111} facets).

What this mode checks
---------------------
  * Whether the 3D anisotropy functions a(theta), b(theta) are effective.
  * Whether the n111 rotation computation is correct in 3D.
  * Whether the 3D torque term produces faceted growth.
  * The baseline 3D growth morphology without grain-grain interaction.

Expected result (Zhu et al. 2023, Fig.2i-k)
-------------------------------------------
At dT = 4K, a single silicon crystal grows into an octahedral shape
with 8 facets aligned to {111} orientations. If the simulation yields
a spherical (isotropic) grain, the anisotropy is not correctly
implemented or parameterized.

Phases
------
  gid = 0 : liquid
  gid = 1 : single solid grain

Output
------
  result/diagnostic_3d/.../step_0_*.png        (xy/xz/yz center slices)
  result/diagnostic_3d/.../step_{N}_*.png      (every save_every steps)
  result/diagnostic_3d/config_snapshot.yaml
  result/diagnostic_3d/run_params.yaml
"""

import argparse
import math
import os

import numpy as np
import yaml
from numba import cuda

from src.gpu_kernels_3d import (
    KMAX as KERNEL_KMAX,
    kernel_update_nfmf_3d,
    kernel_update_phasefield_active_3d,
)
from src.orientation_utils import build_quaternion_from_config, compute_rotated_n111
from src.seed_modes_3d import (
    init_singlemode_sphere_3d,
    build_interaction_matrices,
)
from src.plot_utils_3d import save_phase_map_slice_3d, save_run_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D single-crystal isothermal diagnostic.")
    parser.add_argument("--config", default="config_3d.yaml", help="Path to config YAML.")
    return parser.parse_args()


def eps_from_gamma(delta, gamma):
    return math.sqrt(8.0 * delta * gamma / (math.pi * math.pi))


def w_from_gamma(delta, gamma):
    return 4.0 * gamma / delta


def mij_from_M(delta, mobility):
    return (math.pi * math.pi / (8.0 * delta)) * mobility


def save_three_center_slices(phi, out_dir, number_of_grain, prefix, title_suffix):
    """Save xy, xz, yz center slices using save_phase_map_slice_3d."""
    for axis in ("xy", "xz", "yz"):
        save_phase_map_slice_3d(
            phi,
            out_dir,
            f"{prefix}_{axis}.png",
            number_of_grain,
            axis=axis,
            index=None,  # center slice
            title=f"{title_suffix} ({axis})",
        )


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pi = math.pi

    # Grid
    nx = int(cfg["grid"]["nx"])
    ny = int(cfg["grid"]["ny"])
    nz = int(cfg["grid"]["nz"])
    dx = float(cfg["grid"]["dx"])
    dy = float(cfg["grid"]["dy"])
    dz = float(cfg["grid"]["dz"])
    dt = float(cfg["grid"]["dt"])
    nsteps = int(cfg["grid"]["nsteps"])

    # Physical
    T_melt = float(cfg["physical"]["T_melt"])
    Sf     = float(cfg["physical"]["Sf"])

    # Interface
    delta     = float(cfg["interface"]["delta_factor"]) * dx
    gamma_100 = float(cfg["interface"]["gamma_100"])
    gamma_GB  = float(cfg["interface"]["gamma_GB"])

    # Anisotropy
    a0          = float(cfg["anisotropy"]["a0_deg"]) * pi / 180.0
    delta_a     = float(cfg["anisotropy"]["delta_a"])
    mu_a        = float(cfg["anisotropy"]["mu_a"])
    p_round     = float(cfg["anisotropy"]["p_round"])
    ksi         = float(cfg["anisotropy"]["ksi"])
    theta_c_rad = float(cfg["anisotropy"]["omg_deg"]) * pi / 180.0

    # Mobility
    M_SL = float(cfg["mobility"]["M_SL"])
    M_GB = M_SL * float(cfg["mobility"]["M_GB_ratio"])

    # GPU
    MAX_GRAINS      = int(cfg["gpu"]["MAX_GRAINS"])
    config_kmax     = int(cfg["gpu"]["KMAX"])
    threadsperblock = tuple(int(v) for v in cfg["gpu"]["threads_per_block"])

    # Output
    save_every = int(cfg["output"]["save_every"])

    # diagnostic_3d-specific settings
    dg_cfg = cfg.get("diagnostic_3d", {})
    initial_radius_cells = int(dg_cfg.get("initial_radius_cells", 10))
    center_x_frac        = float(dg_cfg.get("center_x_frac", 0.5))
    center_y_frac        = float(dg_cfg.get("center_y_frac", 0.5))
    center_z_frac        = float(dg_cfg.get("center_z_frac", 0.5))
    undercooling_K       = float(dg_cfg.get("undercooling_K", 4.0))

    out_dir = "result/diagnostic_3d"
    os.makedirs(out_dir, exist_ok=True)

    # Sanity checks (mirroring run_randommode_3d.py)
    if (dx != dy) or (dx != dz):
        raise ValueError("3D kernels assume dx == dy == dz.")
    if len(threadsperblock) != 3:
        raise ValueError("gpu.threads_per_block must have 3 entries for 3D.")
    if config_kmax != KERNEL_KMAX:
        raise ValueError(
            f"config gpu.KMAX={config_kmax} does not match compile-time "
            f"KMAX={KERNEL_KMAX} in src/gpu_kernels_3d.py."
        )
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. A CUDA-capable GPU is required for 3D runs.")

    # Phases: liquid (gid=0) + single solid (gid=1)
    n_solid = 1
    number_of_grain = n_solid + 1
    if number_of_grain > MAX_GRAINS:
        raise RuntimeError(
            f"number_of_grain={number_of_grain} exceeds MAX_GRAINS={MAX_GRAINS}."
        )

    # Interface coefficients
    eps0_sl  = eps_from_gamma(delta, gamma_100)
    w0_sl    = w_from_gamma(delta, gamma_100)
    m_sl_phi = mij_from_M(delta, M_SL)
    eps_GB   = eps_from_gamma(delta, gamma_GB)
    w_GB     = w_from_gamma(delta, gamma_GB)
    m_GB_phi = mij_from_M(delta, M_GB)

    # Grain orientations
    grain_quaternions = np.zeros((number_of_grain, 4), dtype=np.float64)
    grain_quaternions[0] = np.array([0.0, 0.0, 0.0, 1.0])  # liquid dummy
    grain_quaternions[1] = build_quaternion_from_config(dg_cfg)
    grain_n111 = compute_rotated_n111(grain_quaternions)

    # Interaction matrices
    wij, aij, mij = build_interaction_matrices(
        number_of_grain,
        eps0_sl, w0_sl, m_sl_phi,
        eps_GB, w_GB, m_GB_phi,
    )

    # Initial conditions: sphere of solid in liquid + uniform undercooled T
    radius   = initial_radius_cells * dx
    center_x = center_x_frac * nx * dx
    center_y = center_y_frac * ny * dy
    center_z = center_z_frac * nz * dz

    phi_cpu = init_singlemode_sphere_3d(
        nx, ny, nz, dx, dy, dz, delta, radius,
        center_x=center_x, center_y=center_y, center_z=center_z,
        solid_gid=1,
    )

    T_iso = T_melt - undercooling_K
    temp_cpu = np.full((nx, ny, nz), T_iso, dtype=np.float32)

    # Startup log
    print("Mode                : diagnostic_3d")
    print(f"Config              : {args.config}")
    print(f"Grid                : {nx}x{ny}x{nz}, dx={dx:.1e}, dt={dt:.1e}, nsteps={nsteps}")
    print(f"Threads/block       : {threadsperblock}")
    print(f"Phases              : {number_of_grain}")
    print(f"Diagnostic mode     : 3D single-crystal isothermal growth")
    print(f"Undercooling        : {undercooling_K} K  (T_iso = {T_iso} K)")
    print(f"Initial sphere      : radius = {initial_radius_cells} cells = {initial_radius_cells * dx:.2e} m")
    print(f"Sphere center (frac): ({center_x_frac:.2f}, {center_y_frac:.2f}, {center_z_frac:.2f})")
    print(f"Grain quaternion    : {grain_quaternions[1]}")
    print(f"grain_n111[1]:")
    for t in range(8):
        n = grain_n111[1, t]
        print(f"  t={t}: ({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f})  norm={np.linalg.norm(n):.3f}")
    print(f"Output dir          : {out_dir}")

    # Save initial state (3 center slices)
    save_three_center_slices(
        phi_cpu, out_dir, number_of_grain,
        prefix="step_0", title_suffix="diagnostic_3d -- step 0",
    )
    print("Saved step_0 slices (xy, xz, yz)")

    save_run_config(out_dir, cfg, {
        "mode": "diagnostic_3d",
        "config_path": args.config,
        "number_of_grain": number_of_grain,
        "n_solid": n_solid,
        "initial_radius_cells": initial_radius_cells,
        "center_x_frac": center_x_frac,
        "center_y_frac": center_y_frac,
        "center_z_frac": center_z_frac,
        "undercooling_K": undercooling_K,
        "T_iso": T_iso,
        "threads_per_block": list(threadsperblock),
        "save_every": save_every,
        "out_dir": out_dir,
        "grain_quaternions": grain_quaternions.tolist(),
    })

    # GPU transfer
    d_phi     = cuda.to_device(phi_cpu.astype(np.float32))
    d_phi_new = cuda.to_device(phi_cpu.astype(np.float32).copy())
    d_temp    = cuda.to_device(temp_cpu.astype(np.float64))
    d_mf      = cuda.to_device(np.zeros((MAX_GRAINS, nx, ny, nz), dtype=np.int32))
    d_nf      = cuda.to_device(np.zeros((nx, ny, nz), dtype=np.int32))
    d_wij     = cuda.to_device(wij.astype(np.float32))
    d_aij     = cuda.to_device(aij.astype(np.float32))
    d_mij     = cuda.to_device(mij.astype(np.float32))
    d_n111    = cuda.to_device(grain_n111.astype(np.float32))

    blockspergrid = (
        math.ceil(nx / threadsperblock[0]),
        math.ceil(ny / threadsperblock[1]),
        math.ceil(nz / threadsperblock[2]),
    )

    T_melt_f   = np.float32(T_melt)
    Sf_f       = np.float32(Sf)
    g2_floor_f = np.float64((0.001 / dx) ** 2)

    # Main time evolution loop (no temperature update -- isothermal)
    print(f"\nStarting time evolution ({nsteps} steps, saving every {save_every}) ...")

    for nstep in range(1, nsteps + 1):
        kernel_update_nfmf_3d[blockspergrid, threadsperblock](
            d_phi, d_mf, d_nf, nx, ny, nz, number_of_grain
        )

        kernel_update_phasefield_active_3d[blockspergrid, threadsperblock](
            d_phi, d_phi_new, d_temp, d_mf, d_nf,
            d_wij, d_aij, d_mij, d_n111,
            nx, ny, nz, number_of_grain,
            np.float32(dx), np.float32(dt),
            T_melt_f, Sf_f,
            np.float32(eps0_sl), np.float32(w0_sl),
            np.float32(a0), np.float32(delta_a), np.float32(mu_a), np.float32(p_round),
            g2_floor_f, np.float32(ksi), np.float32(theta_c_rad),
        )

        d_phi, d_phi_new = d_phi_new, d_phi

        if nstep % save_every == 0:
            current_phi = d_phi.copy_to_host()
            save_three_center_slices(
                current_phi, out_dir, number_of_grain,
                prefix=f"step_{nstep}",
                title_suffix=f"diagnostic_3d -- step {nstep}",
            )
            print(f"saved step_{nstep}")

    print("\nDone.")


if __name__ == "__main__":
    main()
