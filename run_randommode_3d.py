"""
run_randommode_3d.py
====================
3D multi-grain polycrystal solidification with Voronoi random grain structure.

This is the 3D counterpart of run_randommode.py. The seed layer is generated
on the x-y plane and grows along +z.
"""

import argparse
import copy
import math
import os

import numpy as np
import yaml
from numba import cuda

from src.gpu_kernels_3d import (
    KMAX as KERNEL_KMAX,
    kernel_update_nfmf_3d,
    kernel_update_phasefield_active_3d,
    kernel_update_temp_3d,
)
from src.orientation_utils import (
    assign_quaternions_to_grains,
    compute_rotated_n111,
)
from src.seed_modes_3d import (
    generate_random_grain_map_3d,
    init_phi_from_grain_map_3d,
    init_temperature_field_3d,
    build_interaction_matrices,
)
from src.plot_utils_3d import (
    save_phase_map_slice_3d,
    save_interface_position_3d,
    save_run_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D random-mode solidification.")
    parser.add_argument("--config", default="config_3d.yaml", help="Path to config YAML.")
    parser.add_argument("--out-dir", default=None, help="Override output directory.")
    parser.add_argument("--nx", type=int, default=None, help="Override grid nx.")
    parser.add_argument("--ny", type=int, default=None, help="Override grid ny.")
    parser.add_argument("--nz", type=int, default=None, help="Override grid nz.")
    parser.add_argument("--nsteps", type=int, default=None, help="Override total time steps.")
    parser.add_argument("--save-every", type=int, default=None, help="Override save interval.")
    parser.add_argument("--seed-height", type=int, default=None, help="Override initial seed height.")
    parser.add_argument("--n-solid", type=int, default=None, help="Override number of solid grains.")
    return parser.parse_args()


def eps_from_gamma(delta, gamma):
    return math.sqrt(8.0 * delta * gamma / (math.pi * math.pi))


def w_from_gamma(delta, gamma):
    return 4.0 * gamma / delta


def mij_from_M(delta, mobility):
    return (math.pi * math.pi / (8.0 * delta)) * mobility


def save_requested_slices(phi, out_dir, number_of_grain, save_slices, slice_index, prefix, title_suffix):
    for axis in save_slices:
        save_phase_map_slice_3d(
            phi,
            out_dir,
            f"{prefix}_{axis}.png",
            number_of_grain,
            axis=axis,
            index=slice_index,
            title=f"{title_suffix} ({axis})",
        )

    save_interface_position_3d(
        phi,
        out_dir,
        f"{prefix}_interface.png",
        title=f"Interface z position -- {title_suffix}",
    )


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_runtime = copy.deepcopy(cfg)

    if args.nx is not None:
        cfg_runtime["grid"]["nx"] = args.nx
    if args.ny is not None:
        cfg_runtime["grid"]["ny"] = args.ny
    if args.nz is not None:
        cfg_runtime["grid"]["nz"] = args.nz
    if args.nsteps is not None:
        cfg_runtime["grid"]["nsteps"] = args.nsteps
    if args.save_every is not None:
        cfg_runtime["output"]["save_every"] = args.save_every
    if args.seed_height is not None:
        cfg_runtime.setdefault("randommode", {})["seed_height"] = args.seed_height
    if args.n_solid is not None:
        cfg_runtime.setdefault("randommode", {})["n_solid"] = args.n_solid

    pi = math.pi

    nx = int(cfg_runtime["grid"]["nx"])
    ny = int(cfg_runtime["grid"]["ny"])
    nz = int(cfg_runtime["grid"]["nz"])
    dx = float(cfg_runtime["grid"]["dx"])
    dy = float(cfg_runtime["grid"]["dy"])
    dz = float(cfg_runtime["grid"]["dz"])
    dt = float(cfg_runtime["grid"]["dt"])
    nsteps = int(cfg_runtime["grid"]["nsteps"])

    T_melt = float(cfg_runtime["physical"]["T_melt"])
    G = float(cfg_runtime["physical"]["G"])
    V_pulling = float(cfg_runtime["physical"]["V_pulling"])
    Sf = float(cfg_runtime["physical"]["Sf"])

    delta = float(cfg_runtime["interface"]["delta_factor"]) * dx
    gamma_100 = float(cfg_runtime["interface"]["gamma_100"])
    gamma_GB = float(cfg_runtime["interface"]["gamma_GB"])

    a0 = float(cfg_runtime["anisotropy"]["a0_deg"]) * pi / 180.0
    delta_a = float(cfg_runtime["anisotropy"]["delta_a"])
    mu_a = float(cfg_runtime["anisotropy"]["mu_a"])
    p_round = float(cfg_runtime["anisotropy"]["p_round"])
    ksi = float(cfg_runtime["anisotropy"]["ksi"])
    theta_c_rad = float(cfg_runtime["anisotropy"]["omg_deg"]) * pi / 180.0

    M_SL = float(cfg_runtime["mobility"]["M_SL"])
    M_GB = M_SL * float(cfg_runtime["mobility"]["M_GB_ratio"])

    MAX_GRAINS = int(cfg_runtime["gpu"]["MAX_GRAINS"])
    config_kmax = int(cfg_runtime["gpu"]["KMAX"])
    threadsperblock = tuple(int(v) for v in cfg_runtime["gpu"]["threads_per_block"])

    save_every = int(cfg_runtime["output"]["save_every"])
    save_slices = list(cfg_runtime["output"].get("save_slices", ["xy", "xz"]))
    slice_index = cfg_runtime["output"].get("slice_index")

    rm_cfg = cfg_runtime.get("randommode", {})
    seed_height = int(rm_cfg.get("seed_height", 32))
    n_solid = int(rm_cfg.get("n_solid", 10))
    random_seed = int(rm_cfg.get("random_seed", 42))
    orientation_mode = str(rm_cfg.get("orientation_mode", "random"))
    orientation_seed = int(rm_cfg.get("orientation_seed", 42))
    orientation_csv = rm_cfg.get("orientation_csv", None) or None

    out_dir = args.out_dir or cfg_runtime.get("output", {}).get("dir", "result/randommode_3d")
    os.makedirs(out_dir, exist_ok=True)

    if (dx != dy) or (dx != dz):
        raise ValueError("3D kernels assume dx == dy == dz.")
    if len(threadsperblock) != 3:
        raise ValueError("gpu.threads_per_block must have 3 entries for 3D.")
    if config_kmax != KERNEL_KMAX:
        raise ValueError(
            f"config_3d.yaml gpu.KMAX={config_kmax} does not match compile-time "
            f"KMAX={KERNEL_KMAX} in src/gpu_kernels_3d.py."
        )

    number_of_grain = n_solid + 1
    if number_of_grain > MAX_GRAINS:
        raise RuntimeError(
            f"number_of_grain={number_of_grain} exceeds MAX_GRAINS={MAX_GRAINS}. "
            "Reduce randommode.n_solid or increase gpu.MAX_GRAINS."
        )

    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. A CUDA-capable GPU is required for 3D runs.")

    eps0_sl = eps_from_gamma(delta, gamma_100)
    w0_sl = w_from_gamma(delta, gamma_100)
    m_sl_phi = mij_from_M(delta, M_SL)
    eps_GB = eps_from_gamma(delta, gamma_GB)
    w_GB = w_from_gamma(delta, gamma_GB)
    m_GB_phi = mij_from_M(delta, M_GB)

    grain_quaternions = assign_quaternions_to_grains(
        n_solid,
        mode=orientation_mode,
        orientation_seed=orientation_seed,
        csv_path=orientation_csv,
    )
    grain_n111 = compute_rotated_n111(grain_quaternions)

    wij, aij, mij = build_interaction_matrices(
        number_of_grain,
        eps0_sl, w0_sl, m_sl_phi,
        eps_GB, w_GB, m_GB_phi,
    )

    grain_map = generate_random_grain_map_3d(nx, ny, n_solid, random_seed=random_seed)
    phi_cpu = init_phi_from_grain_map_3d(
        grain_map, n_solid, nx, ny, nz, dz, delta, seed_height
    )
    temp_cpu = init_temperature_field_3d(nx, ny, nz, T_melt, G, dz, seed_height)

    print("Mode             : randommode_3d")
    print(f"Config           : {args.config}")
    print(f"Grid             : {nx}x{ny}x{nz}, dx={dx:.1e}, dt={dt:.1e}, nsteps={nsteps}")
    print(f"Threads/block    : {threadsperblock}")
    print(f"Seed height      : {seed_height}")
    print(f"n_solid          : {n_solid}  (number_of_grain={number_of_grain})")
    print(f"Orientation mode : {orientation_mode}  (seed={orientation_seed})")
    print(f"Save slices      : {save_slices}, slice_index={slice_index}")
    print(f"Output dir       : {out_dir}")

    save_requested_slices(
        phi_cpu,
        out_dir,
        number_of_grain,
        save_slices,
        slice_index,
        prefix="step_0",
        title_suffix="randommode3d -- step 0",
    )

    save_run_config(out_dir, cfg_runtime, {
        "mode": "randommode_3d",
        "config_path": args.config,
        "number_of_grain": number_of_grain,
        "n_solid": n_solid,
        "seed_height": seed_height,
        "random_seed": random_seed,
        "orientation_mode": orientation_mode,
        "orientation_seed": orientation_seed,
        "orientation_csv": orientation_csv,
        "threads_per_block": list(threadsperblock),
        "save_every": save_every,
        "save_slices": save_slices,
        "slice_index": slice_index,
        "out_dir": out_dir,
        "grain_quaternions": grain_quaternions.tolist(),
    })

    d_phi = cuda.to_device(phi_cpu.astype(np.float32))
    d_phi_new = cuda.to_device(phi_cpu.astype(np.float32).copy())
    d_temp = cuda.to_device(temp_cpu.astype(np.float32))
    d_mf = cuda.to_device(np.zeros((MAX_GRAINS, nx, ny, nz), dtype=np.int32))
    d_nf = cuda.to_device(np.zeros((nx, ny, nz), dtype=np.int32))
    d_wij = cuda.to_device(wij.astype(np.float32))
    d_aij = cuda.to_device(aij.astype(np.float32))
    d_mij = cuda.to_device(mij.astype(np.float32))
    d_n111 = cuda.to_device(grain_n111.astype(np.float32))

    blockspergrid = (
        math.ceil(nx / threadsperblock[0]),
        math.ceil(ny / threadsperblock[1]),
        math.ceil(nz / threadsperblock[2]),
    )

    cooling_rate = np.float32(G * V_pulling * dt)
    T_melt_f = np.float32(T_melt)
    Sf_f = np.float32(Sf)
    g2_floor_f = np.float32((0.1 / dx) ** 2)

    print(f"\nStarting time evolution ({nsteps} steps, saving every {save_every}) ...")

    for nstep in range(1, nsteps + 1):
        kernel_update_temp_3d[blockspergrid, threadsperblock](
            d_temp, cooling_rate, nx, ny, nz
        )

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
            save_requested_slices(
                current_phi,
                out_dir,
                number_of_grain,
                save_slices,
                slice_index,
                prefix=f"step_{nstep}",
                title_suffix=f"randommode3d -- step {nstep}",
            )
            print(f"  saved step_{nstep}")

    print("\nDone.")


if __name__ == "__main__":
    main()
