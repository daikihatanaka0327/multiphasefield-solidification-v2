"""
run_twomode.py
==============
Validation mode: two-grain competitive growth and grain boundary formation.

Purpose
-------
Verify the multi-grain phase field solver with the minimum number of grains
needed to observe grain-boundary physics:
  - Two solid grains with distinct crystallographic orientations grow
    simultaneously into an undercooled melt.
  - The grains are split left / right in x at the start.
  - A grain boundary (GB) forms along x ≈ split_index and evolves.

What this mode checks:
  * Grain boundary formation from a sharp interface.
  * Competitive growth driven by orientation-dependent solidification speed.
  * Interface tilt relative to the pulling direction.
  * GB / solid-liquid interface interaction near the triple junction.
  * Both grains' anisotropy computed independently via grain_n111.

Phases
------
  gid = 0 : liquid   (always; dummy quaternion [0, 0, 0, 1])
  gid = 1 : grain 1  (left half, orientation set via config twomode.grain1)
  gid = 2 : grain 2  (right half, orientation set via config twomode.grain2)
  number_of_grain = 3

Output
------
  result/twomode/step_0.png
  result/twomode/initial_phase_map.png
  result/twomode/initial_temperature.png
  result/twomode/step_{N}.png  (every save_every steps)

Config additions (config.yaml → twomode block)
----------------------------------------------
  twomode:
    seed_height: 32
    split_ratio: 0.5          # grain1/grain2 split (fraction of nx)
    grain1_seed_offset: 0     # extra height offset for grain 1 [grid pts]
    grain2_seed_offset: 0     # extra height offset for grain 2 [grid pts]

    grain1:
      orientation_type: "euler"
      euler_deg: [0.0, 0.0, 0.0]

    grain2:
      orientation_type: "euler"
      euler_deg: [0.0, 20.0, 0.0]
"""

import os
import math
import numpy as np
import yaml
from numba import cuda

from src.gpu_kernels_3d import (
    kernel_update_nfmf_3d,
    kernel_update_phasefield_active_3d,
    kernel_update_temp_3d,
)
from src.orientation_utils import build_quaternion_from_config, compute_rotated_n111

from src.plot_utils_3d import (
    save_phase_map_slice_3d,
    save_interface_position_3d,
    save_run_config,
)

from src.seed_modes_3d import (
    init_twomode_phi_3d,  # 3D version of init_twomode_phi
    init_temperature_field_3d,
    build_interaction_matrices
)


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG_PATH = "config_3d.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

pi = math.pi

# Grid
nx     = cfg["grid"]["nx"]
ny     = cfg["grid"]["ny"]
nz     = cfg["grid"]["nz"]
dx     = cfg["grid"]["dx"]
dy     = cfg["grid"]["dy"]
dz     = cfg["grid"]["dz"]
dt     = cfg["grid"]["dt"]
nsteps = cfg["grid"]["nsteps"]

# Physical
T_melt    = cfg["physical"]["T_melt"]
G         = cfg["physical"]["G"]
V_pulling = cfg["physical"]["V_pulling"]
Sf        = cfg["physical"]["Sf"]

# Interface
delta     = cfg["interface"]["delta_factor"] * dx
gamma_100 = cfg["interface"]["gamma_100"]
gamma_GB  = cfg["interface"]["gamma_GB"]

# Anisotropy
a0      = cfg["anisotropy"]["a0_deg"] * pi / 180.0
delta_a = cfg["anisotropy"]["delta_a"]
mu_a    = cfg["anisotropy"]["mu_a"]
p_round = cfg["anisotropy"]["p_round"]
ksi     = cfg["anisotropy"]["ksi"]
theta_c_rad = cfg["anisotropy"]["omg_deg"] * pi / 180.0

# Mobility
M_SL = cfg["mobility"]["M_SL"]
M_GB = M_SL * cfg["mobility"]["M_GB_ratio"]

# GPU
MAX_GRAINS      = cfg["gpu"]["MAX_GRAINS"]
threadsperblock = tuple(cfg["gpu"]["threads_per_block"])
LIQ             = 0

# Output
save_every  = cfg["output"]["save_every"]
save_slices = list(cfg["output"].get("save_slices", ["xy", "xz"]))
slice_index = cfg["output"].get("slice_index")

# twomode-specific settings
tm_cfg = cfg.get("twomode", {})
seed_height        = int(tm_cfg.get("seed_height", 32))
split_ratio        = float(tm_cfg.get("split_ratio", 0.5))
split_index        = int(nx * split_ratio)
grain1_seed_offset = int(tm_cfg.get("grain1_seed_offset", 0))
grain2_seed_offset = int(tm_cfg.get("grain2_seed_offset", 0))
out_dir            = f"result/twomode_3d/{V_pulling*G}"
os.makedirs(out_dir, exist_ok=True)


# ─── Interface parameter helpers ──────────────────────────────────────────────

def eps_from_gamma(gamma):
    """ε = sqrt(8·δ·γ / π²)."""
    return math.sqrt(8.0 * delta * gamma / (pi * pi))


def w_from_gamma(gamma):
    """w = 4·γ / δ."""
    return 4.0 * gamma / delta


def mij_from_M(M):
    """m_ij = (π² / (8·δ)) · M."""
    return (pi * pi / (8.0 * delta)) * M


eps0_sl  = eps_from_gamma(gamma_100)
w0_sl    = w_from_gamma(gamma_100)
m_sl_phi = mij_from_M(M_SL)
eps_GB   = eps_from_gamma(gamma_GB)
w_GB     = w_from_gamma(gamma_GB)
m_GB_phi = mij_from_M(M_GB)


# ─── Grain orientations ───────────────────────────────────────────────────────
# twomode: number_of_grain = 3  (liquid gid=0, grain1 gid=1, grain2 gid=2)

number_of_grain = 3
N = number_of_grain

# Default orientation configs if not provided in config.yaml
_default_g1 = {"orientation_type": "euler", "euler_deg": [0.0, 0.0, 0.0]}
_default_g2 = {"orientation_type": "euler", "euler_deg": [0.0, 45.0, 0.0]}

grain_quaternions = np.zeros((N, 4), dtype=np.float64)
grain_quaternions[0] = np.array([0.0, 0.0, 0.0, 1.0])   # liquid dummy
grain_quaternions[1] = build_quaternion_from_config(
    tm_cfg.get("grain1", _default_g1))
grain_quaternions[2] = build_quaternion_from_config(
    tm_cfg.get("grain2", _default_g2))

grain_n111 = compute_rotated_n111(grain_quaternions)


# ─── Interaction matrices ─────────────────────────────────────────────────────

wij, aij, mij = build_interaction_matrices(
    N, eps0_sl, w0_sl, m_sl_phi, eps_GB, w_GB, m_GB_phi)


# ─── APT arrays ───────────────────────────────────────────────────────────────

mf_cpu = np.zeros((MAX_GRAINS, nx, ny, nz), dtype=np.int32)
nf_cpu = np.zeros((nx, ny, nz), dtype=np.int32)


# ─── Initial conditions ───────────────────────────────────────────────────────

phi_cpu  = init_twomode_phi_3d(
    nx, ny, nz, dz, delta, seed_height, split_index,
    grain1_seed_offset, grain2_seed_offset)
temp_cpu = init_temperature_field_3d(nx, ny, nz, T_melt, G, dz, seed_height)


# ─── Startup log ─────────────────────────────────────────────────────────────

print("Mode        : twomode")
print(f"Grid        : {nx}x{ny}x{nz}, dx={dx:.1e}, dt={dt:.1e}, nsteps={nsteps}")
print(f"Seed height : {seed_height}  (split_index={split_index}, "
      f"split_ratio={split_ratio:.2f})")
print(f"Phases      : {number_of_grain}")
print("Orientations:")
print(grain_quaternions)
print(f"Output dir  : {out_dir}")


# ─── Save helper ─────────────────────────────────────────────────────────────

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


# ─── Save initial state ───────────────────────────────────────────────────────

save_requested_slices(
    phi_cpu, out_dir, number_of_grain, save_slices, slice_index,
    prefix="step_0",
    title_suffix="twomode3d -- step 0",
)
print("Saved step_0.png, initial_phase_map.png, initial_temperature.png")

save_run_config(out_dir, cfg, {
    "mode": "twomode",
    "number_of_grain": number_of_grain,
    "seed_height": seed_height,
    "split_ratio": split_ratio,
    "split_index": split_index,
    "grain1_seed_offset": grain1_seed_offset,
    "grain2_seed_offset": grain2_seed_offset,
    "grain_quaternions": grain_quaternions.tolist(),
    "out_dir": out_dir,
})


# ─── GPU transfer ─────────────────────────────────────────────────────────────

phi_cpu  = phi_cpu.astype(np.float32)
temp_cpu = temp_cpu.astype(np.float32)

d_phi     = cuda.to_device(phi_cpu)
d_phi_new = cuda.to_device(phi_cpu.copy())
d_temp    = cuda.to_device(temp_cpu)
d_mf      = cuda.to_device(mf_cpu)
d_nf      = cuda.to_device(nf_cpu)
d_wij     = cuda.to_device(wij.astype(np.float32))
d_aij     = cuda.to_device(aij.astype(np.float32))
d_mij     = cuda.to_device(mij.astype(np.float32))
d_n111    = cuda.to_device(grain_n111.astype(np.float32))

blockspergrid = (math.ceil(nx / threadsperblock[0]),
                 math.ceil(ny / threadsperblock[1]),
                 math.ceil(nz / threadsperblock[2]))

cooling_rate = np.float32(G * V_pulling * dt)
T_melt_f     = np.float32(T_melt)
Sf_f         = np.float32(Sf)
g2_floor_f   = np.float32((0.1 / dx) ** 2)


# ─── Main time evolution loop ─────────────────────────────────────────────────

print(f"\nStarting time evolution ({nsteps} steps, saving every {save_every}) ...")

for nstep in range(1, nsteps + 1):

    kernel_update_temp_3d[blockspergrid, threadsperblock](
        d_temp, cooling_rate, nx, ny, nz)

    kernel_update_nfmf_3d[blockspergrid, threadsperblock](
        d_phi, d_mf, d_nf, nx, ny, nz, number_of_grain)

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
            title_suffix=f"twomode3d -- step {nstep}",
        )
        print(f"  saved step_{nstep}")

print("\nDone.")
