"""
run_diagnostic.py
=================
Single-crystal isothermal growth diagnostic.

Purpose
-------
Place a small disk-shaped solid grain at the domain center in an
undercooled melt (isothermal, no cooling) and run the phase field
evolution. The purpose is to verify whether anisotropic surface
energy and kinetic anisotropy produce the expected faceted
growth morphology (hexagonal shape in 2D, reflecting {111} facets
when viewed along <110>).

What this mode checks
---------------------
  * Whether the anisotropy functions a(theta), b(theta) are effective.
  * Whether the n111 rotation computation is correct.
  * Whether the torque term produces faceted growth.
  * The baseline growth morphology without grain-grain interaction.

Expected result (Zhu et al. 2023, Fig.2)
-----------------------------------------
At dT = 4K, a single silicon crystal grows into a hexagonal shape
with facets aligned to {111} orientations (when viewed along <110>).
If the simulation yields a circular (isotropic) grain, the anisotropy
is not correctly implemented or parameterized.

Phases
------
  gid = 0 : liquid
  gid = 1 : single solid grain

Output
------
  result/diagnostic/step_0.png
  result/diagnostic/initial_phase_map.png
  result/diagnostic/initial_temperature.png
  result/diagnostic/step_{N}.png  (every save_every steps)
  result/diagnostic/config_snapshot.yaml
  result/diagnostic/run_params.yaml
"""

import os
import math
import numpy as np
import yaml
from numba import cuda

from src.gpu_kernels import (
    kernel_update_nfmf,
    kernel_update_phasefield_active,
)
from src.orientation_utils import build_quaternion_from_config, compute_rotated_n111
from src.seed_modes import (
    init_singlemode_disk_2d,
    build_interaction_matrices,
)
from src.plot_utils import save_phase_map, save_temperature_map, save_run_config


# --- Configuration -----------------------------------------------------------

CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

pi = math.pi

# Grid
nx     = cfg["grid"]["nx"]
ny     = cfg["grid"]["ny"]
dx     = cfg["grid"]["dx"]
dy     = cfg["grid"]["dy"]
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

# Output
save_every = cfg["output"]["save_every"]

# diagnostic-specific settings
dg_cfg = cfg.get("diagnostic", {})
initial_radius_cells = int(dg_cfg.get("initial_radius_cells", 10))
center_x_frac        = float(dg_cfg.get("center_x_frac", 0.5))
center_y_frac        = float(dg_cfg.get("center_y_frac", 0.5))
undercooling_K       = float(dg_cfg.get("undercooling_K", 4.0))

out_dir = "result/diagnostic"
os.makedirs(out_dir, exist_ok=True)


# --- Interface parameter helpers ---------------------------------------------

def eps_from_gamma(gamma):
    """eps = sqrt(8 * delta * gamma / pi^2)."""
    return math.sqrt(8.0 * delta * gamma / (pi * pi))


def w_from_gamma(gamma):
    """w = 4 * gamma / delta."""
    return 4.0 * gamma / delta


def mij_from_M(M):
    """m_ij = (pi^2 / (8 * delta)) * M."""
    return (pi * pi / (8.0 * delta)) * M


eps0_sl  = eps_from_gamma(gamma_100)
w0_sl    = w_from_gamma(gamma_100)
m_sl_phi = mij_from_M(M_SL)
eps_GB   = eps_from_gamma(gamma_GB)
w_GB     = w_from_gamma(gamma_GB)
m_GB_phi = mij_from_M(M_GB)


# --- Grain orientations ------------------------------------------------------
# diagnostic: number_of_grain = 2 (liquid gid=0, single solid gid=1)

n_solid = 1
number_of_grain = n_solid + 1
N = number_of_grain

grain_quaternions = np.zeros((N, 4), dtype=np.float64)
grain_quaternions[0] = np.array([0.0, 0.0, 0.0, 1.0])  # liquid dummy
grain_quaternions[1] = build_quaternion_from_config(dg_cfg)

grain_n111 = compute_rotated_n111(grain_quaternions)


# --- Interaction matrices ----------------------------------------------------

wij, aij, mij = build_interaction_matrices(
    N, eps0_sl, w0_sl, m_sl_phi, eps_GB, w_GB, m_GB_phi)


# --- APT arrays --------------------------------------------------------------

mf_cpu = np.zeros((MAX_GRAINS, nx, ny), dtype=np.int32)
nf_cpu = np.zeros((nx, ny), dtype=np.int32)


# --- Initial conditions ------------------------------------------------------

radius   = initial_radius_cells * dx
center_x = center_x_frac * nx * dx
center_y = center_y_frac * ny * dy

phi_cpu = init_singlemode_disk_2d(
    nx, ny, dx, dy, delta, radius,
    center_x=center_x, center_y=center_y, solid_gid=1)

# Uniform undercooled temperature field (isothermal: no cooling)
T_iso = T_melt - undercooling_K
temp_cpu = np.full((nx, ny), T_iso, dtype=np.float32)


# --- Startup log -------------------------------------------------------------

print("Mode             : diagnostic")
print(f"Grid             : {nx}x{ny}, dx={dx:.1e}, dt={dt:.1e}, nsteps={nsteps}")
print(f"Phases           : {number_of_grain}")
print(f"Diagnostic mode  : single-crystal isothermal growth")
print(f"Undercooling     : {undercooling_K} K  (T_iso = {T_melt - undercooling_K} K)")
print(f"Initial disk     : radius = {initial_radius_cells} cells = {initial_radius_cells * dx:.2e} m")
print(f"Disk center      : ({center_x_frac:.2f}, {center_y_frac:.2f}) fraction")
print(f"Grain quaternion : {grain_quaternions[1]}")
print(f"Output dir       : {out_dir}")


# --- Save initial state ------------------------------------------------------

save_phase_map(phi_cpu, out_dir, "step_0.png",
               number_of_grain, title="diagnostic -- step 0")
save_phase_map(phi_cpu, out_dir, "initial_phase_map.png",
               number_of_grain, title="diagnostic -- Initial Phase Map")
save_temperature_map(temp_cpu, out_dir, "initial_temperature.png",
                     title="Initial Temperature [K]")
print("Saved step_0.png, initial_phase_map.png, initial_temperature.png")

save_run_config(out_dir, cfg, {
    "mode": "diagnostic",
    "number_of_grain": number_of_grain,
    "n_solid": n_solid,
    "initial_radius_cells": initial_radius_cells,
    "center_x_frac": center_x_frac,
    "center_y_frac": center_y_frac,
    "undercooling_K": undercooling_K,
    "T_iso": T_iso,
    "grain_quaternions": grain_quaternions.tolist(),
    "out_dir": out_dir,
})


# --- GPU transfer ------------------------------------------------------------

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
                 math.ceil(ny / threadsperblock[1]))

T_melt_f   = np.float32(T_melt)
Sf_f       = np.float32(Sf)
g2_floor_f = np.float32((0.1 / dx) ** 2)


# --- Main time evolution loop ------------------------------------------------

print(f"\nStarting time evolution ({nsteps} steps, saving every {save_every}) ...")
print("grain_n111[0] (liquid):")
print(grain_n111[0])
print("grain_n111[1] (solid):")
print(grain_n111[1])
print("Norms of grain_n111[1]:")
for t in range(8):
    print(f"  t={t}: norm = {np.linalg.norm(grain_n111[1, t])}")
for nstep in range(1, nsteps + 1):

    kernel_update_nfmf[blockspergrid, threadsperblock](
        d_phi, d_mf, d_nf, nx, ny, number_of_grain)

    kernel_update_phasefield_active[blockspergrid, threadsperblock](
        d_phi, d_phi_new, d_temp, d_mf, d_nf,
        d_wij, d_aij, d_mij, d_n111,
        nx, ny, number_of_grain,
        np.float32(dx), np.float32(dt),
        T_melt_f, Sf_f,
        np.float32(eps0_sl), np.float32(w0_sl),
        np.float32(a0), np.float32(delta_a), np.float32(mu_a), np.float32(p_round),
        g2_floor_f, np.float32(ksi), np.float32(theta_c_rad),
    )

    d_phi, d_phi_new = d_phi_new, d_phi

    if nstep % save_every == 0:
        current_phi = d_phi.copy_to_host()
        save_phase_map(current_phi, out_dir, f"step_{nstep}.png",
                       number_of_grain, title=f"diagnostic -- step {nstep}")
        print(f"  saved step_{nstep}.png")

print("\nDone.")
