"""
run_imagemode.py
================
Multi-grain solidification with grain structure loaded from an image file.

Purpose
-------
Simulate solidification where the initial grain shape and arrangement are
read from an external image.  Each distinct RGB colour in the image maps to
one solid grain; the image is resized to the simulation grid with
nearest-neighbour interpolation.  The solid-liquid interface is always
initialised as a flat tanh profile at y = seed_height, so only the x-z
(in-plane) grain topology comes from the image.

Orientation assignment
----------------------
Three strategies are available (config: imagemode.orientation_mode):

  "random"  -- each grain receives an independent random unit quaternion
               seeded by imagemode.orientation_seed.
  "file"    -- quaternions (x,y,z,w) are loaded row-by-row from a CSV file
               whose path is given by imagemode.orientation_csv.
  "rgb"     -- quaternions are derived deterministically from the grain's
               original RGB colour.  Same colour => same orientation.

Phases
------
  gid = 0        : liquid   (always; dummy quaternion [0, 0, 0, 1])
  gid = 1..n_solid : one solid grain per unique colour in the image

Output
------
  result/imagemode/step_0.png
  result/imagemode/initial_phase_map.png
  result/imagemode/initial_temperature.png
  result/imagemode/step_{N}.png  (every save_every steps)

Config block (config.yaml -> imagemode)
----------------------------------------
  imagemode:
    seed_height: 32
    image_path: "path/to/grain_map.bmp"
    orientation_mode: "random"   # "random", "file", or "rgb"
    orientation_seed: 42
    orientation_csv: ""          # path to CSV when orientation_mode = "file"
"""

import os
import math
import numpy as np
import yaml
from numba import cuda

from gpu_kernels import (
    kernel_update_nfmf,
    kernel_update_phasefield_active,
    kernel_update_temp,
)
from orientation_utils import (
    assign_quaternions_to_grains,
    compute_rotated_n111,
)
from seed_modes import (
    load_grain_map_from_image,
    init_phi_from_grain_map,
    init_temperature_field,
    build_interaction_matrices,
)
from plot_utils import save_phase_map, save_temperature_map


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
omg     = 1.0 / (cfg["anisotropy"]["omg_deg"] * pi / 180.0)

# Mobility
M_SL = cfg["mobility"]["M_SL"]
M_GB = M_SL * cfg["mobility"]["M_GB_ratio"]

# GPU
MAX_GRAINS      = cfg["gpu"]["MAX_GRAINS"]
threadsperblock = tuple(cfg["gpu"]["threads_per_block"])

# Output
save_every = cfg["output"]["save_every"]

# imagemode-specific settings
im_cfg           = cfg.get("imagemode", {})
seed_height      = int(im_cfg.get("seed_height", cfg["seed"].get("height", 32)))
image_path       = im_cfg.get("image_path", "")
orientation_mode = str(im_cfg.get("orientation_mode", "random"))
orientation_seed = int(im_cfg.get("orientation_seed", 42))
orientation_csv  = im_cfg.get("orientation_csv", None) or None
out_dir          = "result/imagemode"
os.makedirs(out_dir, exist_ok=True)

if not image_path:
    raise ValueError(
        "imagemode.image_path is not set in config.yaml. "
        "Provide a path to a grain-map image file.")


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


# --- Load grain map from image -----------------------------------------------

print(f"Loading grain map from: {image_path}")
grain_map, n_solid, gid_to_rgb = load_grain_map_from_image(image_path, nx, ny)

number_of_grain = n_solid + 1  # liquid (gid=0) + n_solid solids
N = number_of_grain

if number_of_grain > MAX_GRAINS:
    raise RuntimeError(
        f"number_of_grain={number_of_grain} (={n_solid} colours + 1 liquid) "
        f"exceeds MAX_GRAINS={MAX_GRAINS}. "
        f"Use an image with fewer grain colours or increase gpu.MAX_GRAINS.")


# --- Grain orientations ------------------------------------------------------

grain_quaternions = assign_quaternions_to_grains(
    n_solid,
    mode=orientation_mode,
    orientation_seed=orientation_seed,
    csv_path=orientation_csv,
    gid_to_rgb=gid_to_rgb,
)

grain_n111 = compute_rotated_n111(grain_quaternions)


# --- Interaction matrices ----------------------------------------------------

wij, aij, mij = build_interaction_matrices(
    N, eps0_sl, w0_sl, m_sl_phi, eps_GB, w_GB, m_GB_phi)


# --- APT arrays --------------------------------------------------------------

mf_cpu = np.zeros((MAX_GRAINS, nx, ny), dtype=np.int32)
nf_cpu = np.zeros((nx, ny), dtype=np.int32)


# --- Initial conditions ------------------------------------------------------

phi_cpu  = init_phi_from_grain_map(
    grain_map, n_solid, nx, ny, dy, delta, seed_height)
temp_cpu = init_temperature_field(nx, ny, T_melt, G, dy, seed_height)


# --- Startup log -------------------------------------------------------------

print("Mode             : imagemode")
print(f"Image            : {image_path}")
print(f"Grid             : {nx}x{ny}, dx={dx:.1e}, dt={dt:.1e}, nsteps={nsteps}")
print(f"Seed height      : {seed_height}")
print(f"n_solid          : {n_solid}  (number_of_grain={number_of_grain})")
print(f"Orientation mode : {orientation_mode}  (seed={orientation_seed})")
print("Grain quaternions (gid=0 liquid, 1..N-1 solid):")
for gid, q in enumerate(grain_quaternions):
    if gid == 0:
        print(f"  gid={gid} [liquid]       q={np.round(q, 4)}")
    else:
        rgb = gid_to_rgb.get(gid, (0, 0, 0))
        print(f"  gid={gid:2d} [grain, RGB={rgb}]  q={np.round(q, 4)}")
print(f"Output dir       : {out_dir}")


# --- Save initial state ------------------------------------------------------

save_phase_map(phi_cpu, out_dir, "step_0.png",
               number_of_grain, title="imagemode -- step 0")
save_phase_map(phi_cpu, out_dir, "initial_phase_map.png",
               number_of_grain, title="imagemode -- Initial Phase Map")
save_temperature_map(temp_cpu, out_dir, "initial_temperature.png",
                     title="Initial Temperature [K]")
print("Saved step_0.png, initial_phase_map.png, initial_temperature.png")


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

cooling_rate = np.float32(G * V_pulling * dt)
T_melt_f     = np.float32(T_melt)
Sf_f         = np.float32(Sf)
g2_floor_f   = np.float32((0.1 / dx) ** 2)


# --- Main time evolution loop ------------------------------------------------

print(f"\nStarting time evolution ({nsteps} steps, saving every {save_every}) ...")

for nstep in range(1, nsteps + 1):

    kernel_update_temp[blockspergrid, threadsperblock](
        d_temp, cooling_rate, nx, ny)

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
        g2_floor_f, np.float32(ksi), np.float32(omg),
    )

    d_phi, d_phi_new = d_phi_new, d_phi

    if nstep % save_every == 0:
        current_phi = d_phi.copy_to_host()
        save_phase_map(current_phi, out_dir, f"step_{nstep}.png",
                       number_of_grain, title=f"imagemode -- step {nstep}")
        print(f"  saved step_{nstep}.png")

print("\nDone.")
