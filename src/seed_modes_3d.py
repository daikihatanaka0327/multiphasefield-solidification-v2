"""
seed_modes_3d.py
================
Initial-condition helpers for 3D random-mode solidification.

The seed layer is defined on the x-y plane and extruded along the pulling
direction z using a tanh diffuse solid-liquid profile.
"""

import numpy as np

from .seed_modes import build_interaction_matrices


def generate_random_grain_map_3d(nx: int, ny: int, n_solid: int,
                                 random_seed: int = 42) -> np.ndarray:
    """Generate a Voronoi-based random grain map on the x-y plane."""
    rng = np.random.default_rng(random_seed)
    seeds_x = rng.uniform(0, nx, n_solid).reshape(-1, 1, 1)
    seeds_y = rng.uniform(0, ny, n_solid).reshape(-1, 1, 1)
    lx = np.arange(nx, dtype=np.float32).reshape(1, nx, 1)
    ly = np.arange(ny, dtype=np.float32).reshape(1, 1, ny)
    dist2 = (lx - seeds_x) ** 2 + (ly - seeds_y) ** 2
    return (np.argmin(dist2, axis=0) + 1).astype(np.int32)


def init_temperature_field_3d(nx: int, ny: int, nz: int,
                              T_melt: float, G: float, dz: float,
                              seed_height: int) -> np.ndarray:
    """Initialise T(k) = T_melt + G * (k - seed_height) * dz."""
    k_arr = np.arange(nz, dtype=np.float64)
    temp_1d = (T_melt + G * (k_arr - seed_height) * dz).astype(np.float32)
    temp = np.empty((nx, ny, nz), dtype=np.float32)
    temp[:, :, :] = temp_1d[np.newaxis, np.newaxis, :]
    return temp


def init_phi_from_grain_map_3d(grain_map: np.ndarray, n_solid: int,
                               nx: int, ny: int, nz: int,
                               dz: float, delta: float,
                               seed_height: int) -> np.ndarray:
    """Build a 3D phase field from a 2D grain map on the x-y plane."""
    N = n_solid + 1
    phi = np.zeros((N, nx, ny, nz), dtype=np.float32)

    factor = np.float32(2.2 / delta)
    k_arr = np.arange(nz, dtype=np.float64)
    dist_arr = (k_arr - seed_height) * dz
    phi_s_1d = (0.5 * (1.0 - np.tanh(factor * dist_arr))).astype(np.float32)
    phi_s_3d = phi_s_1d[np.newaxis, np.newaxis, :]

    for gid in range(1, N):
        mask = (grain_map == gid)
        phi[gid] = np.where(mask[:, :, np.newaxis], phi_s_3d, 0.0)

    phi[0] = np.clip(1.0 - phi[1:].sum(axis=0), 0.0, 1.0)
    return phi
