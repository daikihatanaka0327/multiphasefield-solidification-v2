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

def init_twomode_phi_3d(nx: int, ny: int, nz: int, dz: float, delta: float,
                      seed_height: int, split_index: int,
                      grain1_seed_offset: int = 0,
                      grain2_seed_offset: int = 0) -> np.ndarray:
    """Generate the initial phase field for twomode (left grain + right grain + liquid).

    Grain layout:
      l < split_index  → grain 1 (gid = 1)
      l >= split_index → grain 2 (gid = 2)
      m > seed_height  → liquid  (gid = 0)

    A sharp grain boundary forms at x = split_index; the phase field
    solver will relax it to a diffuse profile during time evolution.

    Parameters
    ----------
    nx, ny, nz          : grid dimensions
    dz                  : grid spacing in z (pulling direction) [m]
    delta               : interface thickness parameter [m]
    seed_height         : base solid height [grid points]
    split_index         : x-column index of grain boundary (grain1 | grain2)
    grain1_seed_offset  : vertical offset for grain 1 solid front [grid pts]
    grain2_seed_offset  : vertical offset for grain 2 solid front [grid pts]

    Returns
    -------
    phi : np.ndarray, shape (3, nx, ny), dtype float32
        phi[0] = liquid, phi[1] = grain 1, phi[2] = grain 2
    """
    phi    = np.zeros((3, nx, ny, nz), dtype=np.float32)
    factor = np.float32(2.2 / delta)
    k_arr  = np.arange(nz, dtype=np.float64)

    # Grain 1 (left side: x < split_index), profile along z (pulling direction)
    sh1    = seed_height + grain1_seed_offset
    dist1  = (k_arr - sh1) * dz
    phi_s1 = (0.5 * (1.0 - np.tanh(factor * dist1))).astype(np.float32)
    phi[1, :split_index, :, :] = phi_s1[np.newaxis, np.newaxis, :]

    # Grain 2 (right side: x >= split_index), profile along z (pulling direction)
    sh2    = seed_height + grain2_seed_offset
    dist2  = (k_arr - sh2) * dz
    phi_s2 = (0.5 * (1.0 - np.tanh(factor * dist2))).astype(np.float32)
    phi[2, split_index:, :, :] = phi_s2[np.newaxis, np.newaxis, :]

    # Liquid = 1 − (grain1 + grain2), clipped to [0, 1]
    phi[0] = 1.0 - phi[1] - phi[2]
    np.clip(phi[0], 0.0, 1.0, out=phi[0])

    return phi
