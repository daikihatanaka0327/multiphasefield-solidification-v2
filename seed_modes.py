"""
seed_modes.py
=============
Initial condition generation for singlemode and twomode simulations.

All functions return NumPy arrays in float32 (phi, temp) or int32 (mf, nf)
that match the dtypes expected by the CUDA kernels and GPU transfer code.
"""

import math
import numpy as np


def init_singlemode_phi(nx: int, ny: int, dy: float, delta: float,
                         seed_height: int, solid_gid: int = 1) -> np.ndarray:
    """Generate the initial phase field for singlemode (single crystal + liquid).

    The solid occupies m < seed_height with a tanh diffuse profile;
    the liquid fills the rest.  The x-direction is uniform — there is
    no grain boundary.

    Parameters
    ----------
    nx, ny      : grid dimensions
    dy          : grid spacing in y [m]
    delta       : interface thickness parameter [m]  (= delta_factor * dx)
    seed_height : initial solid front height [grid points]
    solid_gid   : grain ID for the solid phase (default 1)

    Returns
    -------
    phi : np.ndarray, shape (2, nx, ny), dtype float32
        phi[0]         = liquid phase field
        phi[solid_gid] = solid phase field
    """
    phi    = np.zeros((2, nx, ny), dtype=np.float32)
    factor = np.float32(2.2 / delta)

    m_arr    = np.arange(ny, dtype=np.float64)
    dist_arr = m_arr * dy - seed_height * dy        # positive above seed front
    phi_s_1d = (0.5 * (1.0 - np.tanh(factor * dist_arr))).astype(np.float32)

    # Broadcast identical profile across all x columns
    phi[solid_gid] = phi_s_1d[np.newaxis, :]   # (1, ny) broadcasts to (nx, ny)
    phi[0]         = 1.0 - phi_s_1d[np.newaxis, :]

    return phi


def init_twomode_phi(nx: int, ny: int, dy: float, delta: float,
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
    nx, ny              : grid dimensions
    dy                  : grid spacing in y [m]
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
    phi    = np.zeros((3, nx, ny), dtype=np.float32)
    factor = np.float32(2.2 / delta)
    m_arr  = np.arange(ny, dtype=np.float64)

    # Grain 1 (left side)
    sh1    = seed_height + grain1_seed_offset
    dist1  = m_arr * dy - sh1 * dy
    phi_s1 = (0.5 * (1.0 - np.tanh(factor * dist1))).astype(np.float32)
    phi[1, :split_index, :] = phi_s1[np.newaxis, :]

    # Grain 2 (right side)
    sh2    = seed_height + grain2_seed_offset
    dist2  = m_arr * dy - sh2 * dy
    phi_s2 = (0.5 * (1.0 - np.tanh(factor * dist2))).astype(np.float32)
    phi[2, split_index:, :] = phi_s2[np.newaxis, :]

    # Liquid = 1 − (grain1 + grain2), clipped to [0, 1]
    phi[0] = 1.0 - phi[1] - phi[2]
    np.clip(phi[0], 0.0, 1.0, out=phi[0])

    return phi


def init_temperature_field(nx: int, ny: int, T_melt: float, G: float,
                            dy: float, seed_height: int) -> np.ndarray:
    """Initialise a linear temperature field.

    T(m) = T_melt + G * (m − seed_height) * dy

    This places the melting point exactly at the solid-liquid seed front,
    consistent with the original notebook's convention.

    Parameters
    ----------
    nx, ny      : grid dimensions
    T_melt      : melting temperature [K]
    G           : temperature gradient [K/m]
    dy          : grid spacing in y [m]
    seed_height : reference row that corresponds to T_melt [grid points]

    Returns
    -------
    temp : np.ndarray, shape (nx, ny), dtype float32
    """
    m_arr   = np.arange(ny, dtype=np.float64)
    temp_1d = (T_melt + G * (m_arr - seed_height) * dy).astype(np.float32)
    temp    = np.empty((nx, ny), dtype=np.float32)
    temp[:, :] = temp_1d[np.newaxis, :]
    return temp


def build_interaction_matrices(N: int,
                                eps0_sl: float, w0_sl: float, m_sl_phi: float,
                                eps_GB: float,  w_GB: float,  m_GB_phi: float):
    """Build wij, aij, mij interaction matrices of shape (N, N).

    Filling rules (identical to original notebook):
      - Solid–Liquid pairs  (one index = 0, other > 0): SL baseline values.
      - Grain-Boundary pairs (both indices > 0, i ≠ j): isotropic GB values.
      - Diagonal entries remain 0 (no self-interaction).

    Note: the SL gradient energy (aij) and potential energy (wij) stored here
    serve as baselines.  The main kernel overwrites w_sl locally with the
    anisotropic value at each cell.

    Parameters
    ----------
    N         : total number of phases (liquid gid=0 included)
    eps0_sl   : baseline SL gradient energy coefficient ε₀ [√(J/m)]
    w0_sl     : baseline SL potential energy coefficient w₀ [J/m³]
    m_sl_phi  : SL mobility coefficient (phase-field units)
    eps_GB    : GB gradient energy coefficient [√(J/m)]
    w_GB      : GB potential energy coefficient [J/m³]
    m_GB_phi  : GB mobility coefficient (phase-field units)

    Returns
    -------
    wij, aij, mij : np.ndarray, each shape (N, N), dtype float32
    """
    wij = np.zeros((N, N), dtype=np.float32)
    aij = np.zeros((N, N), dtype=np.float32)
    mij = np.zeros((N, N), dtype=np.float32)

    # Grain-Boundary (solid–solid, isotropic)
    for i in range(1, N):
        for j in range(1, N):
            if i == j:
                continue
            wij[i, j] = w_GB
            aij[i, j] = eps_GB
            mij[i, j] = m_GB_phi

    # Solid–Liquid baseline (anisotropy applied locally in kernel)
    for i in range(1, N):
        wij[0, i] = wij[i, 0] = w0_sl
        aij[0, i] = aij[i, 0] = eps0_sl
        mij[0, i] = mij[i, 0] = m_sl_phi

    return wij, aij, mij
