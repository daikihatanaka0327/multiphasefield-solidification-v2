"""
seed_modes.py
=============
Initial condition generation for all simulation modes.

Functions
---------
singlemode / twomode (validation):
  init_singlemode_phi, init_twomode_phi

randommode / imagemode (production):
  generate_random_grain_map   -- Voronoi tessellation
  load_grain_map_from_image   -- label-image import
  init_phi_from_grain_map     -- phi array from grain map

shared:
  init_temperature_field, build_interaction_matrices

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


def generate_random_grain_map(nx: int, ny: int, n_solid: int,
                               random_seed: int = 42) -> np.ndarray:
    """Generate a Voronoi-based random grain map.

    Places n_solid seed points uniformly at random in the nx x ny grid and
    assigns each cell to the nearest seed point (Euclidean distance).

    Parameters
    ----------
    nx, ny      : grid dimensions
    n_solid     : number of solid grain seeds (grain IDs will be 1..n_solid)
    random_seed : integer seed for reproducibility

    Returns
    -------
    grain_map : np.ndarray, shape (nx, ny), dtype int32
        Values 1..n_solid.  Every cell is assigned to exactly one grain.
    """
    rng = np.random.default_rng(random_seed)
    # Seed point coordinates — shape (n_solid, 1, 1) for broadcasting
    seeds_x = rng.uniform(0, nx, n_solid).reshape(-1, 1, 1)
    seeds_y = rng.uniform(0, ny, n_solid).reshape(-1, 1, 1)

    lx = np.arange(nx, dtype=np.float32).reshape(1, nx, 1)   # (1, nx, 1)
    ly = np.arange(ny, dtype=np.float32).reshape(1, 1, ny)   # (1, 1, ny)

    dist2 = (lx - seeds_x) ** 2 + (ly - seeds_y) ** 2       # (n_solid, nx, ny)
    grain_map = (np.argmin(dist2, axis=0) + 1).astype(np.int32)  # 1..n_solid
    return grain_map


def load_grain_map_from_image(image_path: str, nx: int, ny: int) -> tuple:
    """Load a grain structure from an image file.

    The image is resized to (nx, ny) using nearest-neighbor interpolation.
    Each unique RGB colour in the resized image becomes one grain.
    Grain IDs are assigned 1..n_solid in the order they are first
    encountered by np.unique (ascending encoded-colour value).

    Parameters
    ----------
    image_path : path to the grain-map image (BMP, PNG, TIFF, ...)
    nx, ny     : target grid dimensions

    Returns
    -------
    grain_map  : np.ndarray, shape (nx, ny), dtype int32 -- values 1..n_solid
    n_solid    : int -- number of distinct grains found
    gid_to_rgb : dict {gid: (R, G, B)} -- original colour for each grain ID
    """
    from PIL import Image as _PILImage

    img = _PILImage.open(image_path).convert("RGB")
    # PIL resize argument is (width, height); width=nx, height=ny matches our axes
    img_resized = img.resize((nx, ny), _PILImage.NEAREST)
    arr = np.array(img_resized, dtype=np.uint8)    # (ny, nx, 3) -- PIL is h x w
    arr = arr.transpose(1, 0, 2)                   # (nx, ny, 3) -- match grid axes

    # Encode RGB as a single int32 for fast unique detection
    encoded = (arr[:, :, 0].astype(np.int32) * 65536
               + arr[:, :, 1].astype(np.int32) * 256
               + arr[:, :, 2].astype(np.int32))    # (nx, ny)

    unique_vals, inverse = np.unique(encoded, return_inverse=True)
    grain_map = (inverse + 1).reshape(nx, ny).astype(np.int32)  # 1..n_unique
    n_solid = int(unique_vals.shape[0])

    gid_to_rgb = {}
    for gid0, val in enumerate(unique_vals):
        r = int((val >> 16) & 0xFF)
        g = int((val >> 8) & 0xFF)
        b = int(val & 0xFF)
        gid_to_rgb[gid0 + 1] = (r, g, b)

    return grain_map, n_solid, gid_to_rgb


def init_phi_from_grain_map(grain_map: np.ndarray, n_solid: int,
                             nx: int, ny: int,
                             dy: float, delta: float,
                             seed_height: int) -> np.ndarray:
    """Build the initial phase field from a grain map.

    A tanh diffuse solid-liquid interface is placed at y = seed_height.
    Below this line each cell's solid fraction is assigned to the grain ID
    given by grain_map.  Above the line the cell is pure liquid.

    Parameters
    ----------
    grain_map   : np.ndarray, shape (nx, ny), int32, values 1..n_solid
    n_solid     : number of solid grains
    nx, ny      : grid dimensions
    dy          : grid spacing in y [m]
    delta       : interface thickness parameter [m]  (= delta_factor * dx)
    seed_height : initial solid front height [grid points]

    Returns
    -------
    phi : np.ndarray, shape (n_solid + 1, nx, ny), dtype float32
        phi[0]      = liquid phase field
        phi[1..N-1] = solid grain phase fields
    """
    N   = n_solid + 1
    phi = np.zeros((N, nx, ny), dtype=np.float32)

    factor   = np.float32(2.2 / delta)
    m_arr    = np.arange(ny, dtype=np.float64)
    dist_arr = m_arr * dy - seed_height * dy           # positive above interface
    phi_s_1d = (0.5 * (1.0 - np.tanh(factor * dist_arr))).astype(np.float32)

    # Broadcast 1-D tanh profile to (nx, ny): shape (1, ny) -> (nx, ny)
    phi_s_2d = np.broadcast_to(phi_s_1d[np.newaxis, :], (nx, ny))

    for gid in range(1, n_solid + 1):
        phi[gid] = np.where(grain_map == gid, phi_s_2d, 0.0)

    phi[0] = np.clip(1.0 - phi[1:].sum(axis=0), 0.0, 1.0)
    return phi


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
