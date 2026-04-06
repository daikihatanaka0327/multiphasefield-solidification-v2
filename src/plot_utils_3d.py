"""
plot_utils_3d.py
================
Visualization helpers for 3D multi-phase field simulations.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import save_run_config


def _resolve_slice_index(axis: str, shape: tuple[int, int, int], index: int | None) -> int:
    nx, ny, nz = shape
    if axis == "xy":
        upper = nz - 1
        default = nz // 2
    elif axis == "xz":
        upper = ny - 1
        default = ny // 2
    elif axis == "yz":
        upper = nx - 1
        default = nx // 2
    else:
        raise ValueError(f"Unsupported slice axis: {axis}")

    if index is None:
        return default
    return int(np.clip(index, 0, upper))


def save_phase_map_slice_3d(phi: np.ndarray, out_dir: str, filename: str,
                            number_of_grain: int, axis: str = "xy",
                            index: int | None = None, title: str = "",
                            dpi: int = 150) -> None:
    """Save one phase-map slice from a 3D phi array."""
    phase_map_3d = np.argmax(phi, axis=0)
    nx, ny, nz = phase_map_3d.shape
    idx = _resolve_slice_index(axis, (nx, ny, nz), index)

    if axis == "xy":
        slc = phase_map_3d[:, :, idx]
    elif axis == "xz":
        slc = phase_map_3d[:, idx, :]
    elif axis == "yz":
        slc = phase_map_3d[idx, :, :]
    else:
        raise ValueError(f"Unsupported slice axis: {axis}")

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        slc.T,
        cmap="tab20",
        origin="lower",
        vmin=0,
        vmax=max(number_of_grain - 1, 1),
    )
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(number_of_grain), shrink=0.7)
    cbar.set_label("Phase ID", fontsize=24)
    cbar.ax.tick_params(labelsize=14)

    ax.set_title(title or f"{axis} slice @ {idx}", fontsize=24)
    ax.axis("off")
    plt.tight_layout()
    out_dir_3dphase = os.path.join(out_dir+"/3dphase", "3dphase")
    os.makedirs(out_dir_3dphase, exist_ok=True)
    plt.savefig(os.path.join(out_dir_3dphase, filename), dpi=dpi)
    plt.close(fig)


def save_interface_position_3d(phi: np.ndarray, out_dir: str, filename: str,
                               title: str = "", dpi: int = 150) -> None:
    """Save the highest solid z-position for each (x,y) column."""
    solid_frac = 1.0 - phi[0]
    _, _, nz = solid_frac.shape

    solid_mask = solid_frac > 0.5
    has_solid = solid_mask.any(axis=2)
    reverse_idx = np.argmax(solid_mask[:, :, ::-1], axis=2)
    interface_z = np.where(has_solid, nz - 1 - reverse_idx, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(interface_z.T, cmap="viridis", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Interface z position [cells]")
    ax.set_title(title or "Interface z position")
    ax.axis("off")
    plt.tight_layout()
    out_dir_interface = os.path.join(out_dir+"/interface", "interface")
    os.makedirs(out_dir_interface, exist_ok=True)
    plt.savefig(os.path.join(out_dir_interface, filename), dpi=dpi)
    plt.close(fig)
