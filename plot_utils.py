"""
plot_utils.py
=============
Visualization utilities for multi-phase field simulation results.

These helpers are shared by run_singlemode.py and run_twomode.py and
produce images consistent with the original notebook's style.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; required when no display is available
import matplotlib.pyplot as plt


def save_phase_map(phi: np.ndarray, out_dir: str, filename: str,
                   number_of_grain: int, title: str = "",
                   dpi: int = 150) -> None:
    """Save a phase-ID map image from a phi array.

    The dominant phase at each cell is determined by argmax(phi, axis=0).
    Colourbar ticks are set to integer phase IDs 0..number_of_grain-1.

    Parameters
    ----------
    phi             : np.ndarray, shape (N, nx, ny), float32
    out_dir         : output directory path (must already exist)
    filename        : output file name, e.g. 'step_0.png'
    number_of_grain : total number of phases (sets colorbar ticks)
    title           : plot title string
    dpi             : image resolution
    """
    phase_map = np.argmax(phi, axis=0)  # (nx, ny)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(phase_map.T, cmap="tab20", origin="lower",
                   vmin=0, vmax=max(number_of_grain - 1, 1))
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(number_of_grain), shrink=0.7)
    cbar.set_label("Phase ID", fontsize=30)
    cbar.ax.tick_params(labelsize=20)

    ax.set_title(title, fontsize=35)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=dpi)
    plt.close(fig)


def save_temperature_map(temp: np.ndarray, out_dir: str, filename: str,
                          title: str = "Temperature [K]",
                          dpi: int = 150) -> None:
    """Save a temperature field image.

    Parameters
    ----------
    temp     : np.ndarray, shape (nx, ny)
    out_dir  : output directory path (must already exist)
    filename : output file name
    title    : plot title and colourbar label
    dpi      : image resolution
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(temp.T, cmap="coolwarm", origin="lower")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(title, fontsize=30)
    cbar.ax.tick_params(labelsize=20)

    ax.set_title(title, fontsize=35)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=dpi)
    plt.close(fig)
