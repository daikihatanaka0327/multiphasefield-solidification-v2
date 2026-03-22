"""
plot_utils.py
=============
Visualization utilities for multi-phase field simulation results.

These helpers are shared by run_singlemode.py and run_twomode.py and
produce images consistent with the original notebook's style.
"""

import os
import datetime
import numpy as np
import yaml
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


def _to_serializable(obj):
    """Recursively convert numpy types to plain Python for YAML serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_run_config(out_dir: str, cfg: dict, runtime_info: dict) -> None:
    """Save the config snapshot and resolved runtime parameters to out_dir.

    Two YAML files are written:

      config_snapshot.yaml
          Exact copy of the config dict loaded from config.yaml at run time.
          Useful for bit-exact reproduction of the simulation.

      run_params.yaml
          Mode-specific resolved parameters: number_of_grain, orientations,
          derived quantities, and a timestamp.  Complements the snapshot by
          showing what the script actually used (e.g. split_index computed
          from split_ratio, or grain quaternions after normalisation).

    Parameters
    ----------
    out_dir      : output directory (must already exist)
    cfg          : raw config dict from yaml.safe_load(config.yaml)
    runtime_info : dict of resolved parameters for this run
                   (numpy arrays are auto-converted to plain lists)
    """
    info = _to_serializable(dict(runtime_info))
    info["saved_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    snapshot_path = os.path.join(out_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    params_path = os.path.join(out_dir, "run_params.yaml")
    with open(params_path, "w", encoding="utf-8") as f:
        yaml.dump(info, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Saved config_snapshot.yaml and run_params.yaml -> {out_dir}")


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
