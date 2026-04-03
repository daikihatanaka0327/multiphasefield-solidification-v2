"""
paper_solver_compat.py
======================

Verification-only compatibility helpers for replaying the Zhu et al. paper
parameters with the existing explicit CUDA kernels, without modifying the
production kernels in ``src/gpu_kernels.py``.

Why this exists
---------------
The paper lists attachment-kinetics coefficients beta_100 and beta_GB.
Feeding those beta values directly into the current explicit solver via

    m = (pi^2 / (8 delta)) * beta

pushes the phase-field update far outside the stable/usable range of this
codebase's time integration.  Production runs in this repository use much
smaller effective mobilities.

To preserve the original source kernels, the paper verification path opts into
this small compatibility layer, which rescales the paper beta values to the
solver range while leaving the original kernels untouched.

This module is intentionally narrow in scope:
  - production scripts do not import it
  - original kernels are unchanged
  - callers must opt in explicitly
"""

from __future__ import annotations

import math


# Calibrated against the flat-front Case-B-like diagnostic so that the
# <001>//T orientation responds near the nominal 40 um/s pulling speed when
# used with the current explicit CUDA solver.
PAPER_BETA_TO_SOLVER_SCALE = 7.0e-7


def raw_beta_to_phasefield_mobility(delta: float, beta: float) -> float:
    """Return the direct Eq.(5)-style mobility conversion."""
    return (math.pi * math.pi / (8.0 * delta)) * beta


def beta_to_solver_mobility(
    delta: float,
    beta: float,
    solver_scale: float = PAPER_BETA_TO_SOLVER_SCALE,
) -> float:
    """Return a verification-compatible mobility for the current solver."""
    return raw_beta_to_phasefield_mobility(delta, beta) * solver_scale


def build_solver_mobilities(
    delta: float,
    beta_100: float,
    beta_gb_ratio: float,
    solver_scale: float = PAPER_BETA_TO_SOLVER_SCALE,
) -> dict[str, float]:
    """Return raw and solver-compatible SL/GB mobilities."""
    raw_sl = raw_beta_to_phasefield_mobility(delta, beta_100)
    raw_gb = raw_beta_to_phasefield_mobility(delta, beta_100 * beta_gb_ratio)
    return {
        "solver_scale": solver_scale,
        "raw_m_sl_phi": raw_sl,
        "raw_m_gb_phi": raw_gb,
        "m_sl_phi": raw_sl * solver_scale,
        "m_gb_phi": raw_gb * solver_scale,
    }
