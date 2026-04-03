"""
run_paper_verification.py
=========================
Paper reproduction script for:

    Chuanqi Zhu et al., "Influences of growth front surfaces on the grain
    boundary development of multi-crystalline silicon during directional
    solidification: 2D/3D multi-phase-field study",
    Materialia 27 (2023) 101702.
    https://doi.org/10.1016/j.mtla.2023.101702

Purpose
-------
Verify that the existing GPU kernels and seed-mode helpers reproduce the
paper's published results when the exact parameters from Table 1 are used.

Cases simulated
---------------
  Case A  Single-grain isothermal growth (Fig. 2 benchmark)
          Grid 200x200, dx=2e-5 m, dt=2e-3 s, 2750 steps, ΔT=4 K
          Grain: <001> // temperature-gradient direction

  Case B  Two-grain competitive growth, high cooling rate (Fig. 3b,c)
          Grid 200x200, G=2500 K/m, R=0.1 K/s → v=40 μm/s, 10 000 steps
          Grain 1: <001> // T-gradient (left)
          Grain 2: <1̄11> // T-gradient (right)
          Expected: GB inclines to the right (~bisector 27.35°) [Fig. 5a]

  Case C  Two-grain competitive growth, low cooling rate (Fig. 3d,e)
          Grid 200x200, G=10000 K/m, R=0.01 K/s → v=1 μm/s, 5000 steps
          Same orientations as Case B
          Expected: GB inclines to the left (energetic mode) [Fig. 5a]
          NOTE: Paper used 900 000 steps; only early-stage trend checked here.

No existing source files are modified.

Output
------
  result/paper_verification/
    caseA_step_{N}.png
    caseB_step_{N}.png
    caseC_step_{N}.png
    verification_report.txt
"""

import os
import math
import time
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from numba import cuda

# ── import existing source modules (unchanged) ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gpu_kernels import (
    kernel_update_nfmf,
    kernel_update_phasefield_active,
    kernel_update_temp,
    KMAX as KERNEL_KMAX,
)
from src.seed_modes import (
    init_singlemode_phi,
    init_twomode_phi,
    init_temperature_field,
    build_interaction_matrices,
)
from src.orientation_utils import compute_rotated_n111
from src.paper_solver_compat import (
    PAPER_BETA_TO_SOLVER_SCALE,
    build_solver_mobilities,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paper Table 1 parameters (exact)
# ─────────────────────────────────────────────────────────────────────────────

PI = math.pi

# — Grid —
DX = 2.0e-5          # m   grid length Δx
DY = DX              # isotropic
DT = 2.0e-3          # s   time step Δt

# — Material —
T_MELT    = 1687.0   # K   melting temperature for silicon
GAMMA_100 = 0.44     # J/m²  interface energy in {100} orientations
GAMMA_111 = 0.32     # J/m²  interface energy on {111} orientations (for check)
GAMMA_GB  = 0.60     # J/m²  grain boundary energy
SF        = 2.12e4   # J/(K·m³)  fusion entropy

# — Attachment kinetics (Table 1) —
BETA_100      = 4.62e-4   # m⁴/(J·s)  kinetic coefficient in <100>
BETA_GB_RATIO = 0.05      # β_GB = 0.05 × β_100

# — Anisotropy functions a(θ) and b(θ) (Table 1) —
A0_DEG   = 54.7    # °  cusp centre angle α₀
DELTA_A  = 0.36    # coefficient δ in a(θ)
MU_A     = 0.6156  # amplitude coefficient μ in a(θ)
KSI      = 0.30    # ζ  cusp depth in b(θ)
OMG_DEG  = 10.0    # °  cusp end angle ω in b(θ)
P_ROUND  = 0.05    # ρ  corner rounding

# — Interface width —
DELTA_FACTOR = 6.0
DELTA        = DELTA_FACTOR * DX   # = 1.2e-4 m  (δ in paper equations)

# — Derived interface parameters (Eqs. 2–5) —
EPS0_SL  = math.sqrt(8.0 * DELTA * GAMMA_100 / (PI ** 2))   # ε₀
W0_SL    = 4.0 * GAMMA_100 / DELTA                           # w₀
EPS_GB   = math.sqrt(8.0 * DELTA * GAMMA_GB  / (PI ** 2))
W_GB     = 4.0 * GAMMA_GB  / DELTA

# — Phase-field mobilities —
# Raw Eq.(5) values from the paper are kept for reporting. The verification run
# itself uses a compatibility-scaled mobility so the original explicit CUDA
# solver can respond without immediately saturating/clipping.
MOBILITY_INFO = build_solver_mobilities(DELTA, BETA_100, BETA_GB_RATIO)
M_SL_PHI_RAW = MOBILITY_INFO["raw_m_sl_phi"]
M_GB_PHI_RAW = MOBILITY_INFO["raw_m_gb_phi"]
M_SL_PHI = MOBILITY_INFO["m_sl_phi"]
M_GB_PHI = MOBILITY_INFO["m_gb_phi"]

# — Radian conversions —
A0      = A0_DEG  * PI / 180.0
THETA_C = OMG_DEG * PI / 180.0

# — Grid size for benchmark (paper Section 3.1) —
NX = 200
NY = 200

# — GPU settings —
MAX_GRAINS      = 10     # ≥ number_of_grain for all cases
THREADS         = (16, 16)
G2_FLOOR        = np.float32((0.1 / DX) ** 2)   # |∇φ|² regularisation floor
GB_SEED_BUFFER_ROWS = int(math.ceil(DELTA / DY)) + 1
GB_MIN_POINTS       = 5
DIAG_NSTEPS_HIGH_RATE = 50
DIAG_VELOCITY_REL_TOL = 0.05
PAPER_PHASE_SUBSTEPS = 50
EXPLICIT_UPDATE_TARGET = 0.10

# — Output —
OUT_DIR    = "result/paper_verification"
SEED_HEIGHT = 50   # initial solid-liquid front [grid points from bottom]

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Parameter verification against Table 1
# ─────────────────────────────────────────────────────────────────────────────

def verify_parameters() -> str:
    """
    Compute and print derived parameters, checking consistency with Table 1.
    Returns a formatted report string.
    """
    lines = []
    lines.append("=" * 66)
    lines.append("PARAMETER VERIFICATION vs Table 1 of Zhu et al. (2023)")
    lines.append("=" * 66)

    # Interface thickness
    lines.append(f"\n[Interface]")
    lines.append(f"  δ  = delta_factor × Δx = {DELTA_FACTOR} × {DX:.2e} = {DELTA:.4e} m")

    # ε₀ from γ₁₀₀ (Eq. 2)
    lines.append(f"\n[Gradient energy coefficient, Eq.2: ε = √(8δγ/π²)]")
    lines.append(f"  ε₀_SL  (from γ₁₀₀={GAMMA_100}) = {EPS0_SL:.6e} √(J/m)")
    lines.append(f"  ε_GB   (from γ_GB={GAMMA_GB})  = {EPS_GB:.6e} √(J/m)")

    # w₀ from γ₁₀₀ (Eq. 3)
    lines.append(f"\n[Penalty coefficient, Eq.3: w = 4γ/δ]")
    lines.append(f"  w₀_SL  = {W0_SL:.4e} J/m³")
    lines.append(f"  w_GB   = {W_GB:.4e} J/m³")

    # Phase-field mobility (Eq. 5 + solver compatibility layer)
    lines.append(f"\n[Phase-field mobility]")
    lines.append(f"  β₁₀₀                    = {BETA_100:.3e} m⁴/(J·s)")
    lines.append(f"  Raw Eq.5 m_SL_phi       = {M_SL_PHI_RAW:.6e} m³/(J·s)")
    lines.append(f"  β_GB = 0.05×β₁₀₀       = {BETA_100*BETA_GB_RATIO:.3e} m⁴/(J·s)")
    lines.append(f"  Raw Eq.5 m_GB_phi       = {M_GB_PHI_RAW:.6e} m³/(J·s)")
    lines.append(f"  Solver compatibility scale = {PAPER_BETA_TO_SOLVER_SCALE:.3e}")
    lines.append(f"  Effective m_SL_phi      = {M_SL_PHI:.6e} m³/(J·s)")
    lines.append(f"  Effective m_GB_phi      = {M_GB_PHI:.6e} m³/(J·s)")

    # Verify γ₁₁₁ via a(θ=0) [Table 1 check]
    # a(θ=0): cost=1, C=√(1+ρ²), S=ρ
    cost0 = 1.0
    C0 = math.sqrt(cost0 ** 2 + P_ROUND ** 2)
    S0 = math.sqrt(max(1.0 - cost0 ** 2, 0.0) + P_ROUND ** 2)
    a_at_0 = MU_A * (1.0 + DELTA_A * (C0 + math.tan(A0) * S0))
    gamma_111_computed = GAMMA_100 * a_at_0 ** 2
    lines.append(f"\n[Surface energy at {{111}} (θ=0 check)]")
    lines.append(f"  a(θ=0)                 = {a_at_0:.6f}")
    lines.append(f"  γ₁₁₁ = γ₁₀₀×a(0)²    = {gamma_111_computed:.4f} J/m²")
    lines.append(f"  Table 1 γ₁₁₁           = {GAMMA_111:.4f} J/m²")
    err_pct = abs(gamma_111_computed - GAMMA_111) / GAMMA_111 * 100
    match = "✓ PASS" if err_pct < 5.0 else "✗ WARNING"
    lines.append(f"  Relative error         = {err_pct:.2f}%  {match}")

    # Time step and grid check
    lines.append(f"\n[Grid / time step]")
    lines.append(f"  Δx = {DX:.2e} m  (Table 1: 2.0×10⁻⁵ m)")
    lines.append(f"  Δt = {DT:.2e} s  (Table 1: 2.0×10⁻³ s)")

    # Kinetic cusp check
    lines.append(f"\n[Kinetic anisotropy b(θ)]")
    lines.append(f"  ζ (ksi)  = {KSI}  (Table 1: 0.30)")
    lines.append(f"  ω (omg)  = {OMG_DEG}°  (Table 1: 10°)")
    lines.append(f"  b(θ=0)   = {KSI}  (cusp minimum)")
    lines.append(f"  b(θ≥ω)   = 1.0   (rough-surface limit)")

    lines.append(f"\n[KMAX in gpu_kernels.py]")
    lines.append(f"  KERNEL_KMAX = {KERNEL_KMAX}")

    lines.append("")
    return "\n".join(lines)


def _estimate_explicit_sl_update(undercooling_K: float, b_value: float = 1.0,
                                 phi_product: float = 0.25) -> dict:
    """Estimate the explicit solid-liquid update size near a diffuse front.

    The main kernel updates a two-phase interface cell approximately as:

        phi_new ~ phi + dt * m_eff * term_force

    because the pairwise prefactor 2 / nf becomes 1 when nf=2.  This helper
    uses the same driving-force factor as the CUDA kernel with a representative
    interface state phi_i = phi_j = 0.5 -> phi_i * phi_j = 0.25.
    """
    driving_force = SF * undercooling_K
    term_force = (8.0 / PI) * math.sqrt(max(phi_product, 0.0)) * driving_force
    effective_mobility = M_SL_PHI * b_value
    update_number = DT * effective_mobility * term_force
    substeps_needed = max(1, int(math.ceil(update_number / EXPLICIT_UPDATE_TARGET)))
    return {
        "undercooling_K": undercooling_K,
        "b_value": b_value,
        "driving_force_J_m3": driving_force,
        "term_force_J_m3": term_force,
        "effective_mobility": effective_mobility,
        "dimensionless_update": update_number,
        "substeps_needed": substeps_needed,
    }


def run_numerical_regime_audit():
    """Audit whether the paper conditions are numerically mild for this solver."""
    cases = [
        ("Case A/B, fast-orientation bound", 4.0, 1.0),
        ("Case A/B, slow-orientation bound", 4.0, KSI),
        ("Case C, fast-orientation bound", 0.1, 1.0),
        ("Case C, slow-orientation bound", 0.1, KSI),
    ]

    rows = []
    for label, undercooling_K, b_value in cases:
        row = _estimate_explicit_sl_update(undercooling_K, b_value=b_value)
        row["label"] = label
        row["practical_with_current_substeps"] = (
            PAPER_PHASE_SUBSTEPS >= row["substeps_needed"]
        )
        rows.append(row)

    peak_update = max(row["dimensionless_update"] for row in rows)
    peak_substeps = max(row["substeps_needed"] for row in rows)

    if peak_update <= EXPLICIT_UPDATE_TARGET:
        verdict = "MILD"
        note = "Estimated explicit update is within the target range."
    else:
        verdict = "SATURATED"
        note = (
            "The paper mobility and undercooling drive dt*m*force well above 1 "
            "for this explicit solver, so front clipping/saturation is likely."
        )

    print("\n" + "=" * 66)
    print("NUMERICAL REGIME AUDIT")
    print("=" * 66)
    print(f"  Target explicit update per substep : <= {EXPLICIT_UPDATE_TARGET:.2f}")
    print(f"  Configured phase substeps          : {PAPER_PHASE_SUBSTEPS}")
    for row in rows:
        practical = "yes" if row["practical_with_current_substeps"] else "no"
        print(f"  {row['label']}:")
        print(f"    undercooling      : {row['undercooling_K']:.3f} K")
        print(f"    b(theta) bound    : {row['b_value']:.3f}")
        print(f"    term_force        : {row['term_force_J_m3']:.3e} J/m^3")
        print(f"    dt*m*force        : {row['dimensionless_update']:.3e}")
        print(f"    substeps needed   : {row['substeps_needed']}")
        print(f"    covered by 50?    : {practical}")
    print(f"  Verdict             : {verdict}")
    print(f"  Note                : {note}")

    return {
        "target_update": EXPLICIT_UPDATE_TARGET,
        "configured_phase_substeps": PAPER_PHASE_SUBSTEPS,
        "peak_dimensionless_update": peak_update,
        "peak_substeps_needed": peak_substeps,
        "verdict": verdict,
        "note": note,
        "rows": rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Grain orientations (crystal axes // temperature-gradient, +y axis)
# ─────────────────────────────────────────────────────────────────────────────

def _build_liquid_dummy_quat() -> np.ndarray:
    """Identity quaternion for liquid phase (gid=0)."""
    return np.array([0.0, 0.0, 0.0, 1.0])


def _normalise_vec(v) -> np.ndarray:
    """Return a float64 unit vector."""
    arr = np.array(v, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        raise ValueError("Cannot normalise a near-zero vector.")
    return arr / norm


PAPER_VIEW_NORMAL_110 = _normalise_vec([1.0, 1.0, 0.0])


def _build_quat_from_growth_and_view(growth_dir_crystal, view_dir_crystal) -> np.ndarray:
    """
    Build a crystal->simulation rotation from:
      - growth direction in crystal coordinates -> simulation +y
      - paper viewing direction in crystal coordinates -> simulation +z

    The 2D paper figures are observed from a <110> direction, so the simulation
    z-axis is treated as the out-of-plane viewing normal.
    """
    ey_src = _normalise_vec(growth_dir_crystal)
    ez_seed = _normalise_vec(view_dir_crystal)

    # Remove any component of the view direction along the growth direction.
    ez_src = ez_seed - np.dot(ez_seed, ey_src) * ey_src
    ez_src = _normalise_vec(ez_src)

    # Complete a right-handed crystal basis that maps onto sim (x, y, z).
    ex_src = _normalise_vec(np.cross(ey_src, ez_src))
    ez_src = _normalise_vec(np.cross(ex_src, ey_src))

    basis_src = np.column_stack([ex_src, ey_src, ez_src])
    rot = Rotation.from_matrix(basis_src.T)
    return rot.as_quat()


def _build_quat_001_along_y() -> np.ndarray:
    """
    Grain with:
      - crystal <001> parallel to the temperature-gradient (+y)
      - paper viewing direction <110> parallel to the simulation +z axis
    """
    return _build_quat_from_growth_and_view(
        growth_dir_crystal=[0.0, 0.0, 1.0],
        view_dir_crystal=PAPER_VIEW_NORMAL_110,
    )


def _build_quat_bar111_along_y() -> np.ndarray:
    """
    Grain with:
      - crystal <1̄11> parallel to the temperature-gradient (+y)
      - paper viewing direction <110> parallel to the simulation +z axis
    """
    return _build_quat_from_growth_and_view(
        growth_dir_crystal=[-1.0, 1.0, 1.0],
        view_dir_crystal=PAPER_VIEW_NORMAL_110,
    )


def _verify_quaternion(q: np.ndarray, expected_v_from: np.ndarray,
                       expected_v_to: np.ndarray, label: str) -> str:
    """Check that quaternion q maps expected_v_from → expected_v_to."""
    rot = Rotation.from_quat(q)
    actual = rot.apply(_normalise_vec(expected_v_from))
    target = _normalise_vec(expected_v_to)
    err = np.linalg.norm(actual - target)
    status = "✓ PASS" if err < 1e-10 else f"✗ err={err:.3e}"
    return f"  {label}: rotates {np.round(expected_v_from,4)} → {np.round(actual,4)}  {status}"


def _calc_a_from_cos_cpu(cost: float) -> float:
    """CPU-side copy of calc_a_from_cos for reporting / diagnostics."""
    c = float(np.clip(cost, -1.0, 1.0))
    c2 = c * c
    C = math.sqrt(c2 + P_ROUND * P_ROUND)
    S = math.sqrt(max(1.0 - c2, 0.0) + P_ROUND * P_ROUND)
    return MU_A * (1.0 + DELTA_A * (C + math.tan(A0) * S))


def _calc_b_from_cos_cpu(best_cost: float) -> float:
    """CPU-side copy of calc_b_from_cos for reporting / diagnostics."""
    c = float(np.clip(best_cost, -1.0, 1.0))
    theta = math.acos(c)
    if theta >= THETA_C:
        return 1.0
    x = theta / THETA_C
    eps = 1.0e-6
    y = 0.5 * math.pi * x
    y = min(y, 0.5 * math.pi - eps)
    y = max(y, eps)
    t = math.tan(y)
    return KSI + (1.0 - KSI) * t * math.tanh(1.0 / t)


def _flat_front_response_metrics(grain_quaternion: np.ndarray) -> dict:
    """
    Report the theoretical anisotropy coefficients seen by a flat horizontal
    front whose interface normal is parallel to +y.
    """
    grain_quats = np.array([_build_liquid_dummy_quat(), grain_quaternion], dtype=np.float64)
    grain_n111 = compute_rotated_n111(grain_quats)
    best_cos = float(np.max(np.abs(grain_n111[1, :, 1])))
    a_flat = _calc_a_from_cos_cpu(best_cos)
    b_flat = _calc_b_from_cos_cpu(best_cos)
    return {
        "best_cos_flat": best_cos,
        "a_flat": a_flat,
        "b_flat": b_flat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Simulation runner (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def run_gpu_loop(phi_init, temp_init, grain_quaternions,
                 nsteps, cooling_rate_per_step,
                 case_label, save_every=500,
                 max_grains=MAX_GRAINS, threads=THREADS,
                 phase_substeps=PAPER_PHASE_SUBSTEPS):
    """
    Run the GPU phase-field time loop and return final phi (CPU array).

    Parameters
    ----------
    phi_init            : np.ndarray (N, nx, ny) float32  — initial phase fields
    temp_init           : np.ndarray (nx, ny) float32     — initial temperature [K]
    grain_quaternions   : np.ndarray (N, 4) float64       — grain orientations
    nsteps              : int                              — number of time steps
    cooling_rate_per_step : float                          — ΔT subtracted each step
    case_label          : str                              — label for console output
    save_every          : int                              — save PNG every N steps
    max_grains          : int                              — MAX_GRAINS buffer size
    threads             : tuple of int                     — CUDA threads per block
    phase_substeps      : int                              — explicit substeps per
                                                             physical time step

    Returns
    -------
    phi_final : np.ndarray (N, nx, ny) float32
    """
    N, nx, ny = phi_init.shape
    number_of_grain = N

    # Compute rotated {111} normals for all grains
    grain_n111 = compute_rotated_n111(grain_quaternions)

    # Interaction matrices
    wij, aij, mij = build_interaction_matrices(
        N, EPS0_SL, W0_SL, M_SL_PHI,
        EPS_GB,  W_GB,  M_GB_PHI)

    # APT (active parameter tracking) arrays
    mf_cpu = np.zeros((max_grains, nx, ny), dtype=np.int32)
    nf_cpu = np.zeros((nx, ny), dtype=np.int32)

    # Transfer to GPU
    phi_f32 = phi_init.astype(np.float32)
    tmp_f32 = temp_init.astype(np.float32)

    d_phi     = cuda.to_device(phi_f32)
    d_phi_new = cuda.to_device(phi_f32.copy())
    d_temp    = cuda.to_device(tmp_f32)
    d_mf      = cuda.to_device(mf_cpu)
    d_nf      = cuda.to_device(nf_cpu)
    d_wij     = cuda.to_device(wij.astype(np.float32))
    d_aij     = cuda.to_device(aij.astype(np.float32))
    d_mij     = cuda.to_device(mij.astype(np.float32))
    d_n111    = cuda.to_device(grain_n111.astype(np.float32))

    bpg = (math.ceil(nx / threads[0]), math.ceil(ny / threads[1]))

    phase_substeps = max(int(phase_substeps), 1)
    cr_f32     = np.float32(cooling_rate_per_step / phase_substeps)
    T_melt_f32 = np.float32(T_MELT)
    Sf_f32     = np.float32(SF)
    a0_f32     = np.float32(A0)
    da_f32     = np.float32(DELTA_A)
    mu_f32     = np.float32(MU_A)
    pr_f32     = np.float32(P_ROUND)
    ksi_f32    = np.float32(KSI)
    tc_f32     = np.float32(THETA_C)
    dx_f32     = np.float32(DX)
    dt_f32     = np.float32(DT / phase_substeps)
    eps_f32    = np.float32(EPS0_SL)
    w_f32      = np.float32(W0_SL)
    g2_f32     = G2_FLOOR

    t0 = time.time()
    print(f"  [{case_label}] starting {nsteps} steps ({nsteps * DT:.1f} s physical, "
          f"{phase_substeps} substeps/step) …")

    for nstep in range(1, nsteps + 1):
        for _ in range(phase_substeps):
            kernel_update_temp[bpg, threads](d_temp, cr_f32, nx, ny)

            kernel_update_nfmf[bpg, threads](
                d_phi, d_mf, d_nf, nx, ny, number_of_grain)

            kernel_update_phasefield_active[bpg, threads](
                d_phi, d_phi_new, d_temp, d_mf, d_nf,
                d_wij, d_aij, d_mij, d_n111,
                nx, ny, number_of_grain,
                dx_f32, dt_f32,
                T_melt_f32, Sf_f32,
                eps_f32, w_f32,
                a0_f32, da_f32, mu_f32, pr_f32,
                g2_f32, ksi_f32, tc_f32,
            )

            d_phi, d_phi_new = d_phi_new, d_phi

        if nstep % save_every == 0 or nstep == nsteps:
            phi_cpu = d_phi.copy_to_host()
            _save_phase_map(phi_cpu, case_label, nstep, N)
            elapsed = time.time() - t0
            print(f"    step {nstep:6d}/{nsteps}  ({elapsed:.1f}s elapsed)")

    phi_final = d_phi.copy_to_host()
    print(f"  [{case_label}] done  ({time.time() - t0:.1f}s)")
    return phi_final


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_phase_map(phi, case_label, step, n_phases):
    """Save a color-coded phase map PNG to the output directory."""
    phase_id = np.argmax(phi, axis=0)          # (nx, ny) int

    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.cm.get_cmap("tab10", n_phases)
    im = ax.imshow(phase_id.T, origin="lower", cmap=cmap,
                   vmin=0, vmax=n_phases - 1, interpolation="nearest")
    ax.set_title(f"{case_label} — step {step}", fontsize=9)
    ax.set_xlabel("x  [grid]")
    ax.set_ylabel("y  [grid]")
    plt.colorbar(im, ax=ax, ticks=range(n_phases),
                 label="phase id (0=liq)")
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"{case_label}_step_{step:06d}.png")
    plt.savefig(fname, dpi=120)
    plt.close(fig)


def measure_solid_fraction(phi, solid_gid=1):
    """
    Return the solid volume fraction of phase solid_gid.
    Equals the spatial mean of phi[solid_gid].
    """
    return float(np.mean(phi[solid_gid]))


def measure_interface_anisotropy(phi, solid_gid=1, threshold=0.5):
    """
    Quantify interface anisotropy (faceting) by measuring the standard
    deviation of the interface height (y-position of phi=threshold) across x.

    A perfectly flat interface has std=0.  A faceted interface has high std.
    A random/curved interface has intermediate std.

    Returns
    -------
    std_height : float   — std of interface height [grid points]
    mean_height : float  — mean height [grid points]
    """
    heights = []
    nx, ny = phi.shape[1], phi.shape[2]
    for ix in range(nx):
        profile = phi[solid_gid, ix, :]
        # Find the last y-index where phi >= threshold (top of solid)
        above = np.where(profile >= threshold)[0]
        if len(above) > 0:
            heights.append(float(above[-1]))
    if len(heights) == 0:
        return 0.0, 0.0
    arr = np.array(heights)
    return float(np.std(arr)), float(np.mean(arr))


def measure_interface_curve(phi, solid_gid=1, threshold=0.5):
    """
    Return (x_idx, y_interface) using a linear phi=threshold crossing.
    """
    field = phi[solid_gid]
    nx, ny = field.shape
    x_coords = np.arange(nx, dtype=np.float64)
    y_curve = np.full(nx, np.nan, dtype=np.float64)

    for ix in range(nx):
        col = field[ix]
        above = np.where(col >= threshold)[0]
        if above.size == 0:
            continue
        k0 = int(above[-1])
        if k0 >= ny - 1:
            y_curve[ix] = float(ny - 1)
            continue
        v0 = float(col[k0])
        v1 = float(col[k0 + 1])
        if abs(v1 - v0) < 1.0e-12:
            y_curve[ix] = float(k0)
        else:
            frac = (threshold - v0) / (v1 - v0)
            y_curve[ix] = float(k0) + frac

    return x_coords, y_curve


def measure_interface_position(phi, solid_gid=1, threshold=0.5):
    """
    Mean and std of the diffuse interface position in grid cells.
    """
    _, y_curve = measure_interface_curve(phi, solid_gid=solid_gid, threshold=threshold)
    finite = y_curve[np.isfinite(y_curve)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def measure_front_velocity(phi_initial, phi_final, solid_gid, dy, total_time,
                           threshold=0.5):
    """
    Front velocity from the shift in mean interface position [m/s].
    """
    y0, _ = measure_interface_position(phi_initial, solid_gid=solid_gid, threshold=threshold)
    y1, _ = measure_interface_position(phi_final, solid_gid=solid_gid, threshold=threshold)
    if not (math.isfinite(y0) and math.isfinite(y1)):
        return float("nan"), y0, y1
    velocity = (y1 - y0) * dy / max(total_time, 1.0e-30)
    return float(velocity), float(y0), float(y1)


def measure_gb_inclination(phi, seed_height, split_x=None, min_solid=0.15,
                           y_start=None, min_points=GB_MIN_POINTS):
    """
    Measure the grain boundary inclination angle for a two-grain simulation.

    The GB is located at each y-row by finding the x-column where
    phi[1] ≈ phi[2] (transition between grains 1 and 2) in the solid region
    (where the total solid fraction phi[1]+phi[2] > min_solid).

    A linear fit to (x_gb, y) gives the GB slope; the inclination angle is:
        angle = arctan(Δx / Δy) × 180/π     [degrees from vertical]

    Positive angle = GB tilts to the right (grain-1/left expands).
    Negative angle = GB tilts to the left  (grain-2/right expands).

    Parameters
    ----------
    phi        : np.ndarray (3, nx, ny)
    seed_height: int  — rows ≥ seed_height are in the grown region
    split_x    : int or None  — initial GB x-position (default nx//2)
    min_solid  : float — minimum solid fraction for a row to be analysed
    y_start    : int or None — first y-row to analyse; use this to exclude the
                               initial diffuse seed band
    min_points : int — minimum number of sampled rows required for a fit

    Returns
    -------
    angle_deg  : float  — GB inclination [degrees], or NaN if insufficient data
    gb_xs      : list   — GB x-positions per y-row (for plotting)
    gb_ys      : list   — corresponding y-row indices
    """
    _, nx, ny = phi.shape
    if split_x is None:
        split_x = nx // 2
    if y_start is None:
        y_start = seed_height

    gb_xs, gb_ys = [], []

    for iy in range(max(seed_height, y_start), ny):
        solid_frac = float(phi[1, :, iy].sum() + phi[2, :, iy].sum()) / nx
        if solid_frac < min_solid:
            continue

        diff = phi[1, :, iy] - phi[2, :, iy]   # shape (nx,)
        # Find zero-crossing closest to split_x
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) == 0:
            continue
        # Pick the sign change closest to the initial split_x
        closest = sign_changes[np.argmin(np.abs(sign_changes - split_x))]
        gb_xs.append(float(closest))
        gb_ys.append(float(iy))

    if len(gb_ys) < min_points:
        return float("nan"), gb_xs, gb_ys

    gb_xs_arr = np.array(gb_xs)
    gb_ys_arr = np.array(gb_ys)

    # Linear fit: x_gb = slope * y + intercept
    coeffs = np.polyfit(gb_ys_arr, gb_xs_arr, 1)
    slope  = coeffs[0]   # Δx/Δy

    angle_deg = math.degrees(math.atan(slope))
    return angle_deg, gb_xs, gb_ys


def run_orientation_velocity_diagnostics():
    """
    Single-grain directional-growth diagnostic.

    Uses the high-rate thermal condition from Case B so that a measurable front
    displacement occurs within a short run. The goal is not paper reproduction,
    but checking whether the kernel converts orientation-dependent a(theta) and
    b(theta) into different front velocities.
    """
    print("\n" + "=" * 66)
    print("ORIENTATION RESPONSE DIAGNOSTIC: single-grain front velocity")
    print("=" * 66)

    G_diag = 2500.0
    R_diag = 0.1
    V_diag = R_diag / G_diag
    nsteps = DIAG_NSTEPS_HIGH_RATE
    dt_under = 4.0
    total_time = nsteps * DT
    cooling_rate_per_step = G_diag * V_diag * DT

    print(f"  Condition : Case-B-like directional growth")
    print(f"  G={G_diag:.0f} K/m, R={R_diag:.2f} K/s, v={V_diag * 1e6:.0f} μm/s")
    print(f"  Steps     : {nsteps} ({total_time:.1f} s, {V_diag * total_time / DX:.1f} cells)")

    orientations = [
        ("diag_vel_001", "<001>//T", _build_quat_001_along_y()),
        ("diag_vel_bar111", "<1̄11>//T", _build_quat_bar111_along_y()),
    ]
    results = []

    for case_label, grain_label, grain_quat in orientations:
        print(f"\n  [{grain_label}]")
        theory = _flat_front_response_metrics(grain_quat)
        print(f"    Theoretical flat-front response: "
              f"best_cos={theory['best_cos_flat']:.3f}, "
              f"a={theory['a_flat']:.4f}, b={theory['b_flat']:.4f}")

        phi = init_singlemode_phi(NX, NY, DY, DELTA, SEED_HEIGHT, solid_gid=1)
        temp = init_temperature_field(NX, NY, T_MELT - dt_under, G_diag, DY, SEED_HEIGHT)
        grain_quats = np.array([_build_liquid_dummy_quat(), grain_quat], dtype=np.float64)

        phi_final = run_gpu_loop(
            phi, temp, grain_quats,
            nsteps=nsteps,
            cooling_rate_per_step=float(cooling_rate_per_step),
            case_label=case_label,
            save_every=nsteps,
        )

        v_front, y0, y1 = measure_front_velocity(
            phi, phi_final, solid_gid=1, dy=DY, total_time=total_time)
        rough0 = measure_interface_position(phi, solid_gid=1)[1]
        rough1 = measure_interface_position(phi_final, solid_gid=1)[1]
        print(f"    Mean front position : {y0:.2f} -> {y1:.2f} cells")
        print(f"    Front velocity      : {v_front:.6e} m/s")
        if math.isfinite(v_front):
            print(f"    Velocity / pulling  : {v_front / V_diag:.1f}×")
        print(f"    Front roughness     : {rough0:.2f} -> {rough1:.2f} cells")

        results.append({
            "label": grain_label,
            "case_label": case_label,
            "front_velocity_m_s": v_front,
            "interface_y0_cells": y0,
            "interface_y1_cells": y1,
            "interface_roughness_final_cells": rough1,
            **theory,
        })

    v_001 = results[0]["front_velocity_m_s"]
    v_111 = results[1]["front_velocity_m_s"]
    diff = v_001 - v_111 if math.isfinite(v_001) and math.isfinite(v_111) else float("nan")
    denom = max(abs(v_001), abs(v_111), 1.0e-30) if math.isfinite(diff) else 1.0
    rel_diff = diff / denom if math.isfinite(diff) else float("nan")
    ratio_001 = v_001 / V_diag if math.isfinite(v_001) else float("nan")
    ratio_111 = v_111 / V_diag if math.isfinite(v_111) else float("nan")

    if not math.isfinite(diff):
        verdict = "INCONCLUSIVE"
        note = "Could not compute a finite front velocity for one or both orientations."
    elif abs(rel_diff) <= DIAG_VELOCITY_REL_TOL:
        verdict = "NO CLEAR RESPONSE"
        note = (f"Velocity contrast is only {rel_diff * 100:.1f}% "
                f"(tolerance {DIAG_VELOCITY_REL_TOL * 100:.0f}%).")
        if math.isfinite(ratio_001) and math.isfinite(ratio_111) and max(ratio_001, ratio_111) > 10.0:
            note += " Both fronts also move much faster than the nominal pulling speed."
    elif diff > 0.0:
        verdict = "EXPECTED ORDER"
        note = "<001>//T grows faster than <1̄11>//T under the flat-front diagnostic."
    else:
        verdict = "REVERSED ORDER"
        note = "<1̄11>//T grows faster than <001>//T, contrary to the flat-front expectation."

    print(f"\n  Diagnostic verdict : {verdict}")
    print(f"  Note               : {note}")

    return {
        "nsteps": nsteps,
        "condition": "single-grain directional diagnostic (Case-B-like thermal loading)",
        "velocity_001_m_s": v_001,
        "velocity_bar111_m_s": v_111,
        "velocity_diff_m_s": diff,
        "velocity_diff_rel": rel_diff,
        "velocity_ratio_001_to_pulling": ratio_001,
        "velocity_ratio_bar111_to_pulling": ratio_111,
        "verdict": verdict,
        "note": note,
        "rows": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Case A — Single-grain isothermal growth (Fig. 2 benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def run_case_A():
    """
    Single-crystal isothermal growth.
    Paper: 200×200 grid, 2750 steps, ΔT = 4 K, viewed from <110>.
    Expected: octahedral grain shape with {111} facets.
    """
    print("\n" + "=" * 66)
    print("CASE A: Single-grain isothermal growth  (Fig. 2 benchmark)")
    print("=" * 66)
    print(f"  Grid    : {NX}×{NY},  Δx={DX:.2e} m,  Δt={DT:.2e} s")
    print(f"  Steps   : 2750  (= {2750 * DT:.1f} s physical time)")
    print(f"  Cond.   : isothermal ΔT = 4 K")
    print(f"  Grain   : <001> // temperature-gradient (y-axis), viewed from <110>")

    N           = 2   # liquid (0) + solid (1)
    NSTEPS_A    = 2750
    UNDERCOOL_A = 4.0  # K

    # — Grain orientations (paper: viewed from <110>, grain <001> // T-grad) —
    q_liq  = _build_liquid_dummy_quat()
    q_g1   = _build_quat_001_along_y()
    grain_quats = np.array([q_liq, q_g1])

    print(_verify_quaternion(
        q_g1, np.array([0., 0., 1.]), np.array([0., 1., 0.]), "<001>→+y"))
    print(_verify_quaternion(
        q_g1, PAPER_VIEW_NORMAL_110, np.array([0., 0., 1.]), "<110>→+z"))

    # — Initial conditions —
    phi  = init_singlemode_phi(NX, NY, DY, DELTA, SEED_HEIGHT, solid_gid=1)
    # Isothermal: uniform T = T_melt - ΔT everywhere (no gradient, no cooling)
    temp = np.full((NX, NY), T_MELT - UNDERCOOL_A, dtype=np.float32)

    # — Initial diagnostics —
    sf0 = measure_solid_fraction(phi, solid_gid=1)
    print(f"\n  Initial solid fraction  : {sf0:.4f}")
    print(f"  Expected (seed={SEED_HEIGHT}/NY=200): {SEED_HEIGHT/NY:.4f}")

    # — Run GPU loop (cooling_rate=0 → isothermal) —
    phi_final = run_gpu_loop(
        phi, temp, grain_quats,
        nsteps=NSTEPS_A,
        cooling_rate_per_step=0.0,   # isothermal
        case_label="caseA",
        save_every=500,
    )

    # — Final analysis —
    sf_final = measure_solid_fraction(phi_final, solid_gid=1)
    std_h, mean_h = measure_interface_anisotropy(phi_final, solid_gid=1)

    print(f"\n  RESULTS after {NSTEPS_A} steps ({NSTEPS_A * DT:.1f} s):")
    print(f"  Final solid fraction     : {sf_final:.4f}  (was {sf0:.4f})")
    print(f"  Grain grew by            : {(sf_final - sf0) * NY:.1f} rows (avg)")
    print(f"  Interface height std     : {std_h:.2f} grid cells")
    print(f"  Interface height mean    : {mean_h:.2f} grid cells")
    print(f"  Anisotropy note: std>0 indicates non-flat (faceted) interface.")

    # Qualitative screen only: this detects front corrugation but does not by
    # itself prove a full Fig.2 match.
    faceted = std_h > 1.0
    result  = "✓ DETECTED (qualitative faceting signal present)" if faceted \
              else "? NOT DETECTED (front stayed nearly flat)"
    print(f"  Faceting check           : {result}")

    return {
        "case": "A",
        "nsteps": NSTEPS_A,
        "initial_solid_frac": sf0,
        "final_solid_frac": sf_final,
        "interface_height_std": std_h,
        "interface_height_mean": mean_h,
        "faceting_pass": faceted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Case B — Two-grain, high cooling rate (Fig. 3b, v=40 μm/s)
# ─────────────────────────────────────────────────────────────────────────────

def run_case_B():
    """
    Two-grain competitive growth at high cooling rate (kinetic-dominated mode).
    Paper: G=2500 K/m, R=0.1 K/s → v=40 μm/s.
    Expected: GB inclines to the right (toward bisector ≈27.35°). [Fig. 5a]
    """
    print("\n" + "=" * 66)
    print("CASE B: Two-grain, high rate  G=2500 K/m, R=0.1 K/s  (Fig. 3b)")
    print("=" * 66)

    G_B       = 2500.0    # K/m
    R_B       = 0.1       # K/s
    V_B       = R_B / G_B  # = 4e-5 m/s = 40 μm/s
    NSTEPS_B  = 10_000
    DT_UNDER  = 4.0        # K  initial undercooling at seed front (paper p.5)

    print(f"  Grid      : {NX}×{NY},  Δx={DX:.2e} m,  Δt={DT:.2e} s")
    print(f"  G         : {G_B:.0f} K/m")
    print(f"  R         : {R_B} K/s")
    print(f"  v (steady): {V_B * 1e6:.0f} μm/s")
    print(f"  Steps     : {NSTEPS_B}  ({NSTEPS_B * DT:.0f} s physical)")
    print(f"  Grain 1   : <001>  // T-gradient (left half)")
    print(f"  Grain 2   : <1̄11> // T-gradient (right half)")
    print(f"  View      : <110> normal to paper (simulation +z)")
    print(f"  Paper bisector angle: 27.35°")

    N = 3   # liquid (0) + grain1 (1) + grain2 (2)

    # — Grain orientations —
    q_liq = _build_liquid_dummy_quat()
    q_g1  = _build_quat_001_along_y()
    q_g2  = _build_quat_bar111_along_y()
    grain_quats = np.array([q_liq, q_g1, q_g2])

    v_from_001 = np.array([0., 0., 1.])
    v_from_111 = np.array([0., 0., 1.])
    print(_verify_quaternion(q_g1, v_from_001, np.array([0., 1., 0.]),
                             "<001>→+y  (grain1)"))
    print(_verify_quaternion(q_g1, PAPER_VIEW_NORMAL_110, np.array([0., 0., 1.]),
                             "<110>→+z  (grain1)"))
    print(_verify_quaternion(q_g2, np.array([-1., 1., 1.]) / math.sqrt(3.),
                             np.array([0., 1., 0.]),
                             "<1̄11>→+y  (grain2)"))
    print(_verify_quaternion(q_g2, PAPER_VIEW_NORMAL_110, np.array([0., 0., 1.]),
                             "<110>→+z  (grain2)"))

    # — Initial conditions —
    split_index = NX // 2
    phi = init_twomode_phi(NX, NY, DY, DELTA, SEED_HEIGHT, split_index)
    # Paper p.5: "initial temperature at growth front set to 4 K below melting"
    temp = init_temperature_field(NX, NY, T_MELT - DT_UNDER, G_B, DY, SEED_HEIGHT)

    cooling_rate_per_step = G_B * V_B * DT   # = G × V × Δt
    expected_growth_cells = V_B * NSTEPS_B * DT / DX
    y_start = SEED_HEIGHT + GB_SEED_BUFFER_ROWS

    # — Run GPU loop —
    phi_final = run_gpu_loop(
        phi, temp, grain_quats,
        nsteps=NSTEPS_B,
        cooling_rate_per_step=float(cooling_rate_per_step),
        case_label="caseB",
        save_every=1000,
    )

    # — Measure GB inclination —
    angle_deg, gb_xs, gb_ys = measure_gb_inclination(
        phi_final, SEED_HEIGHT, split_x=split_index,
        y_start=y_start, min_points=GB_MIN_POINTS)
    gb_reliable = math.isfinite(angle_deg)

    print(f"\n  RESULTS after {NSTEPS_B} steps:")
    print(f"  Physical growth      : {expected_growth_cells:.1f} grid cells")
    print(f"  GB fit start row     : y >= {y_start}  "
          f"(seed buffer {GB_SEED_BUFFER_ROWS} rows)")
    if gb_reliable:
        print(f"  GB inclination angle : {angle_deg:+.2f}°")
        print(f"  (positive = GB tilts right = grain-1/<001> expands)")
    else:
        print(f"  GB inclination angle : n/a  (insufficient post-seed GB points)")
    print(f"  Paper Fig.5a (40 μm/s): ~+27.35° (bisector)")
    print(f"  Note: 10 000 steps < paper's 22 500; early-stage inclination")

    # Save a GB-position overlay
    _save_gb_overlay(phi_final, gb_xs, gb_ys, split_index,
                     "caseB", NSTEPS_B, angle_deg)

    positive_incl = gb_reliable and angle_deg > 0.0
    if not gb_reliable:
        result = "? INCONCLUSIVE (insufficient post-seed GB points)"
    elif positive_incl:
        result = "✓ PASS (GB inclines right as expected)"
    else:
        result = "✗ FAIL (GB should incline right at high rate)"
    print(f"  Inclination direction: {result}")

    return {
        "case": "B",
        "G": G_B, "R": R_B, "v_um_s": V_B * 1e6,
        "nsteps": NSTEPS_B,
        "expected_growth_cells": expected_growth_cells,
        "gb_angle_deg": angle_deg,
        "gb_n_points": len(gb_ys),
        "gb_reliable": gb_reliable,
        "direction_pass": positive_incl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Case C — Two-grain, low cooling rate (Fig. 3d, v=1 μm/s)
# ─────────────────────────────────────────────────────────────────────────────

def run_case_C():
    """
    Two-grain competitive growth at low cooling rate (energetic-dominated mode).
    Paper: G=10000 K/m, R=0.01 K/s → v=1 μm/s.
    Expected: GB inclines to the LEFT (negative angle). [Fig. 5a, Fig. 3d]

    NOTE: Paper ran 900 000 steps (~30 min/GPU for 200×200).
    Here we run 5 000 steps only as a reduced-time sanity check.
    At 1 μm/s growth, 5 000 steps = 10 s physical = 10 μm growth = 0.5 cells,
    which is not enough to fit a reliable GB angle after excluding the initial
    diffuse seed band. This case is therefore reported as inconclusive.
    """
    print("\n" + "=" * 66)
    print("CASE C: Two-grain, low rate  G=10000 K/m, R=0.01 K/s  (Fig. 3d)")
    print("=" * 66)

    G_C       = 10_000.0   # K/m
    R_C       = 0.01       # K/s
    V_C       = R_C / G_C  # = 1e-6 m/s = 1 μm/s
    NSTEPS_C  = 5_000
    DT_UNDER  = 0.1        # K  initial undercooling at seed (paper p.5)

    print(f"  Grid      : {NX}×{NY},  Δx={DX:.2e} m,  Δt={DT:.2e} s")
    print(f"  G         : {G_C:.0f} K/m")
    print(f"  R         : {R_C} K/s")
    print(f"  v (steady): {V_C * 1e6:.0f} μm/s")
    print(f"  Steps     : {NSTEPS_C}  ({NSTEPS_C * DT:.1f} s physical)")
    print(f"  Physical growth: {V_C * NSTEPS_C * DT * 1e6:.1f} μm = "
          f"{V_C * NSTEPS_C * DT / DX:.2f} grid cells  (very slow!)")
    print(f"  View      : <110> normal to paper (simulation +z)")
    print(f"  ⚠  Paper ran 900 000 steps; this reduced run is not enough for a "
          f"reliable GB-angle fit")

    N = 3
    q_liq = _build_liquid_dummy_quat()
    q_g1  = _build_quat_001_along_y()
    q_g2  = _build_quat_bar111_along_y()
    grain_quats = np.array([q_liq, q_g1, q_g2])

    # — Initial conditions —
    split_index = NX // 2
    phi  = init_twomode_phi(NX, NY, DY, DELTA, SEED_HEIGHT, split_index)
    # Paper p.5: "initial temperature at growth front set to 0.1 K below melting"
    temp = init_temperature_field(NX, NY, T_MELT - DT_UNDER, G_C, DY, SEED_HEIGHT)

    cooling_rate_per_step = G_C * V_C * DT
    expected_growth_cells = V_C * NSTEPS_C * DT / DX
    y_start = SEED_HEIGHT + GB_SEED_BUFFER_ROWS

    # — Run GPU loop —
    phi_final = run_gpu_loop(
        phi, temp, grain_quats,
        nsteps=NSTEPS_C,
        cooling_rate_per_step=float(cooling_rate_per_step),
        case_label="caseC",
        save_every=1000,
    )

    # — Measure GB inclination only if growth exceeds the initial diffuse seed band —
    if expected_growth_cells <= GB_SEED_BUFFER_ROWS:
        angle_deg = float("nan")
        gb_xs, gb_ys = [], []
        gb_reliable = False
        note = (
            f"Expected growth is only {expected_growth_cells:.2f} grid cells, "
            f"below the {GB_SEED_BUFFER_ROWS}-row seed buffer; GB angle not "
            f"measured."
        )
    else:
        angle_deg, gb_xs, gb_ys = measure_gb_inclination(
            phi_final, SEED_HEIGHT, split_x=split_index, min_solid=0.05,
            y_start=y_start, min_points=GB_MIN_POINTS)
        gb_reliable = math.isfinite(angle_deg)
        if gb_reliable:
            note = "Reduced-time GB-angle fit from post-seed rows only."
        else:
            note = "Insufficient post-seed GB points for a reliable angle fit."

    print(f"\n  RESULTS after {NSTEPS_C} steps:")
    print(f"  Physical growth      : {expected_growth_cells:.2f} grid cells")
    print(f"  GB fit start row     : y >= {y_start}  "
          f"(seed buffer {GB_SEED_BUFFER_ROWS} rows)")
    if gb_reliable:
        print(f"  GB inclination angle : {angle_deg:+.2f}°  (n_points={len(gb_ys)})")
    else:
        print(f"  GB inclination angle : n/a  (n_points={len(gb_ys)})")
    print(f"  Paper Fig.5a (1 μm/s): small negative angle (energetic mode)")
    print(f"  Note: {note}")

    # For Case C we expect a much smaller positive angle or negative angle
    # compared to Case B — the key physics is the REDUCTION in inclination
    # at low rate, not an exact negative value after only 5000 steps.
    _save_gb_overlay(phi_final, gb_xs, gb_ys, split_index,
                     "caseC", NSTEPS_C, angle_deg)

    return {
        "case": "C",
        "G": G_C, "R": R_C, "v_um_s": V_C * 1e6,
        "nsteps": NSTEPS_C,
        "expected_growth_cells": expected_growth_cells,
        "gb_angle_deg": angle_deg,
        "gb_n_points": len(gb_ys),
        "gb_reliable": gb_reliable,
        "note": note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: GB overlay visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_gb_overlay(phi, gb_xs, gb_ys, split_x, case_label, step, angle_deg):
    """Save a phase map with the measured GB positions overlaid."""
    angle_label = f"{angle_deg:+.1f}°" if math.isfinite(angle_deg) else "n/a"
    phase_id = np.argmax(phi, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: grain 1 fraction
    ax = axes[0]
    im = ax.imshow(phi[1].T, origin="lower", cmap="Reds", vmin=0, vmax=1,
                   interpolation="nearest")
    if gb_xs:
        ax.plot(gb_xs, gb_ys, "b.", ms=1.5, label="GB (measured)")
    ax.axvline(split_x, color="gray", lw=0.5, ls="--")
    ax.set_title(f"φ grain1 (<001>)  {case_label}", fontsize=9)
    ax.set_xlabel("x  [grid]")
    ax.set_ylabel("y  [grid]")
    plt.colorbar(im, ax=ax)
    ax.legend(fontsize=7)

    # Right: dominant phase map
    ax = axes[1]
    cmap = plt.cm.get_cmap("tab10", 3)
    im2 = ax.imshow(phase_id.T, origin="lower", cmap=cmap,
                    vmin=0, vmax=2, interpolation="nearest")
    if gb_xs:
        ax.plot(gb_xs, gb_ys, "k.", ms=1.5, label="GB (measured)")
    ax.set_title(f"Phase map  step {step}  angle={angle_label}", fontsize=9)
    ax.set_xlabel("x  [grid]")
    ax.set_ylabel("y  [grid]")
    plt.colorbar(im2, ax=ax, ticks=[0, 1, 2],
                 label="0=liq 1=<001> 2=<1̄11>")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"{case_label}_gb_overlay_step{step}.png")
    plt.savefig(fname, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Summary report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(param_check_str, audit_results, diag_results,
                 results_A, results_B, results_C):
    """Write a plain-text verification report to file and stdout."""
    def fmt_angle(angle: float) -> str:
        return f"{angle:+.2f}°" if math.isfinite(angle) else "n/a"

    lines = []
    lines.append("=" * 66)
    lines.append("VERIFICATION REPORT — Zhu et al. Materialia 27 (2023) 101702")
    lines.append("=" * 66)
    lines.append(f"Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Grid      : {NX}×{NY},  Δx={DX:.2e} m,  Δt={DT:.2e} s")
    lines.append(f"δ         : {DELTA:.4e} m  (delta_factor={DELTA_FACTOR})")
    lines.append(f"β₁₀₀      : {BETA_100:.3e} m⁴/(J·s)  (Table 1)")
    lines.append(f"Raw Eq.5 m_SL_phi : {M_SL_PHI_RAW:.6e} m³/(J·s)")
    lines.append(f"Solver scale      : {PAPER_BETA_TO_SOLVER_SCALE:.3e}")
    lines.append(f"Effective m_SL_phi: {M_SL_PHI:.6e} m³/(J·s)")
    lines.append("")

    # Numerical-regime audit
    lines.append("─" * 40)
    lines.append("Numerical Regime Audit")
    lines.append(f"  Target dt*m*force per substep : <= {audit_results['target_update']:.2f}")
    lines.append(f"  Configured phase substeps     : {audit_results['configured_phase_substeps']}")
    for row in audit_results["rows"]:
        covered = "yes" if row["practical_with_current_substeps"] else "no"
        lines.append(f"  {row['label']}")
        lines.append(f"  dt*m*force    : {row['dimensionless_update']:.3e}")
        lines.append(f"  Substeps need : {row['substeps_needed']}")
        lines.append(f"  Covered now?  : {covered}")
    lines.append(f"  Verdict   : {audit_results['verdict']}")
    lines.append(f"  Note      : {audit_results['note']}")

    lines.append("")

    # Orientation diagnostic
    lines.append("─" * 40)
    lines.append("Orientation Response Diagnostic")
    lines.append(f"  Condition : {diag_results['condition']}")
    lines.append(f"  Steps     : {diag_results['nsteps']}")
    lines.append(f"  v(<001>//T)   : {diag_results['velocity_001_m_s']:.6e} m/s")
    lines.append(f"  v(<1̄11>//T)  : {diag_results['velocity_bar111_m_s']:.6e} m/s")
    if math.isfinite(diag_results["velocity_ratio_001_to_pulling"]):
        lines.append(f"  v001 / pulling  : {diag_results['velocity_ratio_001_to_pulling']:.1f}×")
    if math.isfinite(diag_results["velocity_ratio_bar111_to_pulling"]):
        lines.append(f"  v1̄11 / pulling : {diag_results['velocity_ratio_bar111_to_pulling']:.1f}×")
    if math.isfinite(diag_results["velocity_diff_m_s"]):
        lines.append(f"  Δv = v001-v1̄11 : {diag_results['velocity_diff_m_s']:.6e} m/s")
        lines.append(f"  Relative Δv    : {diag_results['velocity_diff_rel'] * 100:.1f}%")
    else:
        lines.append("  Δv = v001-v1̄11 : n/a")
    lines.append(f"  Verdict   : {diag_results['verdict']}")
    lines.append(f"  Note      : {diag_results['note']}")

    # Case A
    lines.append("")
    lines.append("─" * 40)
    lines.append("Case A: Single-grain isothermal (Fig. 2)")
    lines.append(f"  Steps  : {results_A['nsteps']} (paper: 2750)")
    lines.append(f"  φ_s initial : {results_A['initial_solid_frac']:.4f}")
    lines.append(f"  φ_s final   : {results_A['final_solid_frac']:.4f}")
    lines.append(f"  Interface std (anisotropy): {results_A['interface_height_std']:.2f} cells")
    status_A = "DETECTED" if results_A["faceting_pass"] else "NOT DETECTED"
    lines.append(f"  Faceting signal : {status_A}  (qualitative only)")

    # Case B
    lines.append("")
    lines.append("─" * 40)
    lines.append("Case B: Two-grain, v=40 μm/s (Fig. 3b, kinetic mode)")
    lines.append(f"  Steps  : {results_B['nsteps']} (paper: 22 500)")
    lines.append(f"  Expected growth : {results_B['expected_growth_cells']:.1f} cells")
    lines.append(f"  GB angle    : {fmt_angle(results_B['gb_angle_deg'])}")
    lines.append(f"  GB n_pts    : {results_B['gb_n_points']}")
    lines.append(f"  Paper ref   : ~+27.35° (bisector, Fig.5a)")
    if not results_B["gb_reliable"]:
        status_B = "INCONCLUSIVE"
    else:
        status_B = "PASS" if results_B["direction_pass"] else "FAIL"
    lines.append(f"  Direction   : {status_B}  (positive = grain-1/<001> expands)")

    # Case C
    lines.append("")
    lines.append("─" * 40)
    lines.append("Case C: Two-grain, v=1 μm/s (Fig. 3d, energetic mode)")
    lines.append(f"  Steps  : {results_C['nsteps']}  ⚠ paper needs 900 000 steps")
    lines.append(f"  Expected growth : {results_C['expected_growth_cells']:.2f} cells")
    lines.append(f"  GB angle    : {fmt_angle(results_C['gb_angle_deg'])}")
    lines.append(f"  GB n_pts    : {results_C['gb_n_points']}")
    lines.append(f"  Paper ref   : small negative angle (energetic mode, Fig.5a)")
    lines.append(f"  NOTE: {results_C['note']}")

    # Compare B vs C inclination direction
    lines.append("")
    lines.append("─" * 40)
    lines.append("Rate comparison (key physics of paper Sec.4.1):")
    angle_B = results_B["gb_angle_deg"]
    angle_C = results_C["gb_angle_deg"]
    lines.append(f"  High rate (40 μm/s): {fmt_angle(angle_B)}")
    lines.append(f"  Low  rate ( 1 μm/s): {fmt_angle(angle_C)}")
    if results_B["gb_reliable"] and results_C["gb_reliable"] and angle_B > angle_C:
        lines.append("  ✓ High rate > Low rate: kinetic-to-energetic transition reproduced")
    elif not results_C["gb_reliable"]:
        lines.append("  ? Inconclusive: low-rate run did not grow beyond the initial seed buffer")
    else:
        lines.append("  ? Trend not yet visible at these step counts")

    lines.append("")
    lines.append("=" * 66)
    report_str = "\n".join(lines)

    # Prepend parameter check
    full_report = param_check_str + "\n\n" + report_str

    # Print to stdout
    print("\n" + full_report)

    # Write to file
    report_path = os.path.join(OUT_DIR, "verification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\nReport saved to: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 66)
    print("Paper verification: Zhu et al. Materialia 27 (2023) 101702")
    print("=" * 66)
    print(f"Output directory : {os.path.abspath(OUT_DIR)}")
    print(f"KERNEL_KMAX      : {KERNEL_KMAX}")

    # Check CUDA
    if not cuda.is_available():
        print("\n✗ CUDA is not available on this machine. Aborting.")
        sys.exit(1)
    gpu_name = cuda.gpus[0].name.decode() if cuda.gpus else "unknown"
    print(f"CUDA device      : {gpu_name}")

    # — Section 1: Parameter check —
    param_str = verify_parameters()
    print(param_str)

    # — Numerical audit / orientation diagnostic —
    audit_results = run_numerical_regime_audit()
    diag_results = run_orientation_velocity_diagnostics()

    # — Case A —
    res_A = run_case_A()

    # — Case B —
    res_B = run_case_B()

    # — Case C —
    res_C = run_case_C()

    # — Final report —
    write_report(param_str, audit_results, diag_results, res_A, res_B, res_C)


if __name__ == "__main__":
    main()
