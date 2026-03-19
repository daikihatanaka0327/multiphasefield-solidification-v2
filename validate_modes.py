"""
validate_modes.py -- CPU-only validation (no GPU needed).
Run with: python validate_modes.py
"""
import sys, io, math
# Force UTF-8 output so special chars print safely on Windows cp932
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import yaml
from orientation_utils import build_quaternion_from_config, compute_rotated_n111
from seed_modes import (init_singlemode_phi, init_twomode_phi,
                        init_temperature_field, build_interaction_matrices)

SEP = "-" * 60
PASS = "[OK  ]"
FAIL = "[FAIL]"

def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def ck(label, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"  {tag} {label}" + (f"  -- {detail}" if detail else ""))
    return cond

# ---- config ------------------------------------------------------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

pi        = math.pi
nx        = cfg["grid"]["nx"]
ny        = cfg["grid"]["ny"]
dx        = cfg["grid"]["dx"]
dy        = cfg["grid"]["dy"]
dt        = cfg["grid"]["dt"]
T_melt    = cfg["physical"]["T_melt"]
G         = cfg["physical"]["G"]
V_pulling = cfg["physical"]["V_pulling"]
Sf        = cfg["physical"]["Sf"]
delta     = cfg["interface"]["delta_factor"] * dx
gamma_100 = cfg["interface"]["gamma_100"]
gamma_GB  = cfg["interface"]["gamma_GB"]
M_SL      = cfg["mobility"]["M_SL"]
M_GB      = M_SL * cfg["mobility"]["M_GB_ratio"]
sm_cfg    = cfg.get("singlemode", {})
tm_cfg    = cfg.get("twomode", {})
sh_s      = int(sm_cfg.get("seed_height", cfg["seed"].get("height", 32)))
sh_t      = int(tm_cfg.get("seed_height", cfg["seed"].get("height", 32)))
split_idx = int(nx * float(tm_cfg.get("split_ratio", 0.5)))

def epsg(g): return math.sqrt(8*delta*g/(pi*pi))
def wg(g):   return 4*g/delta
def mM(M):   return (pi*pi/(8*delta))*M

eps0_sl  = epsg(gamma_100); w0_sl   = wg(gamma_100); m_sl_phi = mM(M_SL)
eps_GB   = epsg(gamma_GB);  w_GB    = wg(gamma_GB);  m_GB_phi = mM(M_GB)

# ==============================================================================
section("1. SINGLEMODE -- initial phi")
# ==============================================================================
phi_s = init_singlemode_phi(nx, ny, dy, delta, sh_s, solid_gid=1)

ck("shape == (2, nx, ny)",  phi_s.shape == (2, nx, ny),  str(phi_s.shape))
ck("dtype == float32",      phi_s.dtype == np.float32)
ck("phi >= 0",              float(phi_s.min()) >= -1e-6, f"min={phi_s.min():.4e}")
ck("phi <= 1",              float(phi_s.max()) <=  1+1e-6, f"max={phi_s.max():.4e}")

phi_sum_s = phi_s.sum(axis=0)
merr_s    = float(np.abs(phi_sum_s - 1.0).max())
ck("sum_i phi_i == 1",      merr_s < 1e-5, f"max|sum-1|={merr_s:.2e}")

pm_s = np.argmax(phi_s, axis=0)
sf   = float((pm_s == 1).mean())
lf   = float((pm_s == 0).mean())
ck("solid phase present",  sf > 0.05, f"solid frac={sf:.3f}")
ck("liquid phase present", lf > 0.05, f"liquid frac={lf:.3f}")

col0   = pm_s[0, :]
xunif  = all(np.array_equal(pm_s[l, :], col0) for l in range(1, nx))
ck("x-direction uniform (no GB)", xunif)

solid_below = float((pm_s[:, :sh_s] == 1).mean())
liq_above   = float((pm_s[:, sh_s+5:] == 0).mean())
ck(f"solid dom below y={sh_s}",    solid_below > 0.8, f"{solid_below:.3f}")
ck(f"liquid dom above y={sh_s+5}", liq_above   > 0.8, f"{liq_above:.3f}")

fac   = 2.2 / delta
m_arr = np.arange(ny, dtype=np.float64)
dist  = m_arr * dy - sh_s * dy
phi_th = (0.5 * (1.0 - np.tanh(fac * dist))).astype(np.float32)
tanh_err = float(np.abs(phi_s[1, 0, :] - phi_th).max())
ck("solid profile matches tanh formula", tanh_err < 1e-5,
   f"max_err={tanh_err:.2e}")

print(f"\n  Fractions: liquid={lf:.3f}  solid={sf:.3f}")
print(f"  phi[0] [{phi_s[0].min():.4f}, {phi_s[0].max():.4f}]"
      f"   phi[1] [{phi_s[1].min():.4f}, {phi_s[1].max():.4f}]")
print(f"  max|sum-1| = {merr_s:.2e}")

# APT (nf) on CPU
nf_s = np.zeros((nx, ny), dtype=np.int32)
for i in range(2):
    pi_ = phi_s[i]
    nb  = ((np.roll(pi_,  1, 0) > 0) | (np.roll(pi_, -1, 0) > 0) |
           (np.roll(pi_,  1, 1) > 0) | (np.roll(pi_, -1, 1) > 0))
    nf_s += ((pi_ > 0) | ((pi_ == 0) & nb)).astype(np.int32)
ck("nf >= 1 everywhere",    int(nf_s.min()) >= 1, f"min={nf_s.min()}")
ck("nf <= 2 for N=2",       int(nf_s.max()) <= 2, f"max={nf_s.max()}")
print(f"  nf: min={nf_s.min()}  max={nf_s.max()}  mean={nf_s.mean():.2f}")

# ==============================================================================
section("2. TWOMODE -- initial phi")
# ==============================================================================
phi_t = init_twomode_phi(nx, ny, dy, delta, sh_t, split_idx)

ck("shape == (3, nx, ny)",  phi_t.shape == (3, nx, ny), str(phi_t.shape))
ck("dtype == float32",      phi_t.dtype == np.float32)
ck("phi >= 0",              float(phi_t.min()) >= -1e-6, f"min={phi_t.min():.4e}")
ck("phi <= 1",              float(phi_t.max()) <=  1+1e-6, f"max={phi_t.max():.4e}")

phi_sum_t = phi_t.sum(axis=0)
merr_t    = float(np.abs(phi_sum_t - 1.0).max())
ck("sum_i phi_i == 1",  merr_t < 1e-5, f"max|sum-1|={merr_t:.2e}")

pm_t  = np.argmax(phi_t, axis=0)
lft   = float((pm_t == 0).mean())
g1ft  = float((pm_t == 1).mean())
g2ft  = float((pm_t == 2).mean())
ck("liquid>0", lft  > 0.05, f"{lft:.3f}")
ck("grain1>0", g1ft > 0.05, f"{g1ft:.3f}")
ck("grain2>0", g2ft > 0.05, f"{g2ft:.3f}")

mid_m = sh_t // 2
cl = int(pm_t[split_idx - 1, mid_m])
cr = int(pm_t[split_idx,     mid_m])
ck("phase_map[split-1, mid_m] == 1 (grain1 side)", cl == 1, f"got {cl}")
ck("phase_map[split,   mid_m] == 2 (grain2 side)", cr == 2, f"got {cr}")

g1_right = bool((pm_t[split_idx:, :sh_t-2] == 1).any())
g2_left  = bool((pm_t[:split_idx, :sh_t-2] == 2).any())
ck("grain1 absent from right-half solid region", not g1_right)
ck("grain2 absent from left-half  solid region", not g2_left)

solid_bot = float(((pm_t[:, :sh_t] == 1) | (pm_t[:, :sh_t] == 2)).mean())
liq_top   = float((pm_t[:, sh_t+5:] == 0).mean())
ck(f"solid dom below y={sh_t}",    solid_bot > 0.8, f"{solid_bot:.3f}")
ck(f"liquid dom above y={sh_t+5}", liq_top   > 0.8, f"{liq_top:.3f}")

print(f"\n  Fractions: liquid={lft:.3f}  grain1={g1ft:.3f}  grain2={g2ft:.3f}")
print(f"  phi ranges: [0]=[{phi_t[0].min():.4f},{phi_t[0].max():.4f}]"
      f"  [1]=[{phi_t[1].min():.4f},{phi_t[1].max():.4f}]"
      f"  [2]=[{phi_t[2].min():.4f},{phi_t[2].max():.4f}]")
print(f"  max|sum-1| = {merr_t:.2e}")

nf_t = np.zeros((nx, ny), dtype=np.int32)
for i in range(3):
    pi_ = phi_t[i]
    nb  = ((np.roll(pi_,  1, 0) > 0) | (np.roll(pi_, -1, 0) > 0) |
           (np.roll(pi_,  1, 1) > 0) | (np.roll(pi_, -1, 1) > 0))
    nf_t += ((pi_ > 0) | ((pi_ == 0) & nb)).astype(np.int32)
ck("nf >= 1 everywhere",  int(nf_t.min()) >= 1, f"min={nf_t.min()}")
ck("nf <= 3 for N=3",     int(nf_t.max()) <= 3, f"max={nf_t.max()}")
print(f"  nf: min={nf_t.min()}  max={nf_t.max()}  mean={nf_t.mean():.2f}")

# ==============================================================================
section("3. TEMPERATURE FIELD")
# ==============================================================================
temp_s = init_temperature_field(nx, ny, T_melt, G, dy, sh_s)

ck("dtype == float32",  temp_s.dtype == np.float32)
ck("shape == (nx, ny)", temp_s.shape == (nx, ny))
ck("x-direction uniform",
   float(np.abs(temp_s - temp_s[0:1, :]).max()) < 1e-3)
ck(f"T at seed_height == T_melt",
   abs(float(temp_s[0, sh_s]) - T_melt) < 0.1,
   f"T={float(temp_s[0, sh_s]):.1f} K  T_melt={T_melt} K")
ck("T(0) < T_melt  (undercooled solid side)",
   float(temp_s[0, 0]) < T_melt, f"T(0)={float(temp_s[0,0]):.1f} K")
ck("T(-1) > T_melt (superheated liquid side)",
   float(temp_s[0, -1]) > T_melt, f"T(-1)={float(temp_s[0,-1]):.1f} K")
print(f"  T range: [{temp_s.min():.1f}, {temp_s.max():.1f}] K")

# ==============================================================================
section("4. QUATERNION and n111")
# ==============================================================================
q_s = build_quaternion_from_config(sm_cfg)
ck("singlemode quaternion is unit",
   abs(np.linalg.norm(q_s) - 1.0) < 1e-10,
   f"|q|={np.linalg.norm(q_s):.10f}")
print(f"  singlemode q = {np.round(q_s,6)}")

_dg1 = {"orientation_type": "euler", "euler_deg": [0, 0, 0]}
_dg2 = {"orientation_type": "euler", "euler_deg": [0, 20, 0]}
q_g1 = build_quaternion_from_config(tm_cfg.get("grain1", _dg1))
q_g2 = build_quaternion_from_config(tm_cfg.get("grain2", _dg2))
ck("grain1 quaternion unit", abs(np.linalg.norm(q_g1) - 1.0) < 1e-10)
ck("grain2 quaternion unit", abs(np.linalg.norm(q_g2) - 1.0) < 1e-10)
ck("grain1 != grain2", not np.allclose(q_g1, q_g2),
   f"g1={np.round(q_g1,4)}  g2={np.round(q_g2,4)}")
print(f"  grain1 q = {np.round(q_g1,6)}")
print(f"  grain2 q = {np.round(q_g2,6)}")

# n111 checks
gqs = np.zeros((2, 4)); gqs[0] = [0, 0, 0, 1]; gqs[1] = q_s
n111_s = compute_rotated_n111(gqs)
ck("n111_s shape == (2,8,3)", n111_s.shape == (2, 8, 3))
ck("n111_s dtype == float32", n111_s.dtype == np.float32)
norms_s = np.linalg.norm(n111_s, axis=-1)
ck("all n111_s are unit vectors",
   float(np.abs(norms_s - 1.0).max()) < 1e-5,
   f"max_err={np.abs(norms_s-1).max():.2e}")

n111b = np.array([[ 1, 1, 1],[ 1, 1,-1],[ 1,-1, 1],[-1, 1, 1],
                  [ 1,-1,-1],[-1, 1,-1],[-1,-1, 1],[-1,-1,-1]],
                 dtype=np.float32) / np.sqrt(3.0)
ck("identity rotation: n111 unchanged",
   float(np.abs(n111_s[1] - n111b).max()) < 1e-5,
   f"max_err={np.abs(n111_s[1]-n111b).max():.2e}")

gqt    = np.array([[0,0,0,1], q_g1, q_g2])
n111_t = compute_rotated_n111(gqt)
ck("n111_t shape == (3,8,3)", n111_t.shape == (3, 8, 3))
norms_t = np.linalg.norm(n111_t, axis=-1)
ck("all n111_t are unit vectors",
   float(np.abs(norms_t - 1.0).max()) < 1e-5)
n111_diff = float(np.abs(n111_t[1] - n111_t[2]).max())
ck("grain1 and grain2 have different n111",
   n111_diff > 0.01, f"max_diff={n111_diff:.4f}")

# ==============================================================================
section("5. INTERACTION MATRICES")
# ==============================================================================
wij2, aij2, mij2 = build_interaction_matrices(
    2, eps0_sl, w0_sl, m_sl_phi, eps_GB, w_GB, m_GB_phi)
ck("wij2 symmetric",       np.allclose(wij2, wij2.T))
ck("mij2 symmetric",       np.allclose(mij2, mij2.T))
ck("wij2[0,1] == w0_sl",   abs(float(wij2[0,1]) - w0_sl) < 1e-8,
   f"{wij2[0,1]:.4e} vs {w0_sl:.4e}")
ck("mij2[0,1] == m_sl_phi",abs(float(mij2[0,1]) - m_sl_phi) < 1e-8,
   f"{mij2[0,1]:.4e} vs {m_sl_phi:.4e}")
ck("wij2 diagonal == 0",   float(np.diag(wij2).max()) == 0.0)
print(f"  wij[N=2]:\n{wij2}")

wij3, aij3, mij3 = build_interaction_matrices(
    3, eps0_sl, w0_sl, m_sl_phi, eps_GB, w_GB, m_GB_phi)
ck("wij3 symmetric",       np.allclose(wij3, wij3.T))
ck("wij3[0,1] == w0_sl",   abs(float(wij3[0,1]) - w0_sl) < 1e-8)
ck("wij3[0,2] == w0_sl",   abs(float(wij3[0,2]) - w0_sl) < 1e-8)
ck("wij3[1,2] == w_GB",    abs(float(wij3[1,2]) - w_GB)  < 1e-8,
   f"{wij3[1,2]:.4e} vs {w_GB:.4e}")
print(f"  wij[N=3]:\n{wij3}")

# ==============================================================================
section("6. CPU MINI TIME-STEP (singlemode, 5 steps, NaN/stability check)")
# ==============================================================================
# Extract a 1x15 sub-grid around the interface for a pure-numpy test.
cx   = nx // 2
sub_phi  = phi_s[:, cx:cx+1, max(0, sh_s-5):sh_s+10].copy().astype(np.float64)
sub_temp = init_temperature_field(
    1, sub_phi.shape[2], T_melt, G, dy, sh_s).astype(np.float64)
cool    = G * V_pulling * dt
inv_dx2 = 1.0 / (dx * dx)
LIQ     = 0
wd = wij2.astype(np.float64)
ad = aij2.astype(np.float64)
md = mij2.astype(np.float64)

ok_nan = ok_range = ok_sum = True
for step in range(5):
    sub_temp -= cool
    N2, nx2, ny2 = sub_phi.shape
    pn = sub_phi.copy()
    for l in range(nx2):
        for m in range(ny2):
            lp = (l+1) % nx2; lm = (l-1) % nx2
            mp = min(m+1, ny2-1); mm_ = max(m-1, 0)
            Tc = float(sub_temp[l, m])
            for i in range(N2):
                dpi = 0.0
                for j in range(N2):
                    if i == j: continue
                    df = (-Sf*(Tc-T_melt) if (i!=LIQ and j==LIQ) else
                           Sf*(Tc-T_melt) if (i==LIQ and j!=LIQ) else 0.0)
                    ppp = 0.0
                    for k in range(N2):
                        lap = (sub_phi[k,lp,m]+sub_phi[k,lm,m]+
                               sub_phi[k,l,mp]+sub_phi[k,l,mm_] -
                               4*sub_phi[k,l,m]) * inv_dx2
                        t1 = (wd[i,k] - wd[j,k]) * sub_phi[k,l,m]
                        t2 = 0.5*(ad[i,k]**2 - ad[j,k]**2)*lap
                        ppp += t1 + t2
                    tf = (8/math.pi)*math.sqrt(max(sub_phi[i,l,m]*sub_phi[j,l,m],0))*df
                    dpi -= 2*md[i,j]/N2*(ppp - tf)
                pn[i,l,m] = sub_phi[i,l,m] + dt*dpi
            # clip + normalise
            s2 = 0.0
            for i in range(N2):
                v = pn[i,l,m]
                if v < 0: v = 0
                if v > 1: v = 1
                pn[i,l,m] = v; s2 += v
            if s2 > 1e-20:
                for i in range(N2): pn[i,l,m] /= s2
            else:
                pn[LIQ,l,m] = 1.0
    sub_phi = pn
    if np.isnan(sub_phi).any() or np.isinf(sub_phi).any():
        ok_nan = False; break
    if sub_phi.min() < -0.01 or sub_phi.max() > 1.01:
        ok_range = False
    if float(np.abs(sub_phi.sum(axis=0) - 1.0).max()) > 1e-4:
        ok_sum = False

ck("No NaN/inf after 5 steps", ok_nan)
ck("phi stays in [0,1]",        ok_range,
   f"min={sub_phi.min():.6f}  max={sub_phi.max():.6f}")
ck("sum_i phi_i == 1 maintained", ok_sum,
   f"max_err={np.abs(sub_phi.sum(axis=0)-1.0).max():.2e}")
print(f"  phi after 5 steps:"
      f" min={sub_phi.min():.6f}  max={sub_phi.max():.6f}"
      f"  sum-err={np.abs(sub_phi.sum(axis=0)-1.0).max():.2e}")

# ==============================================================================
section("7. PARAMETER VALUES & STABILITY ESTIMATE")
# ==============================================================================
print(f"  delta          = {delta:.3e} m  ({delta/dx:.1f} * dx)")
print(f"  eps0_sl        = {eps0_sl:.4e}")
print(f"  w0_sl          = {w0_sl:.4e}")
print(f"  m_sl_phi       = {m_sl_phi:.4e}")
print(f"  eps_GB         = {eps_GB:.4e}")
print(f"  w_GB           = {w_GB:.4e}")
print(f"  m_GB_phi       = {m_GB_phi:.4e}")
print(f"  cooling_rate   = {G*V_pulling*dt:.4e} K/step")
print(f"  g2_floor       = {(0.1/dx)**2:.4e}")
cfl = dt * m_sl_phi * w0_sl
print(f"  dt*m_sl*w0_sl  = {cfl:.4e}  (rough stability estimate)")
ck("dt * m_sl_phi * w0_sl < 0.5", cfl < 0.5, f"{cfl:.4e}")

print(f"\n{SEP}\n  ALL CHECKS COMPLETE\n{SEP}")
