"""
gpu_kernels.py
==============
CUDA device functions and kernels for multi-phase field solidification.

All device functions and kernels from the original notebook are collected here
so that run_singlemode.py and run_twomode.py can import and reuse them without
modifying the notebook itself.

Compile-time constants captured by CUDA:
  KMAX = 18  — max active phases per cell (local array size in kernel)
  LIQ  = 0   — liquid phase ID is always 0
"""

import math
from numba import cuda, float32, int32

# ─── Compile-time constants (captured at first kernel JIT compile) ────────────
KMAX = 50   # Must match config gpu.KMAX
LIQ  = 0    # Liquid phase index


# ─── Surface / kinetic anisotropy ────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def calc_a_from_cos(cost, a0, delta_a, mu_a, p_round):
    """Surface energy anisotropy coefficient a(θ). Paper Appendix A (A2)-(A4)."""
    c2 = cost * cost
    C  = math.sqrt(c2 + p_round * p_round)
    S  = math.sqrt(max(1.0 - c2, 0.0) + p_round * p_round)
    return mu_a * (1.0 + delta_a * (C + math.tan(a0) * S))


@cuda.jit(device=True, inline=True)
def calc_b_from_cos(best_cost, ksi, theta_c_rad):
    """
    Revised kinetic anisotropy factor b(theta).

    Intended to match Fig.1(c) of the paper:
      - b(theta=0) = ksi
      - cusp ends at theta_c_rad
      - b(theta >= theta_c_rad) = 1

    Parameters
    ----------
    best_cost : float
        cos(theta)
    ksi : float
        cusp depth parameter (zeta/xi in paper notation)
    theta_c_rad : float
        cusp end angle in radians, e.g. 10 deg -> pi/18
    """
    # clamp for safety
    c = best_cost
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0

    theta = math.acos(c)

    # saturate outside cusp region
    if theta >= theta_c_rad:
        return 1.0

    # normalized angle in [0, 1)
    x = theta / theta_c_rad

    # map x in [0,1) to y in [0, pi/2)
    # so that b rises smoothly from ksi to ~1
    eps = 1.0e-6
    y = 0.5 * math.pi * x
    if y > 0.5 * math.pi - eps:
        y = 0.5 * math.pi - eps
    if y < eps:
        y = eps

    t = math.tan(y)
    return ksi + (1.0 - ksi) * t * math.tanh(1.0 / t)


# ─── Boundary condition helpers ───────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def idx_xp(l, nx):
    """Periodic boundary: x+1."""
    return l + 1 if l < nx - 1 else 0


@cuda.jit(device=True, inline=True)
def idx_xm(l, nx):
    """Periodic boundary: x-1."""
    return l - 1 if l > 0 else nx - 1


@cuda.jit(device=True, inline=True)
def idx_yp(m, ny):
    """Neumann boundary (mirror): y+1."""
    return m + 1 if m < ny - 1 else ny - 1


@cuda.jit(device=True, inline=True)
def idx_ym(m, ny):
    """Neumann boundary (mirror): y-1."""
    return m - 1 if m > 0 else 0


@cuda.jit(device=True, inline=True)
def grad_phi_xy(phi, gid, l, m, nx, ny, dx):
    """Central-difference gradient of phi[gid] at (l, m)."""
    lp = idx_xp(l, nx); lm = idx_xm(l, nx)
    mp = idx_yp(m, ny); mm = idx_ym(m, ny)
    gx = (phi[gid, lp, m] - phi[gid, lm, m]) / (2.0 * dx)
    gy = (phi[gid, l,  mp] - phi[gid, l,  mm]) / (2.0 * dx)
    return gx, gy


# ─── Best-cos helpers ─────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def best_cos_from_grad(gx, gy, n111, solidid, g2_floor):
    """Max |cos θ| between gradient direction and 8 {111} normals of grain solidid."""
    g2 = gx * gx + gy * gy
    if g2 < g2_floor:
        return 0.0
    inv_g = 1.0 / math.sqrt(g2)
    nxn = gx * inv_g
    nyn = gy * inv_g
    best = 0.0
    for t in range(8):
        c = abs(nxn * n111[solidid, t, 0] + nyn * n111[solidid, t, 1])
        if c > best:
            best = c
    if best > 1.0:
        best = 1.0
    return best


# ─── Epsilon-squared at a cell ────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def eps2_at_cell_from_liquid(phi, l, m, nx, ny, dx, solidid,
                              eps0_sl, a0, delta_a, mu_a, p_round,
                              n111, g2_floor):
    """Anisotropic ε²(θ) at cell (l,m) for the given solid grain."""
    gx, gy   = grad_phi_xy(phi, LIQ, l, m, nx, ny, dx)
    best_cos = best_cos_from_grad(gx, gy, n111, solidid, g2_floor)
    a_sl     = calc_a_from_cos(best_cos, a0, delta_a, mu_a, p_round)
    eps_sl   = eps0_sl * a_sl
    return eps_sl * eps_sl


# ─── Anisotropic diffusion term 1: ∇·(ε²∇φ_liquid) ──────────────────────────

@cuda.jit(device=True, inline=True)
def aniso_term1_solid(phi, l, m, nx, ny, dx, solidid, eps0_sl,
                      a0, delta_a, mu_a, p_round, n111, g2_floor):
    """∇·(ε²∇φ_liquid) with spatially varying ε² for grain solidid."""
    lp = idx_xp(l, nx); lm = idx_xm(l, nx)
    mp = idx_yp(m, ny); mm = idx_ym(m, ny)

    eps2_c  = eps2_at_cell_from_liquid(phi, l,  m,  nx, ny, dx, solidid, eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor)
    eps2_xp = eps2_at_cell_from_liquid(phi, lp, m,  nx, ny, dx, solidid, eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor)
    eps2_xm = eps2_at_cell_from_liquid(phi, lm, m,  nx, ny, dx, solidid, eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor)
    eps2_yp = eps2_at_cell_from_liquid(phi, l,  mp, nx, ny, dx, solidid, eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor)
    eps2_ym = eps2_at_cell_from_liquid(phi, l,  mm, nx, ny, dx, solidid, eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor)

    phi_c  = phi[LIQ, l,  m]
    phi_xp = phi[LIQ, lp, m]
    phi_xm = phi[LIQ, lm, m]
    phi_yp = phi[LIQ, l,  mp]
    phi_ym = phi[LIQ, l,  mm]

    Fx_p = (eps2_c + eps2_xp) * (phi_xp - phi_c) / (2.0 * dx)
    Fx_m = (eps2_c + eps2_xm) * (phi_c  - phi_xm) / (2.0 * dx)
    Fy_p = (eps2_c + eps2_yp) * (phi_yp - phi_c) / (2.0 * dx)
    Fy_m = (eps2_c + eps2_ym) * (phi_c  - phi_ym) / (2.0 * dx)

    return (Fx_p - Fx_m) / dx + (Fy_p - Fy_m) / dx


# ─── 2nd derivatives ─────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def d2_phi_xy(phi, gid, l, m, nx, ny, dx):
    """Second-order finite-difference second derivatives of phi[gid]."""
    lp = idx_xp(l, nx); lm = idx_xm(l, nx)
    mp = idx_yp(m, ny); mm = idx_ym(m, ny)

    c  = phi[gid, l,  m]
    xp = phi[gid, lp, m]; xm = phi[gid, lm, m]
    yp = phi[gid, l,  mp]; ym = phi[gid, l,  mm]

    inv_dx2 = 1.0 / (dx * dx)
    phixx   = (xp - 2.0*c + xm) * inv_dx2
    phiyy   = (yp - 2.0*c + ym) * inv_dx2

    xpy = phi[gid, lp, mp]; xpm = phi[gid, lp, mm]
    xmy = phi[gid, lm, mp]; xmm = phi[gid, lm, mm]
    phixy = (xpy - xpm - xmy + xmm) * (0.25 * inv_dx2)

    return phixx, phiyy, phixy


# ─── Facet selection ──────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def facet_cos_and_nxy_from_grad(phix, phiy, n111, solidid, g2_floor):
    """Identify best-aligned {111} facet; return signed cos, abs cos, and (nx, ny)."""
    q = phix * phix + phiy * phiy
    if q < g2_floor:
        return float32(0.0), float32(0.0), float32(1.0), float32(0.0)

    inv_g = 1.0 / math.sqrt(q)
    gx = phix * inv_g; gy = phiy * inv_g

    best_abs    = 0.0; best_signed = 0.0
    best_nx     = 1.0; best_ny     = 0.0

    for t in range(8):
        ux  = n111[solidid, t, 0]; uy = n111[solidid, t, 1]
        dot = gx * ux + gy * uy
        ad  = abs(dot)
        if ad > best_abs:
            best_abs = ad; best_signed = dot
            best_nx  = ux; best_ny    = uy

    if best_abs    > 1.0:  best_abs    = 1.0
    if best_signed > 1.0:  best_signed = 1.0
    if best_signed < -1.0: best_signed = -1.0

    return float32(best_signed), float32(best_abs), float32(best_nx), float32(best_ny)


# ─── A12: da/dφ_x, da/dφ_y ───────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def da_dphixy_A12(phi, l, m, nx, ny, dx, solidid,
                  a0, delta_a, mu_a, p_round, n111, g2_floor):
    """Return (a, da/dφ_x, da/dφ_y) at cell (l,m). Paper Appendix A12."""
    phix, phiy = grad_phi_xy(phi, LIQ, l, m, nx, ny, dx)
    q = phix * phix + phiy * phiy
    if q < g2_floor:
        return float32(0.0), float32(0.0), float32(0.0)

    cos_signed, cos_abs, nxn, nyn = facet_cos_and_nxy_from_grad(
        phix, phiy, n111, solidid, g2_floor)

    a_val = calc_a_from_cos(cos_abs, a0, delta_a, mu_a, p_round)

    g      = math.sqrt(q); inv_g = 1.0 / g; inv_g3 = 1.0 / (q * g)
    udotg  = phix * nxn + phiy * nyn
    dc_dphix = nxn * inv_g - udotg * phix * inv_g3
    dc_dphiy = nyn * inv_g - udotg * phiy * inv_g3

    c    = cos_signed
    C    = math.sqrt(c * c + p_round * p_round)
    S    = math.sqrt(max(1.0 - c * c, 0.0) + p_round * p_round)
    coef = (c / C) - math.tan(a0) * (c / S)

    da_dphix = mu_a * delta_a * coef * dc_dphix
    da_dphiy = mu_a * delta_a * coef * dc_dphiy

    return float32(a_val), float32(da_dphix), float32(da_dphiy)


# ─── A13: neighbour differences of da/dφ_p ────────────────────────────────────

@cuda.jit(device=True, inline=True)
def d_dx_da_dphix_and_d_dy_da_dphiy_A13(phi, l, m, nx, ny, dx,
                                         solidid, a0, delta_a, mu_a,
                                         p_round, n111, g2_floor):
    """Central-difference approximation of A13 terms."""
    lp = idx_xp(l, nx); lm = idx_xm(l, nx)
    mp = idx_yp(m, ny); mm = idx_ym(m, ny)

    _, da_dphix_lp, _ = da_dphixy_A12(phi, lp, m,  nx, ny, dx, solidid, a0, delta_a, mu_a, p_round, n111, g2_floor)
    _, da_dphix_lm, _ = da_dphixy_A12(phi, lm, m,  nx, ny, dx, solidid, a0, delta_a, mu_a, p_round, n111, g2_floor)
    _, _, da_dphiy_mp = da_dphixy_A12(phi, l,  mp, nx, ny, dx, solidid, a0, delta_a, mu_a, p_round, n111, g2_floor)
    _, _, da_dphiy_mm = da_dphixy_A12(phi, l,  mm, nx, ny, dx, solidid, a0, delta_a, mu_a, p_round, n111, g2_floor)

    d_dx = (da_dphix_lp - da_dphix_lm) * (0.5 / dx)
    d_dy = (da_dphiy_mp - da_dphiy_mm) * (0.5 / dx)

    return float32(d_dx), float32(d_dy)


# ─── Torque term A11 ──────────────────────────────────────────────────────────

@cuda.jit(device=True, inline=True)
def torque_A11(phi, l, m, nx, ny, dx, solidid, eps0_sl,
               a0, delta_a, mu_a, p_round, n111, g2_floor):
    """Full torque term (ε₀²)(Ex + Ey). Paper Appendix A11, 3-term form."""
    phix, phiy = grad_phi_xy(phi, LIQ, l, m, nx, ny, dx)
    q = phix * phix + phiy * phiy
    if q < g2_floor:
        return 0.0

    phixx, phiyy, phixy = d2_phi_xy(phi, LIQ, l, m, nx, ny, dx)
    q_x = 2.0 * (phix * phixx + phiy * phixy)
    q_y = 2.0 * (phix * phixy + phiy * phiyy)

    a_val, da_dphix, da_dphiy = da_dphixy_A12(
        phi, l, m, nx, ny, dx, solidid, a0, delta_a, mu_a, p_round, n111, g2_floor)

    cos_signed, cos_abs, nxn, nyn = facet_cos_and_nxy_from_grad(
        phix, phiy, n111, solidid, g2_floor)

    g      = math.sqrt(q); inv_g = 1.0 / g; inv_g3 = 1.0 / (q * g)
    udotg  = phix * nxn + phiy * nyn
    dc_dphix = nxn * inv_g - udotg * phix * inv_g3
    dc_dphiy = nyn * inv_g - udotg * phiy * inv_g3

    c_x = dc_dphix * phixx + dc_dphiy * phixy
    c_y = dc_dphix * phixy + dc_dphiy * phiyy

    c    = cos_signed
    C    = math.sqrt(c * c + p_round * p_round)
    S    = math.sqrt(max(1.0 - c * c, 0.0) + p_round * p_round)
    coef = (c / C) - math.tan(a0) * (c / S)

    da_dx = mu_a * delta_a * coef * c_x
    da_dy = mu_a * delta_a * coef * c_y

    d_dx_da_dphix, d_dy_da_dphiy = d_dx_da_dphix_and_d_dy_da_dphiy_A13(
        phi, l, m, nx, ny, dx, solidid, a0, delta_a, mu_a, p_round, n111, g2_floor)

    # (I) + (II) + (III) for x and y
    Ex = (da_dx * da_dphix * q) + (a_val * d_dx_da_dphix * q) + (a_val * da_dphix * q_x)
    Ey = (da_dy * da_dphiy * q) + (a_val * d_dy_da_dphiy * q) + (a_val * da_dphiy * q_y)

    return (eps0_sl * eps0_sl) * (Ex + Ey)


# ─── APT update kernel ────────────────────────────────────────────────────────

@cuda.jit
def kernel_update_nfmf(phi, mf, nf, nx, ny, number_of_grain):
    """Update active phase list (APT) per cell.

    A phase i is active at (l,m) if phi[i,l,m] > 0, or if any
    face-neighbour has phi[i,...] > 0 (one-cell buffer zone).
    """
    l, m = cuda.grid(2)
    if l >= nx or m >= ny:
        return
    l_p = idx_xp(l, nx); l_m = idx_xm(l, nx)
    m_p = idx_yp(m, ny); m_m = idx_ym(m, ny)
    n = 0
    for i in range(number_of_grain):
        if (phi[i, l, m] > 0.0) or ((phi[i, l, m] == 0.0) and (
                (phi[i, l_p, m] > 0.0) or (phi[i, l_m, m] > 0.0) or
                (phi[i, l,  m_p] > 0.0) or (phi[i, l,  m_m] > 0.0))):
            n += 1
            mf[n - 1, l, m] = i
    nf[l, m] = n


# ─── Main phase field evolution kernel ───────────────────────────────────────

@cuda.jit
def kernel_update_phasefield_active(phi, phi_new, temp, mf, nf,
                                     wij, aij, mij, n111,
                                     nx, ny, number_of_grain,
                                     dx, dt, T_melt, Sf,
                                     eps0_sl, w0_sl,
                                     a0, delta_a, mu_a, p_round,
                                     g2_floor, ksi, theta_c_rad):
    """Main phase field time-step kernel with anisotropic solid-liquid energy.

    Implements the multi-phase field model with:
      - Anisotropic gradient energy (ε²(θ))
      - Torque term (A11)
      - Anisotropic kinetics b(θ) via calc_b_from_cos(cos, ksi, theta_c_rad)
      - APT (active parameter tracking) for efficiency

    Parameters
    ----------
    ksi         : cusp depth (b=ksi at theta=0, b=1 at theta>=theta_c_rad)
    theta_c_rad : cusp end angle [radians], e.g. 10 deg -> pi/18
    """
    l, m = cuda.grid(2)
    if l >= nx or m >= ny:
        return

    lp = idx_xp(l, nx); lm = idx_xm(l, nx)
    mp = idx_yp(m, ny); mm = idx_ym(m, ny)
    inv_dx2 = 1.0 / (dx * dx)
    Tcur    = temp[l, m]

    # Cache anisotropic terms for each active phase
    lap_sl = cuda.local.array(KMAX, float32)
    b_sl   = cuda.local.array(KMAX, float32)
    w_sl_arr = cuda.local.array(KMAX, float32)
    for t in range(nf[l, m]):
        gid = mf[t, l, m]
        if gid == LIQ:
            b_sl[t]   = 1.0
            lap_sl[t] = 0.0
        else:
            lap_sl[t] = aniso_term1_solid(
                phi, l, m, nx, ny, dx, gid, eps0_sl,
                a0, delta_a, mu_a, p_round, n111, g2_floor)
            lap_sl[t] += torque_A11(
                phi, l, m, nx, ny, dx, gid, eps0_sl,
                a0, delta_a, mu_a, p_round, n111, g2_floor)
            gx, gy   = grad_phi_xy(phi, LIQ, l, m, nx, ny, dx)
            best_cos = best_cos_from_grad(gx, gy, n111, gid, g2_floor)
            b_sl[t]  = calc_b_from_cos(best_cos, ksi, theta_c_rad)

    # Per-solid anisotropic w_sl
    for t in range(nf[l, m]):
        gid = mf[t, l, m]
        if gid == LIQ:
            w_sl_arr[t] = w0_sl
        else:
            phix_ = (phi[gid, lp, m] - phi[gid, lm, m]) * (0.5 / dx)
            phiy_ = (phi[gid, l,  mp] - phi[gid, l,  mm]) * (0.5 / dx)
            gn2   = phix_ * phix_ + phiy_ * phiy_
            cmax  = 0.0
            if gn2 >= g2_floor:
                inv_gn = 1.0 / math.sqrt(gn2)
                nx_ = phix_ * inv_gn; ny_ = phiy_ * inv_gn
                for tt in range(8):
                    c = abs(nx_ * n111[gid, tt, 0] + ny_ * n111[gid, tt, 1])
                    if c > cmax:
                        cmax = c
                if cmax > 1.0:
                    cmax = 1.0
            a_loc = calc_a_from_cos(cmax, a0, delta_a, mu_a, p_round)
            w_sl_arr[t] = w0_sl * (a_loc * a_loc)

    # Zero phi_new
    for i in range(number_of_grain):
        phi_new[i, l, m] = 0.0

    # Main evolution loop
    for t1 in range(nf[l, m]):
        i   = mf[t1, l, m]
        dpi = 0.0

        for t2 in range(nf[l, m]):
            j = mf[t2, l, m]
            if i == j:
                continue

            driving_force = 0.0
            if i != LIQ and j == LIQ:
                driving_force = -Sf * (Tcur - T_melt)
            elif i == LIQ and j != LIQ:
                driving_force =  Sf * (Tcur - T_melt)

            ppp = 0.0
            for t3 in range(nf[l, m]):
                k = mf[t3, l, m]

                lap_k = (phi[k, lp, m] + phi[k, lm, m] +
                         phi[k, l,  mp] + phi[k, l,  mm] -
                         4.0 * phi[k, l, m]) * inv_dx2

                wik = wij[i, k]; wjk = wij[j, k]
                if k == LIQ:
                    if i != LIQ: wik = w_sl_arr[t1]
                    if j != LIQ: wjk = w_sl_arr[t2]
                term1 = (wik - wjk) * phi[k, l, m]

                term2 = 0.0
                if j == LIQ and i != LIQ:
                    if k == i:
                        term2 = 0.5 * lap_sl[t1]
                elif i == LIQ and j != LIQ:
                    if k == j:
                        term2 = -0.5 * lap_sl[t2]
                else:
                    eps2ik = aij[i, k] * aij[i, k]
                    eps2jk = aij[j, k] * aij[j, k]
                    term2  = 0.5 * (eps2ik - eps2jk) * lap_k

                ppp += term1 + term2

            phii_phij  = phi[i, l, m] * phi[j, l, m]
            term_force = (8.0 / 3.1415926535) * math.sqrt(max(phii_phij, 0.0)) * driving_force

            mij_eff = mij[i, j]
            if i != LIQ and j == LIQ:
                mij_eff = mij[i, j] * b_sl[t1]
            elif i == LIQ and j != LIQ:
                mij_eff = mij[i, j] * b_sl[t2]

            dpi -= 2.0 * mij_eff / float(nf[l, m]) * (ppp - term_force)

        phi_new[i, l, m] = phi[i, l, m] + dt * dpi

    # Projection / normalisation
    s = 0.0
    for t in range(nf[l, m]):
        i = mf[t, l, m]
        v = phi_new[i, l, m]
        if v < 0.0: v = 0.0
        if v > 1.0: v = 1.0
        phi_new[i, l, m] = v
        s += v

    if s > 1e-20:
        invs = 1.0 / s
        for t in range(nf[l, m]):
            i = mf[t, l, m]
            phi_new[i, l, m] *= invs
    else:
        phi_new[LIQ, l, m] = 1.0


# ─── Temperature update kernel ────────────────────────────────────────────────

@cuda.jit
def kernel_update_temp(temp, cooling_rate, nx, ny):
    """Uniform temperature decrease: T^{n+1} = T^n - G·V_pulling·dt."""
    l, m = cuda.grid(2)
    if l < nx and m < ny:
        temp[l, m] -= cooling_rate
