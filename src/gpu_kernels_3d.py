"""
gpu_kernels_3d.py
=================
CUDA device functions and kernels for 3D multi-phase field solidification.

This module mirrors src/gpu_kernels.py but extends the phase field, active
parameter tracking, and temperature update kernels from 2D to 3D.

Boundary conditions
-------------------
  x : periodic
  y : periodic
  z : Neumann (mirror)
"""

import math
from numba import cuda, float32

# Compile-time constants captured by CUDA at first JIT compile.
KMAX = 50
LIQ = 0


@cuda.jit(device=True, inline=True)
def calc_a_from_cos(cost, a0, delta_a, mu_a, p_round):
    """Surface-energy anisotropy coefficient a(theta)."""
    c2 = cost * cost
    C = math.sqrt(c2 + p_round * p_round)
    S = math.sqrt(max(1.0 - c2, 0.0) + p_round * p_round)
    return mu_a * (1.0 + delta_a * (C + math.tan(a0) * S))


@cuda.jit(device=True, inline=True)
def calc_b_from_cos(best_cost, ksi, theta_c_rad):
    """Kinetic anisotropy factor b(theta)."""
    c = best_cost
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0

    theta = math.acos(c)
    if theta >= theta_c_rad:
        return 1.0

    x = theta / theta_c_rad
    eps = 1.0e-6
    y = 0.5 * math.pi * x
    if y > 0.5 * math.pi - eps:
        y = 0.5 * math.pi - eps
    if y < eps:
        y = eps

    t = math.tan(y)
    return ksi + (1.0 - ksi) * t * math.tanh(1.0 / t)


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
    """Periodic boundary: y+1."""
    return m + 1 if m < ny - 1 else 0


@cuda.jit(device=True, inline=True)
def idx_ym(m, ny):
    """Periodic boundary: y-1."""
    return m - 1 if m > 0 else ny - 1


@cuda.jit(device=True, inline=True)
def idx_zp(k, nz):
    """Neumann boundary (mirror): z+1."""
    return k + 1 if k < nz - 1 else nz - 1


@cuda.jit(device=True, inline=True)
def idx_zm(k, nz):
    """Neumann boundary (mirror): z-1."""
    return k - 1 if k > 0 else 0


@cuda.jit(device=True, inline=True)
def grad_phi_xyz(phi, gid, l, m, k, nx, ny, nz, dx):
    """Central-difference gradient of phi[gid] at (l, m, k)."""
    lp = idx_xp(l, nx)
    lm = idx_xm(l, nx)
    mp = idx_yp(m, ny)
    mm = idx_ym(m, ny)
    kp = idx_zp(k, nz)
    km = idx_zm(k, nz)

    gx = (phi[gid, lp, m, k] - phi[gid, lm, m, k]) / (2.0 * dx)
    gy = (phi[gid, l, mp, k] - phi[gid, l, mm, k]) / (2.0 * dx)
    gz = (phi[gid, l, m, kp] - phi[gid, l, m, km]) / (2.0 * dx)
    return gx, gy, gz


@cuda.jit(device=True, inline=True)
def best_cos_from_grad_3d(gx, gy, gz, n111, solidid, g2_floor):
    """Max |cos theta| between 3D gradient and 8 {111} normals."""
    g2 = gx * gx + gy * gy + gz * gz
    if g2 < g2_floor:
        return 0.0

    inv_g = 1.0 / math.sqrt(g2)
    nxn = gx * inv_g
    nyn = gy * inv_g
    nzn = gz * inv_g

    best = 0.0
    for t in range(8):
        c = abs(
            nxn * n111[solidid, t, 0]
            + nyn * n111[solidid, t, 1]
            + nzn * n111[solidid, t, 2]
        )
        if c > best:
            best = c

    if best > 1.0:
        best = 1.0
    return best


@cuda.jit(device=True, inline=True)
def eps2_at_cell_from_liquid_3d(phi, l, m, k, nx, ny, nz, dx, solidid,
                                eps0_sl, a0, delta_a, mu_a, p_round,
                                n111, g2_floor):
    """Anisotropic epsilon^2 at cell (l,m,k) for the given solid grain."""
    gx, gy, gz = grad_phi_xyz(phi, LIQ, l, m, k, nx, ny, nz, dx)
    best_cos = best_cos_from_grad_3d(gx, gy, gz, n111, solidid, g2_floor)
    a_sl = calc_a_from_cos(best_cos, a0, delta_a, mu_a, p_round)
    eps_sl = eps0_sl * a_sl
    return eps_sl * eps_sl


@cuda.jit(device=True, inline=True)
def aniso_term1_solid_3d(phi, l, m, k, nx, ny, nz, dx, solidid, eps0_sl,
                         a0, delta_a, mu_a, p_round, n111, g2_floor):
    """3D diffusion term div(eps^2 grad(phi_liquid))."""
    lp = idx_xp(l, nx)
    lm = idx_xm(l, nx)
    mp = idx_yp(m, ny)
    mm = idx_ym(m, ny)
    kp = idx_zp(k, nz)
    km = idx_zm(k, nz)

    eps2_c = eps2_at_cell_from_liquid_3d(
        phi, l, m, k, nx, ny, nz, dx, solidid,
        eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    eps2_xp = eps2_at_cell_from_liquid_3d(
        phi, lp, m, k, nx, ny, nz, dx, solidid,
        eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    eps2_xm = eps2_at_cell_from_liquid_3d(
        phi, lm, m, k, nx, ny, nz, dx, solidid,
        eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    eps2_yp = eps2_at_cell_from_liquid_3d(
        phi, l, mp, k, nx, ny, nz, dx, solidid,
        eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    eps2_ym = eps2_at_cell_from_liquid_3d(
        phi, l, mm, k, nx, ny, nz, dx, solidid,
        eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    eps2_zp = eps2_at_cell_from_liquid_3d(
        phi, l, m, kp, nx, ny, nz, dx, solidid,
        eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    eps2_zm = eps2_at_cell_from_liquid_3d(
        phi, l, m, km, nx, ny, nz, dx, solidid,
        eps0_sl, a0, delta_a, mu_a, p_round, n111, g2_floor,
    )

    phi_c = phi[LIQ, l, m, k]
    phi_xp = phi[LIQ, lp, m, k]
    phi_xm = phi[LIQ, lm, m, k]
    phi_yp = phi[LIQ, l, mp, k]
    phi_ym = phi[LIQ, l, mm, k]
    phi_zp = phi[LIQ, l, m, kp]
    phi_zm = phi[LIQ, l, m, km]

    Fx_p = (eps2_c + eps2_xp) * (phi_xp - phi_c) / (2.0 * dx)
    Fx_m = (eps2_c + eps2_xm) * (phi_c - phi_xm) / (2.0 * dx)
    Fy_p = (eps2_c + eps2_yp) * (phi_yp - phi_c) / (2.0 * dx)
    Fy_m = (eps2_c + eps2_ym) * (phi_c - phi_ym) / (2.0 * dx)
    Fz_p = (eps2_c + eps2_zp) * (phi_zp - phi_c) / (2.0 * dx)
    Fz_m = (eps2_c + eps2_zm) * (phi_c - phi_zm) / (2.0 * dx)

    return (Fx_p - Fx_m) / dx + (Fy_p - Fy_m) / dx + (Fz_p - Fz_m) / dx


@cuda.jit(device=True, inline=True)
def d2_phi_xyz(phi, gid, l, m, k, nx, ny, nz, dx):
    """Second-order finite-difference second derivatives in 3D."""
    lp = idx_xp(l, nx)
    lm = idx_xm(l, nx)
    mp = idx_yp(m, ny)
    mm = idx_ym(m, ny)
    kp = idx_zp(k, nz)
    km = idx_zm(k, nz)

    c = phi[gid, l, m, k]
    xp = phi[gid, lp, m, k]
    xm = phi[gid, lm, m, k]
    yp = phi[gid, l, mp, k]
    ym = phi[gid, l, mm, k]
    zp = phi[gid, l, m, kp]
    zm = phi[gid, l, m, km]

    inv_dx2 = 1.0 / (dx * dx)
    phixx = (xp - 2.0 * c + xm) * inv_dx2
    phiyy = (yp - 2.0 * c + ym) * inv_dx2
    phizz = (zp - 2.0 * c + zm) * inv_dx2

    xpy = phi[gid, lp, mp, k]
    xpm = phi[gid, lp, mm, k]
    xmy = phi[gid, lm, mp, k]
    xmm = phi[gid, lm, mm, k]
    phixy = (xpy - xpm - xmy + xmm) * (0.25 * inv_dx2)

    xpz = phi[gid, lp, m, kp]
    xpzm = phi[gid, lp, m, km]
    xmz = phi[gid, lm, m, kp]
    xmzm = phi[gid, lm, m, km]
    phixz = (xpz - xpzm - xmz + xmzm) * (0.25 * inv_dx2)

    ypz = phi[gid, l, mp, kp]
    ypzm = phi[gid, l, mp, km]
    ymz = phi[gid, l, mm, kp]
    ymzm = phi[gid, l, mm, km]
    phiyz = (ypz - ypzm - ymz + ymzm) * (0.25 * inv_dx2)

    return phixx, phiyy, phizz, phixy, phixz, phiyz


@cuda.jit(device=True, inline=True)
def facet_cos_and_nxyz_from_grad(phix, phiy, phiz, n111, solidid, g2_floor):
    """Return signed cos, abs cos, and best-aligned facet normal."""
    q = phix * phix + phiy * phiy + phiz * phiz
    if q < g2_floor:
        return (
            float32(0.0),
            float32(0.0),
            float32(1.0),
            float32(0.0),
            float32(0.0),
        )

    inv_g = 1.0 / math.sqrt(q)
    gx = phix * inv_g
    gy = phiy * inv_g
    gz = phiz * inv_g

    best_abs = 0.0
    best_signed = 0.0
    best_nx = 1.0
    best_ny = 0.0
    best_nz = 0.0

    for t in range(8):
        ux = n111[solidid, t, 0]
        uy = n111[solidid, t, 1]
        uz = n111[solidid, t, 2]
        dot = gx * ux + gy * uy + gz * uz
        ad = abs(dot)
        if ad > best_abs:
            best_abs = ad
            best_signed = dot
            best_nx = ux
            best_ny = uy
            best_nz = uz

    if best_abs > 1.0:
        best_abs = 1.0
    if best_signed > 1.0:
        best_signed = 1.0
    if best_signed < -1.0:
        best_signed = -1.0

    return (
        float32(best_signed),
        float32(best_abs),
        float32(best_nx),
        float32(best_ny),
        float32(best_nz),
    )


@cuda.jit(device=True, inline=True)
def da_dphixyz_A12(phi, l, m, k, nx, ny, nz, dx, solidid,
                   a0, delta_a, mu_a, p_round, n111, g2_floor):
    """Return (a, da/dphi_x, da/dphi_y, da/dphi_z)."""
    phix, phiy, phiz = grad_phi_xyz(phi, LIQ, l, m, k, nx, ny, nz, dx)
    q = phix * phix + phiy * phiy + phiz * phiz
    if q < g2_floor:
        return float32(0.0), float32(0.0), float32(0.0), float32(0.0)

    cos_signed, cos_abs, nxn, nyn, nzn = facet_cos_and_nxyz_from_grad(
        phix, phiy, phiz, n111, solidid, g2_floor
    )

    a_val = calc_a_from_cos(cos_abs, a0, delta_a, mu_a, p_round)

    inv_g = 1.0 / math.sqrt(q)
    inv_g3 = 1.0 / (q * math.sqrt(q))
    udotg = phix * nxn + phiy * nyn + phiz * nzn

    dc_dphix = nxn * inv_g - udotg * phix * inv_g3
    dc_dphiy = nyn * inv_g - udotg * phiy * inv_g3
    dc_dphiz = nzn * inv_g - udotg * phiz * inv_g3

    c = cos_signed
    C = math.sqrt(c * c + p_round * p_round)
    S = math.sqrt(max(1.0 - c * c, 0.0) + p_round * p_round)
    coef = (c / C) - math.tan(a0) * (c / S)

    da_dphix = mu_a * delta_a * coef * dc_dphix
    da_dphiy = mu_a * delta_a * coef * dc_dphiy
    da_dphiz = mu_a * delta_a * coef * dc_dphiz

    return (
        float32(a_val),
        float32(da_dphix),
        float32(da_dphiy),
        float32(da_dphiz),
    )


@cuda.jit(device=True, inline=True)
def A13_divergence_3d(phi, l, m, k, nx, ny, nz, dx, solidid,
                      a0, delta_a, mu_a, p_round, n111, g2_floor):
    """Central-difference approximation of d/dp(da/dphi_p) for p=x,y,z."""
    lp = idx_xp(l, nx)
    lm = idx_xm(l, nx)
    mp = idx_yp(m, ny)
    mm = idx_ym(m, ny)
    kp = idx_zp(k, nz)
    km = idx_zm(k, nz)

    _, da_dphix_lp, _, _ = da_dphixyz_A12(
        phi, lp, m, k, nx, ny, nz, dx, solidid,
        a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    _, da_dphix_lm, _, _ = da_dphixyz_A12(
        phi, lm, m, k, nx, ny, nz, dx, solidid,
        a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    _, _, da_dphiy_mp, _ = da_dphixyz_A12(
        phi, l, mp, k, nx, ny, nz, dx, solidid,
        a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    _, _, da_dphiy_mm, _ = da_dphixyz_A12(
        phi, l, mm, k, nx, ny, nz, dx, solidid,
        a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    _, _, _, da_dphiz_kp = da_dphixyz_A12(
        phi, l, m, kp, nx, ny, nz, dx, solidid,
        a0, delta_a, mu_a, p_round, n111, g2_floor,
    )
    _, _, _, da_dphiz_km = da_dphixyz_A12(
        phi, l, m, km, nx, ny, nz, dx, solidid,
        a0, delta_a, mu_a, p_round, n111, g2_floor,
    )

    d_dx = (da_dphix_lp - da_dphix_lm) * (0.5 / dx)
    d_dy = (da_dphiy_mp - da_dphiy_mm) * (0.5 / dx)
    d_dz = (da_dphiz_kp - da_dphiz_km) * (0.5 / dx)
    return float32(d_dx), float32(d_dy), float32(d_dz)


@cuda.jit(device=True, inline=True)
def torque_A11_3d(phi, l, m, k, nx, ny, nz, dx, solidid, eps0_sl,
                  a0, delta_a, mu_a, p_round, n111, g2_floor):
    """Full 3D torque term (eps0^2)(Ex + Ey + Ez)."""
    phix, phiy, phiz = grad_phi_xyz(phi, LIQ, l, m, k, nx, ny, nz, dx)
    q = phix * phix + phiy * phiy + phiz * phiz
    if q < g2_floor:
        return 0.0

    phixx, phiyy, phizz, phixy, phixz, phiyz = d2_phi_xyz(
        phi, LIQ, l, m, k, nx, ny, nz, dx
    )
    q_x = 2.0 * (phix * phixx + phiy * phixy + phiz * phixz)
    q_y = 2.0 * (phix * phixy + phiy * phiyy + phiz * phiyz)
    q_z = 2.0 * (phix * phixz + phiy * phiyz + phiz * phizz)

    a_val, da_dphix, da_dphiy, da_dphiz = da_dphixyz_A12(
        phi, l, m, k, nx, ny, nz, dx, solidid,
        a0, delta_a, mu_a, p_round, n111, g2_floor,
    )

    cos_signed, _, nxn, nyn, nzn = facet_cos_and_nxyz_from_grad(
        phix, phiy, phiz, n111, solidid, g2_floor
    )

    inv_g = 1.0 / math.sqrt(q)
    inv_g3 = 1.0 / (q * math.sqrt(q))
    udotg = phix * nxn + phiy * nyn + phiz * nzn

    dc_dphix = nxn * inv_g - udotg * phix * inv_g3
    dc_dphiy = nyn * inv_g - udotg * phiy * inv_g3
    dc_dphiz = nzn * inv_g - udotg * phiz * inv_g3

    c_x = dc_dphix * phixx + dc_dphiy * phixy + dc_dphiz * phixz
    c_y = dc_dphix * phixy + dc_dphiy * phiyy + dc_dphiz * phiyz
    c_z = dc_dphix * phixz + dc_dphiy * phiyz + dc_dphiz * phizz

    c = cos_signed
    C = math.sqrt(c * c + p_round * p_round)
    S = math.sqrt(max(1.0 - c * c, 0.0) + p_round * p_round)
    coef = (c / C) - math.tan(a0) * (c / S)

    da_dx = mu_a * delta_a * coef * c_x
    da_dy = mu_a * delta_a * coef * c_y
    da_dz = mu_a * delta_a * coef * c_z

    d_dx_da_dphix, d_dy_da_dphiy, d_dz_da_dphiz = A13_divergence_3d(
        phi, l, m, k, nx, ny, nz, dx, solidid,
        a0, delta_a, mu_a, p_round, n111, g2_floor,
    )

    Ex = (da_dx * da_dphix * q) + (a_val * d_dx_da_dphix * q) + (a_val * da_dphix * q_x)
    Ey = (da_dy * da_dphiy * q) + (a_val * d_dy_da_dphiy * q) + (a_val * da_dphiy * q_y)
    Ez = (da_dz * da_dphiz * q) + (a_val * d_dz_da_dphiz * q) + (a_val * da_dphiz * q_z)

    return (eps0_sl * eps0_sl) * (Ex + Ey + Ez)


@cuda.jit(device=True, inline=True)
def laplacian_3d(phi, gid, l, m, k, nx, ny, nz, inv_dx2):
    """6-neighbour Laplacian at (l,m,k)."""
    lp = idx_xp(l, nx)
    lm = idx_xm(l, nx)
    mp = idx_yp(m, ny)
    mm = idx_ym(m, ny)
    kp = idx_zp(k, nz)
    km = idx_zm(k, nz)

    return (
        phi[gid, lp, m, k] + phi[gid, lm, m, k]
        + phi[gid, l, mp, k] + phi[gid, l, mm, k]
        + phi[gid, l, m, kp] + phi[gid, l, m, km]
        - 6.0 * phi[gid, l, m, k]
    ) * inv_dx2


@cuda.jit
def kernel_update_nfmf_3d(phi, mf, nf, nx, ny, nz, number_of_grain):
    """Update active phase list (APT) per cell in 3D."""
    l, m, k = cuda.grid(3)
    if l >= nx or m >= ny or k >= nz:
        return

    lp = idx_xp(l, nx)
    lm = idx_xm(l, nx)
    mp = idx_yp(m, ny)
    mm = idx_ym(m, ny)
    kp = idx_zp(k, nz)
    km = idx_zm(k, nz)

    n = 0
    for i in range(number_of_grain):
        if (phi[i, l, m, k] > 0.0) or ((phi[i, l, m, k] == 0.0) and (
            (phi[i, lp, m, k] > 0.0) or (phi[i, lm, m, k] > 0.0) or
            (phi[i, l, mp, k] > 0.0) or (phi[i, l, mm, k] > 0.0) or
            (phi[i, l, m, kp] > 0.0) or (phi[i, l, m, km] > 0.0)
        )):
            n += 1
            mf[n - 1, l, m, k] = i

    nf[l, m, k] = n


@cuda.jit
def kernel_update_nfmf_3d_checked(phi, mf, nf, status, nx, ny, nz, number_of_grain):
    """Checked APT update that clamps writes and reports KMAX overflow.

    Parameters
    ----------
    status : int32[2]
        status[0] = 1 if any cell needs more than KMAX active phases
        status[1] = maximum active-phase count observed over the whole grid
    """
    l, m, k = cuda.grid(3)
    if l >= nx or m >= ny or k >= nz:
        return

    lp = idx_xp(l, nx)
    lm = idx_xm(l, nx)
    mp = idx_yp(m, ny)
    mm = idx_ym(m, ny)
    kp = idx_zp(k, nz)
    km = idx_zm(k, nz)

    n = 0
    for i in range(number_of_grain):
        if (phi[i, l, m, k] > 0.0) or ((phi[i, l, m, k] == 0.0) and (
            (phi[i, lp, m, k] > 0.0) or (phi[i, lm, m, k] > 0.0) or
            (phi[i, l, mp, k] > 0.0) or (phi[i, l, mm, k] > 0.0) or
            (phi[i, l, m, kp] > 0.0) or (phi[i, l, m, km] > 0.0)
        )):
            n += 1
            if n <= KMAX:
                mf[n - 1, l, m, k] = i

    if n > KMAX:
        cuda.atomic.max(status, 0, 1)
        nf[l, m, k] = KMAX
    else:
        nf[l, m, k] = n

    cuda.atomic.max(status, 1, n)


@cuda.jit
def kernel_update_phasefield_active_3d(phi, phi_new, temp, mf, nf,
                                       wij, aij, mij, n111,
                                       nx, ny, nz, number_of_grain,
                                       dx, dt, T_melt, Sf,
                                       eps0_sl, w0_sl,
                                       a0, delta_a, mu_a, p_round,
                                       g2_floor, ksi, theta_c_rad):
    """Main 3D phase-field update kernel with anisotropic SL interface."""
    l, m, k = cuda.grid(3)
    if l >= nx or m >= ny or k >= nz:
        return

    lp = idx_xp(l, nx)
    lm = idx_xm(l, nx)
    mp = idx_yp(m, ny)
    mm = idx_ym(m, ny)
    kp = idx_zp(k, nz)
    km = idx_zm(k, nz)
    inv_dx2 = 1.0 / (dx * dx)
    Tcur = temp[l, m, k]
    n_act = nf[l, m, k]

    lap_sl = cuda.local.array(KMAX, float32)
    b_sl = cuda.local.array(KMAX, float32)
    w_sl_arr = cuda.local.array(KMAX, float32)

    gx_liq, gy_liq, gz_liq = grad_phi_xyz(phi, LIQ, l, m, k, nx, ny, nz, dx)

    for t in range(n_act):
        gid = mf[t, l, m, k]
        if gid == LIQ:
            b_sl[t] = 1.0
            lap_sl[t] = 0.0
        else:
            lap_val = aniso_term1_solid_3d(
                phi, l, m, k, nx, ny, nz, dx, gid, eps0_sl,
                a0, delta_a, mu_a, p_round, n111, g2_floor,
            )
            lap_val += torque_A11_3d(
                phi, l, m, k, nx, ny, nz, dx, gid, eps0_sl,
                a0, delta_a, mu_a, p_round, n111, g2_floor,
            )
            lap_sl[t] = lap_val

            best_cos = best_cos_from_grad_3d(
                gx_liq, gy_liq, gz_liq, n111, gid, g2_floor
            )
            b_sl[t] = calc_b_from_cos(best_cos, ksi, theta_c_rad)

    for t in range(n_act):
        gid = mf[t, l, m, k]
        if gid == LIQ:
            w_sl_arr[t] = w0_sl
        else:
            phix_s, phiy_s, phiz_s = grad_phi_xyz(
                phi, gid, l, m, k, nx, ny, nz, dx
            )
            cmax = best_cos_from_grad_3d(
                phix_s, phiy_s, phiz_s, n111, gid, g2_floor
            )
            a_loc = calc_a_from_cos(cmax, a0, delta_a, mu_a, p_round)
            w_sl_arr[t] = w0_sl * (a_loc * a_loc)

    for phase_id in range(number_of_grain):
        phi_new[phase_id, l, m, k] = 0.0

    for t1 in range(n_act):
        i = mf[t1, l, m, k]
        dpi = 0.0

        for t2 in range(n_act):
            j = mf[t2, l, m, k]
            if i == j:
                continue

            driving_force = 0.0
            if i != LIQ and j == LIQ:
                driving_force = -Sf * (Tcur - T_melt)
            elif i == LIQ and j != LIQ:
                driving_force = Sf * (Tcur - T_melt)

            ppp = 0.0
            for t3 in range(n_act):
                phase_k = mf[t3, l, m, k]
                phi_k = phi[phase_k, l, m, k]

                lap_k = (
                    phi[phase_k, lp, m, k] + phi[phase_k, lm, m, k]
                    + phi[phase_k, l, mp, k] + phi[phase_k, l, mm, k]
                    + phi[phase_k, l, m, kp] + phi[phase_k, l, m, km]
                    - 6.0 * phi_k
                ) * inv_dx2

                wik = wij[i, phase_k]
                wjk = wij[j, phase_k]
                if phase_k == LIQ:
                    if i != LIQ:
                        wik = w_sl_arr[t1]
                    if j != LIQ:
                        wjk = w_sl_arr[t2]
                term1 = (wik - wjk) * phi_k

                term2 = 0.0
                if j == LIQ and i != LIQ:
                    if phase_k == i:
                        term2 = 0.5 * lap_sl[t1]
                elif i == LIQ and j != LIQ:
                    if phase_k == j:
                        term2 = -0.5 * lap_sl[t2]
                else:
                    eps2ik = aij[i, phase_k] * aij[i, phase_k]
                    eps2jk = aij[j, phase_k] * aij[j, phase_k]
                    term2 = 0.5 * (eps2ik - eps2jk) * lap_k

                ppp += term1 + term2

            phii_phij = phi[i, l, m, k] * phi[j, l, m, k]
            term_force = (8.0 / 3.1415926535) * math.sqrt(max(phii_phij, 0.0)) * driving_force

            mij_eff = mij[i, j]
            if i != LIQ and j == LIQ:
                mij_eff = mij[i, j] * b_sl[t1]
            elif i == LIQ and j != LIQ:
                mij_eff = mij[i, j] * b_sl[t2]

            dpi -= 2.0 * mij_eff / float(n_act) * (ppp - term_force)

        phi_new[i, l, m, k] = phi[i, l, m, k] + dt * dpi

    s = 0.0
    for t in range(n_act):
        i = mf[t, l, m, k]
        v = phi_new[i, l, m, k]
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        phi_new[i, l, m, k] = v
        s += v

    if s > 1e-20:
        invs = 1.0 / s
        for t in range(n_act):
            i = mf[t, l, m, k]
            phi_new[i, l, m, k] *= invs
    else:
        phi_new[LIQ, l, m, k] = 1.0


@cuda.jit
def kernel_update_phasefield_active_3d_switchable(
    phi, phi_new, temp, mf, nf,
    wij, aij, mij, n111,
    nx, ny, nz, number_of_grain,
    dx, dt, T_melt, Sf,
    eps0_sl, w0_sl,
    a0, delta_a, mu_a, p_round,
    g2_floor, ksi, theta_c_rad,
    enable_anisotropy, enable_torque,
):
    """3D phase-field update kernel with runtime toggles for verification."""
    l, m, k = cuda.grid(3)
    if l >= nx or m >= ny or k >= nz:
        return

    inv_dx2 = 1.0 / (dx * dx)
    Tcur = temp[l, m, k]
    n_act = nf[l, m, k]

    lap_sl = cuda.local.array(KMAX, float32)
    b_sl = cuda.local.array(KMAX, float32)
    w_sl_arr = cuda.local.array(KMAX, float32)

    gx_liq, gy_liq, gz_liq = grad_phi_xyz(phi, LIQ, l, m, k, nx, ny, nz, dx)
    lap_liq = laplacian_3d(phi, LIQ, l, m, k, nx, ny, nz, inv_dx2)

    for t in range(n_act):
        gid = mf[t, l, m, k]
        if gid == LIQ:
            b_sl[t] = 1.0
            lap_sl[t] = 0.0
        else:
            if enable_anisotropy != 0:
                lap_val = aniso_term1_solid_3d(
                    phi, l, m, k, nx, ny, nz, dx, gid, eps0_sl,
                    a0, delta_a, mu_a, p_round, n111, g2_floor,
                )
                if enable_torque != 0:
                    lap_val += torque_A11_3d(
                        phi, l, m, k, nx, ny, nz, dx, gid, eps0_sl,
                        a0, delta_a, mu_a, p_round, n111, g2_floor,
                    )
                best_cos = best_cos_from_grad_3d(
                    gx_liq, gy_liq, gz_liq, n111, gid, g2_floor
                )
                b_sl[t] = calc_b_from_cos(best_cos, ksi, theta_c_rad)
            else:
                lap_val = (eps0_sl * eps0_sl) * lap_liq
                b_sl[t] = 1.0

            lap_sl[t] = lap_val

    for t in range(n_act):
        gid = mf[t, l, m, k]
        if gid == LIQ:
            w_sl_arr[t] = w0_sl
        else:
            if enable_anisotropy != 0:
                phix_s, phiy_s, phiz_s = grad_phi_xyz(
                    phi, gid, l, m, k, nx, ny, nz, dx
                )
                cmax = best_cos_from_grad_3d(
                    phix_s, phiy_s, phiz_s, n111, gid, g2_floor
                )
                a_loc = calc_a_from_cos(cmax, a0, delta_a, mu_a, p_round)
                w_sl_arr[t] = w0_sl * (a_loc * a_loc)
            else:
                w_sl_arr[t] = w0_sl

    for phase_id in range(number_of_grain):
        phi_new[phase_id, l, m, k] = 0.0

    for t1 in range(n_act):
        i = mf[t1, l, m, k]
        dpi = 0.0

        for t2 in range(n_act):
            j = mf[t2, l, m, k]
            if i == j:
                continue

            driving_force = 0.0
            if i != LIQ and j == LIQ:
                driving_force = -Sf * (Tcur - T_melt)
            elif i == LIQ and j != LIQ:
                driving_force = Sf * (Tcur - T_melt)

            ppp = 0.0
            for t3 in range(n_act):
                phase_k = mf[t3, l, m, k]
                phi_k = phi[phase_k, l, m, k]
                lap_k = laplacian_3d(phi, phase_k, l, m, k, nx, ny, nz, inv_dx2)

                wik = wij[i, phase_k]
                wjk = wij[j, phase_k]
                if phase_k == LIQ:
                    if i != LIQ:
                        wik = w_sl_arr[t1]
                    if j != LIQ:
                        wjk = w_sl_arr[t2]
                term1 = (wik - wjk) * phi_k

                term2 = 0.0
                if j == LIQ and i != LIQ:
                    if phase_k == i:
                        term2 = 0.5 * lap_sl[t1]
                elif i == LIQ and j != LIQ:
                    if phase_k == j:
                        term2 = -0.5 * lap_sl[t2]
                else:
                    eps2ik = aij[i, phase_k] * aij[i, phase_k]
                    eps2jk = aij[j, phase_k] * aij[j, phase_k]
                    term2 = 0.5 * (eps2ik - eps2jk) * lap_k

                ppp += term1 + term2

            phii_phij = phi[i, l, m, k] * phi[j, l, m, k]
            term_force = (8.0 / 3.1415926535) * math.sqrt(max(phii_phij, 0.0)) * driving_force

            mij_eff = mij[i, j]
            if i != LIQ and j == LIQ:
                mij_eff = mij[i, j] * b_sl[t1]
            elif i == LIQ and j != LIQ:
                mij_eff = mij[i, j] * b_sl[t2]

            dpi -= 2.0 * mij_eff / float(n_act) * (ppp - term_force)

        phi_new[i, l, m, k] = phi[i, l, m, k] + dt * dpi

    s = 0.0
    for t in range(n_act):
        i = mf[t, l, m, k]
        v = phi_new[i, l, m, k]
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        phi_new[i, l, m, k] = v
        s += v

    if s > 1e-20:
        invs = 1.0 / s
        for t in range(n_act):
            i = mf[t, l, m, k]
            phi_new[i, l, m, k] *= invs
    else:
        phi_new[LIQ, l, m, k] = 1.0


@cuda.jit
def kernel_update_temp_3d(temp, cooling_rate, nx, ny, nz):
    """Uniform temperature decrease in 3D."""
    l, m, k = cuda.grid(3)
    if l < nx and m < ny and k < nz:
        temp[l, m, k] -= cooling_rate
