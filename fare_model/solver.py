"""The staged FARE time-integration operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import fft, fftfreq

from .helper import dea_fftx, dea_ifftx, thomas_factor, thomas_solve, thomas_solve_PPE, zinterp

if TYPE_CHECKING:
    from .model import FARE


#========================================================================================================================================================================================
#              MAIN SOLVER                  MAIN SOLVER                 MAIN SOLVER                   MAIN SOLVER                   MAIN SOLVER                   MAIN SOLVER
#========================================================================================================================================================================================


# ================================================   SET UP   ===========================================================

#Precalculate general parameters:
def _prepare_static_fields(self: "FARE") -> None:
    """Build grid coordinates, static profiles, masks, and factored linear operators for the active configuration."""

    if self._static_ready:
        return
    self._require_configuration()
    numerics = self.numerics
    scenario = self.scenario
    param = self.param

    #unpack 2d parameters
    Lx, Lz = numerics.L
    I, J = numerics.I, numerics.J
    dx = Lx / (I - 1)
    dz = Lz / (J - 1)
    kx = 2 * np.pi * fftfreq(I - 1, d=dx)
    x, z = np.meshgrid(np.linspace(0, Lx - dx, I - 1), np.linspace(-dz / 2, Lz + 1.5 * dz, J + 2), indexing="ij")
    zi, zf = 1, J + 1

    #if vis_art and gam_art are set to None (i.e. default), we initialize them to values based on dx, dt, and dz
    vis_art = numerics.vis_art if numerics.vis_art is not None else 0.01 * dz**2 / numerics.dt
    gam_art = numerics.gam_art if numerics.gam_art is not None else vis_art * dx**4 / (np.pi * dz) ** 2

    #Advection-Diffusion Parameters:
    Sz = vis_art * numerics.dt / dz**2
    ubg = fft(self._field(scenario.u_relax, x, z), axis=0)
    ubg_ph = dea_ifftx(ubg, I)
    u_surface = np.mean(ubg[:, 0] + ubg[:, 1])
    u_bc = -1 if -0.2 <= u_surface <= 0.2 else 1
    potil = fft(self._field(scenario.pot0, x, z), axis=0)
    qvtil = fft(self._field(scenario.qvi, x, z), axis=0)
    DpotilCS = np.zeros((I - 1, J + 1), dtype=complex)
    DqvtilCS = (qvtil[:, zi : zf + 1] - qvtil[:, zi - 1 : zf]) / dz

    #Other useful stuff
    f0 = self._field(scenario.f0, x, z)
    qvs = fft(param.qvso / f0**param.CpRd * np.exp(-param.LL / param.Rv * (1 / (f0 * self._field(scenario.pot0, x, z)) - 1 / param.theta0)), axis=0)
    qvs_ph = dea_ifftx(qvs, I)
    potil_ph = dea_ifftx(potil, I)
    qvtil_ph = dea_ifftx(qvtil, I)

    #rho0 = p0/(Rd*pot0(x,z))*f0(x,z)**(CpRd-1) 
    # Boussinesq assumptions require constant rho0; we use the surface value
    rho0 = param.p0 / (param.Rd * param.theta0) * np.ones_like(x)

    #interpolated physical background rainy potential temperature and background + saturated vapor specific humidity (needed for buoyancy)
    # Calculate large-scale background forcing profiles in spectral space
    moistenprof = fft(self._field(scenario.moistening, x, z), axis=0) / 2
    coolingprof = fft(self._field(scenario.cooling, x, z), axis=0) / 2

    #Calculate spongelayer (hereby referred to as boundmask)
    boundmask = fft(numerics.dt / scenario.tauS * self._field(scenario.spongelayer, x, z), axis=0)

    #Calculate mountain mask (hereby referred to as mountmask)
    mountmask = fft(numerics.dt / scenario.tauM * self._field(scenario.terrainmask, x, z), axis=0)
    boundWmask = fft(numerics.dt / scenario.tauS * self._field(scenario.spongelayer, x, z + dz / 2)[:, :-1], axis=0)
    mountWmask = fft(numerics.dt / scenario.tauM * self._field(scenario.terrainmask, x, z + dz / 2)[:, :-1], axis=0)

    # ----------------------------------------------  implicit tridiagonal matrix callers  --------------------------------------------------
    # dirichlet BC matrix
    apy_diff_dir, bpy_diff_dir, cpy_diff_dir = thomas_factor(-Sz / 2 * np.ones(zf - zi - 2), (Sz + 1) * np.ones(zf - zi - 1), -Sz / 2 * np.ones(zf - zi - 2))

    # neumann BC matrix
    a_neu = -Sz * np.ones(zf - zi + 1)
    a_neu[-1] = -1
    b_neu = np.concatenate(([-1], 2 * (Sz + 1) * np.ones(zf - zi), [1]))
    c_neu = -Sz * np.ones(zf - zi + 1)
    c_neu[0] = 1
    apy_diff_neu, bpy_diff_neu, cpy_diff_neu = thomas_factor(a_neu, b_neu, c_neu)

    # combination (dir/neu) BC matrix for u if necessary
    a_com = -Sz / 2 * np.ones(zf - zi + 1)
    a_com[-1] = -1
    b_com = np.concatenate(([1], (Sz + 1) * np.ones(zf - zi), [1]))
    c_com = -Sz / 2 * np.ones(zf - zi + 1)
    c_com[0] = -1
    apy_diff_com, bpy_diff_com, cpy_diff_com = thomas_factor(a_com, b_com, c_com)

    #parameters for 2nd-order PPE
    apz_PPE = np.zeros((len(kx), zf - zi + 1))
    bpz_PPE = np.zeros((len(kx), zf - zi + 2))
    cpz_PPE = np.zeros((len(kx), zf - zi + 1))
    for mode in range(len(kx)):
        apz_PPE[mode], bpz_PPE[mode], cpz_PPE[mode] = thomas_factor(np.ones(zf - zi + 1), np.concatenate(([-1], (-2 - kx[mode] ** 2 * dz**2) * np.ones(zf - zi), [-1])), np.ones(zf - zi + 1))
    self._cache = {
        "dx": dx, "dz": dz, "dZ": 2 * dz, "dz2": dz**2, "kx": kx, "zi": zi, "zf": zf, "Sz": Sz,
        "xdamp": np.exp(-gam_art * kx**4 * numerics.dt)[:, None], "g0": param.g / param.theta0, "x": x, "z": z,
        "ubg": ubg, "ubg_ph": ubg_ph, "u_bc": u_bc, "DpotilCS": DpotilCS, "DqvtilCS": DqvtilCS,
        "potil_int": zinterp(potil_ph), "potil_ph": potil_ph, "qvtil_int": zinterp(qvtil_ph), "qvtil_ph": qvtil_ph,
        "qvs": qvs, "qvs_int": zinterp(qvs_ph), "qvs_ph": qvs_ph, "rho0": rho0, "rho0_int": zinterp(rho0),
        "moistenprof": moistenprof, "coolingprof": coolingprof, "boundmask_ph": dea_ifftx(boundmask, I),
        "mountmask_ph": dea_ifftx(mountmask, I), "boundWmask_ph": dea_ifftx(boundWmask, I), "mountWmask_ph": dea_ifftx(mountWmask, I),
        "apy_diff_dir": apy_diff_dir, "bpy_diff_dir": bpy_diff_dir, "cpy_diff_dir": cpy_diff_dir,
        "apy_diff_neu": apy_diff_neu, "bpy_diff_neu": bpy_diff_neu, "cpy_diff_neu": cpy_diff_neu,
        "apy_diff_com": apy_diff_com, "bpy_diff_com": bpy_diff_com, "cpy_diff_com": cpy_diff_com,
        "apz_PPE": apz_PPE, "bpz_PPE": bpz_PPE, "cpz_PPE": cpz_PPE,
    }
    self._static_ready = True



# ========================================== NONLINEAR TERMS + EXPLICIT TIME-STEPPING =====================================================

def explicit_terms(self: "FARE", scheme: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute nonlinear tendencies and explicit updates for the selected EF, LF, or AB3 step."""

    state = self.state
    cache = self._cache
    numerics = self.numerics
    scenario = self.scenario
    param = self.param
    zi, zf = cache["zi"], cache["zf"]
    kx, dZ, dz, Sz = cache["kx"], cache["dZ"], cache["dz"], cache["Sz"]

    # flux form of advection; will cut off vertical by 2
    def fluz(arr_int: np.ndarray, w: np.ndarray) -> np.ndarray:
        temps = arr_int * w
        return (temps[:, 1:] - temps[:, :-1]) / dz

    #skew symmetric form of advection specifically for w (which is at its own node, and thus uses centered diff dZ)
    def sksym_w(w: np.ndarray) -> np.ndarray:
        w_sq = w**2
        return 0.5 * (w[:, 1:-1] * (w[:, 2:] - w[:, :-2]) / dZ + (w_sq[:, 2:] - w_sq[:, :-2]) / dZ)

    #skew symmetric form of advection specifically for u (which is at its own node, and thus uses centered diff dZ)
    def sksym_u(u: np.ndarray, w: np.ndarray) -> np.ndarray:
        u_int = zinterp(u)
        w_int = zinterp(w)
        return 0.5 * (fluz(u_int, w) + w_int * (u[:, 2:] - u[:, :-2]) / dZ)

    # --------------------------------------------------- CALCULATE ALL NONLINEAR TERMS --------------------------------------------------

    #  IFFT all necessary terms to physical space with 3/2-dealiasing (basically all terms related to nonlinear advection and/or piecewise functions)
    u_ph = dea_ifftx(state.u[0], numerics.I)
    pot_ph = dea_ifftx(state.pot[0], numerics.I)
    w_ph = dea_ifftx(state.w[0, :, :-1], numerics.I)
    qt_ph = dea_ifftx(state.qt[0], numerics.I)
    dudx_ph = dea_ifftx(1j * kx[:, None] * state.u[0], numerics.I)
    dpotdx_ph = dea_ifftx(1j * kx[:, None] * state.pot[0], numerics.I)
    dqtdx_ph = dea_ifftx(1j * kx[:, None] * state.qt[0], numerics.I)
    dwdx_ph = dea_ifftx(1j * kx[:, None] * state.w[0, :, :-1], numerics.I)
    pot_int = zinterp(pot_ph)
    qt_int = zinterp(qt_ph)
    u_int = zinterp(u_ph[:, 1:-1])

    # calculate nonlinear advection in physical space and then fft/de-pad back to fourier space (multiplication in physical space is far cheaper than convolution in spectral)
    f_u = 0.5 * 1j * kx[:, None] * dea_fftx(u_ph[:, 1:-1] ** 2, numerics.I) + dea_fftx(0.5 * (u_ph[:, 1:-1] * dudx_ph[:, 1:-1]) + sksym_u(u_ph, w_ph), numerics.I)
    f_pot = 1j * kx[:, None] * dea_fftx(u_ph[:, 1:-1] * pot_ph[:, 1:-1], numerics.I) + dea_fftx(fluz(pot_int, w_ph), numerics.I)
    f_qt = 1j * kx[:, None] * dea_fftx(u_ph[:, 1:-1] * qt_ph[:, 1:-1], numerics.I) + dea_fftx(fluz(qt_int, w_ph), numerics.I)
    f_w = 0.5 * 1j * kx[:, None] * dea_fftx(u_int * w_ph[:, 1:-1], numerics.I) + dea_fftx(0.5 * (u_int * dwdx_ph[:, 1:-1]) + sksym_w(w_ph), numerics.I)

    # calculate buoyancy/water piecewise terms and then fft/de-pad back to foureir space (these must also be calculated in physical space since piecewise is nonlinear)
    potterm = cache["g0"] * (pot_int - cache["potil_int"])
    b_ph = potterm + param.g * np.where(qt_int < cache["qvs_int"], param.epsilon0 * (qt_int - cache["qvtil_int"]), (param.LCp / param.theta0 - 1) * (qt_int - cache["qvs_int"]) + param.epsilon0 * (cache["qvs_int"] - cache["qvtil_int"]))
    b = dea_fftx(b_ph, numerics.I)
    qr_ph = np.maximum(qt_ph - cache["qvs_ph"], 0)
    state.qr = dea_fftx(qr_ph, numerics.I)
    state.qr[:, 0], state.qr[:, numerics.J + 1] = state.qr[:, 1], state.qr[:, numerics.J]
    dqrdz = (state.qr[:, zi + 1 : zf + 1] - state.qr[:, zi - 1 : zf - 1]) / dZ

    # ------------------------------------------------ ALL EXPLICIT TIME STEPPING ---------------------------------------------------------------------

    # gather all forward terms, including the forcing/relaxation terms
    state.uRHS[0] = numerics.dt * (-f_u - 1 / scenario.tauu * (np.mean(state.u[0, :, 1:-1], axis=0) - cache["ubg"][:, 1:-1]))
    state.wRHS[0] = numerics.dt * (-f_w + b[:, 1:-1])
    state.qtRHS[0] = numerics.dt * (-f_qt + scenario.Vt * dqrdz + cache["moistenprof"][:, zi:zf])
    state.potRHS[0] = numerics.dt * (-f_pot - param.LCp * scenario.Vt * dqrdz + cache["coolingprof"][:, zi:zf])
    u_st = np.zeros((numerics.I - 1, numerics.J + 2), dtype=complex)
    w_st = np.zeros((numerics.I - 1, numerics.J + 2), dtype=complex)

    # apply explicit time schemes
    if scheme == "EF":
        state.pot[1, :, zi:zf] = state.pot[0, :, zi:zf] + state.potRHS[0]
        state.qt[1, :, zi:zf] = state.qt[0, :, zi:zf] + state.qtRHS[0]
        u_st[:, zi:zf] = state.u[0, :, zi:zf] + state.uRHS[0] + Sz / 2 * (state.u[0, :, zi + 1 : zf + 1] - 2 * state.u[0, :, zi:zf] + state.u[0, :, zi - 1 : zf - 1])
        w_st[:, zi : zf - 1] = state.w[0, :, zi : zf - 1] + state.wRHS[0] + Sz / 2 * (state.w[0, :, zi + 1 : zf] - 2 * state.w[0, :, zi : zf - 1] + state.w[0, :, zi - 1 : zf - 2])
    elif scheme == "LF":
        state.pot[1, :, zi:zf] = state.pot[-1, :, zi:zf] + 2 * state.potRHS[0]
        state.qt[1, :, zi:zf] = state.qt[-1, :, zi:zf] + 2 * state.qtRHS[0]
        u_st[:, zi:zf] = state.u[-1, :, zi:zf] + 2 * state.uRHS[0] + Sz * (state.u[0, :, zi + 1 : zf + 1] - 2 * state.u[0, :, zi:zf] + state.u[0, :, zi - 1 : zf - 1])
        w_st[:, zi : zf - 1] = state.w[-1, :, zi : zf - 1] + 2 * state.wRHS[0] + Sz * (state.w[0, :, zi + 1 : zf] - 2 * state.w[0, :, zi : zf - 1] + state.w[0, :, zi - 1 : zf - 2])
    else:
        a_AB3, b_AB3, c_AB3 = 23 / 12, -16 / 12, 5 / 12
        state.pot[1, :, zi:zf] = state.pot[0, :, zi:zf] + a_AB3 * state.potRHS[0] + b_AB3 * state.potRHS[-1] + c_AB3 * state.potRHS[-2]
        state.qt[1, :, zi:zf] = state.qt[0, :, zi:zf] + a_AB3 * state.qtRHS[0] + b_AB3 * state.qtRHS[-1] + c_AB3 * state.qtRHS[-2]
        u_st[:, zi:zf] = state.u[0, :, zi:zf] + a_AB3 * state.uRHS[0] + b_AB3 * state.uRHS[-1] + c_AB3 * state.uRHS[-2] + Sz / 2 * (state.u[0, :, zi + 1 : zf + 1] - 2 * state.u[0, :, zi:zf] + state.u[0, :, zi - 1 : zf - 1])
        w_st[:, zi : zf - 1] = state.w[0, :, zi : zf - 1] + a_AB3 * state.wRHS[0] + b_AB3 * state.wRHS[-1] + c_AB3 * state.wRHS[-2] + Sz / 2 * (state.w[0, :, zi + 1 : zf] - 2 * state.w[0, :, zi : zf - 1] + state.w[0, :, zi - 1 : zf - 2])

    # apply Euler Forward pressure gradient force to intermediate velocities (note that we can't use leapfrog or AB3 here because it would leak pressure from previous time steps)
    u_st[:, zi:zf] += -numerics.dt * (1 / cache["rho0"][:, zi:zf]) * 1j * kx[:, None] * state.P[0, :, zi:zf]
    w_st[:, zi : zf - 1] += -numerics.dt * (1 / cache["rho0_int"][:, zi : zf - 1]) * (state.P[0, :, zi + 1 : zf] - state.P[0, :, zi : zf - 1]) / dz

    #boundary conditions
    state.pot[1, :, 0], state.pot[1, :, numerics.J + 1] = state.pot[1, :, 1] - cache["DpotilCS"][:, 0] * dz, state.pot[1, :, numerics.J] + cache["DpotilCS"][:, -1] * dz
    state.qt[1, :, 0], state.qt[1, :, numerics.J + 1] = state.qt[1, :, 1] - cache["DqvtilCS"][:, 0] * dz, state.qt[1, :, numerics.J] + cache["DqvtilCS"][:, -1] * dz
    u_st[:, 0], u_st[:, numerics.J + 1] = cache["u_bc"] * u_st[:, 1], u_st[:, numerics.J]
    w_st[:, 0], w_st[:, numerics.J], w_st[:, numerics.J + 1] = 0, 0, 0
    self._latest_b_ph = b_ph
    return u_st, w_st


# =============================================================  SEMI-IMPLICIT / EXACT DIFFUSION FOR SCALARS ONLY  ================================================================

def implicit_diffusion(self: "FARE") -> None:
    """Apply exact horizontal hyperdiffusion and Crank-Nicolson vertical diffusion to scalar fields."""

    state = self.state
    cache = self._cache
    J = self.numerics.J

    def Z_Diffusion_CN_NEU(arr: np.ndarray, bg: np.ndarray) -> np.ndarray:
        RHS = np.zeros((self.numerics.I - 1, cache["zf"] - cache["zi"] + 2), dtype=complex)
        zi, zf, Sz = cache["zi"], cache["zf"], cache["Sz"]
        RHS[:, 1:-1] = Sz * (arr[:, zi - 1 : zf - 1] + arr[:, zi + 1 : zf + 1]) + 2 * (1 - Sz) * arr[:, zi:zf]
        # Directly enforce physical gradients in the RHS:
        RHS[:, 0] = bg[:, 0] * cache["dz"]
        RHS[:, -1] = bg[:, -1] * cache["dz"]
        return thomas_solve(RHS, cache["apy_diff_neu"], cache["bpy_diff_neu"], cache["cpy_diff_neu"])

    # exact horizontal hyperdiffusion (xdamp) and implicit vertical diffusion second step (Crank-Nicolson 2, only scalar pot/qt terms)
    state.pot[1] = cache["xdamp"] * Z_Diffusion_CN_NEU(state.pot[1], cache["DpotilCS"])
    state.qt[1] = cache["xdamp"] * Z_Diffusion_CN_NEU(state.qt[1], cache["DqvtilCS"])

    # Re-enforce B.C.s again (just in case yknow)
    state.pot[1, :, 0], state.pot[1, :, J + 1] = state.pot[1, :, 1] - cache["DpotilCS"][:, 0] * cache["dz"], state.pot[1, :, J] + cache["DpotilCS"][:, -1] * cache["dz"]
    state.qt[1, :, 0], state.qt[1, :, J + 1] = state.qt[1, :, 1] - cache["DqvtilCS"][:, 0] * cache["dz"], state.qt[1, :, J] + cache["DqvtilCS"][:, -1] * cache["dz"]



# =====================================================  SEMI-IMPLICIT INCREMENTAL PRESSURE POISSON SOLVER FOR P, U, and W  ==============================================================

def incremental_PPE(self: "FARE", u_st: np.ndarray, w_st: np.ndarray) -> None:
    """Apply the incremental pressure-Poisson projection and semi-implicit velocity correction."""

    state = self.state
    cache = self._cache
    numerics = self.numerics
    zi, zf, dz = cache["zi"], cache["zf"], cache["dz"]

    P_RHS = np.zeros((numerics.I - 1, numerics.J + 2), dtype=complex)
    P_RHS[:, 1:-1] = cache["rho0"][:, 1:-1] * cache["dz2"] / numerics.dt * (1j * cache["kx"][:, None] * u_st[:, zi:zf] + (w_st[:, zi:zf] - w_st[:, zi - 1 : zf - 1]) / dz)
    P_RHS[0] = 0.0
    P_st = thomas_solve_PPE(P_RHS, cache["apz_PPE"], cache["bpz_PPE"], cache["cpz_PPE"])

    #  pressure correction Neumann BC for all kx
    P_st[:, 0], P_st[:, numerics.J + 1] = P_st[:, 1], P_st[:, numerics.J]
    P_st[0, zi] = 0.0

    #something to keep the 0th wave mode consistent with PPE (otherwise it'll just be all zeros regardless of height)
    increments = w_st[0, zi : zf - 1] * cache["rho0_int"][0, zi : zf - 1] * dz / numerics.dt
    P_st[0, zi + 1 : zf] = P_st[0, zi] + np.cumsum(increments)

    # Re-apply Neumann for kx=0 mode specifically
    P_st[0, 0], P_st[0, numerics.J + 1] = P_st[0, zi], P_st[0, zf - 1]

    # update velocities with semi-implicit pressure + crank-nicolson vertical diffusion + exact horizontal hyperdiffusion
    RHS_u_solver = u_st + numerics.dt * (-1 / cache["rho0"] * 1j * cache["kx"][:, None] * P_st)
    RHS_u_solver[:, 0], RHS_u_solver[:, -1] = 0, 0
    state.u[1] = cache["xdamp"] * thomas_solve(RHS_u_solver, cache["apy_diff_com"], cache["bpy_diff_com"], cache["cpy_diff_com"])
    RHS_w_solver = w_st[:, zi : zf - 1] + numerics.dt * (-1 / cache["rho0_int"][:, zi : zf - 1] * (P_st[:, zi + 1 : zf] - P_st[:, zi : zf - 1]) / dz)
    state.w[1, :, zi : zf - 1] = cache["xdamp"] * thomas_solve(RHS_w_solver, cache["apy_diff_dir"], cache["bpy_diff_dir"], cache["cpy_diff_dir"])

    # re-enforce BCs
    state.u[1, :, 0], state.u[1, :, numerics.J + 1] = cache["u_bc"] * state.u[1, :, 1], state.u[1, :, numerics.J]
    state.w[1, :, 0], state.w[1, :, numerics.J], state.w[1, :, numerics.J + 1] = 0, 0, 0

    #final pressure update
    state.P[1] = state.P[0] + P_st
    state.P[1, :, 0], state.P[1, :, numerics.J + 1] = state.P[1, :, 1], state.P[1, :, numerics.J]



# ========================================================  SPONGES AND FILTERS ==================================================================

def apply_sponge(self: "FARE") -> None:
    """Apply time-ramped terrain and boundary sponge filters to the newly calculated state."""

    state = self.state
    cache = self._cache
    numerics = self.numerics
    scenario = self.scenario
    zi, zf = cache["zi"], cache["zf"]

    # Time-dependent ramp for masks
    # makes it so that the terrain and sponge layer aren't immediately forced into being at the start of the simulation, and instead take ~2 hours to reach their maximum forcing, which is less than the time it takes for physically realistic solutions to start appearing
    t_cur = (state.n + 1) * numerics.dt
    tau_ramp = 7200.0
    ramp = 1.0 if t_cur >= tau_ramp else np.sin(np.pi / 2 * (t_cur / tau_ramp)) ** 2
    current_mask_ph = cache["boundmask_ph"] + cache["mountmask_ph"] * ramp
    current_Wmask_ph = cache["boundWmask_ph"] + cache["mountWmask_ph"] * ramp

    #apply exact sponge layers + mountain mask to scalars
    qt_ph2 = dea_ifftx(state.qt[1], numerics.I)
    pot_ph2 = dea_ifftx(state.pot[1], numerics.I)
    state.qt[1, :, zi:zf] = dea_fftx(((qt_ph2 + cache["boundmask_ph"] * cache["qvtil_ph"]) / (1 + cache["boundmask_ph"]))[:, 1:-1], numerics.I)
    state.pot[1, :, zi:zf] = dea_fftx(((pot_ph2 + cache["boundmask_ph"] * cache["potil_ph"]) / (1 + cache["boundmask_ph"]))[:, 1:-1], numerics.I)

    #apply exact sponge layers + mountain mask to velocities
    u_clean_ph = dea_ifftx(state.u[1], numerics.I)
    w_clean_ph = dea_ifftx(state.w[1, :, :-1], numerics.I)
    state.u[1, :, zi:zf] = dea_fftx(((u_clean_ph + cache["boundmask_ph"] * cache["ubg_ph"]) / (1 + current_mask_ph))[:, 1:-1], numerics.I)
    state.w[1, :, zi : zf - 1] = dea_fftx((w_clean_ph / (1 + current_Wmask_ph))[:, 1:-1], numerics.I)
    J, dz = numerics.J, cache["dz"]

    # re-enforce BCs again
    state.pot[1, :, 0], state.pot[1, :, J + 1] = state.pot[1, :, 1] - cache["DpotilCS"][:, 0] * dz, state.pot[1, :, J] + cache["DpotilCS"][:, -1] * dz
    state.qt[1, :, 0], state.qt[1, :, J + 1] = state.qt[1, :, 1] - cache["DqvtilCS"][:, 0] * dz, state.qt[1, :, J] + cache["DqvtilCS"][:, -1] * dz
    state.u[1, :, 0], state.u[1, :, J + 1] = cache["u_bc"] * state.u[1, :, 1], state.u[1, :, J]
    state.w[1, :, 0], state.w[1, :, J], state.w[1, :, J + 1] = 0, 0, 0

    # RAW (Robert-Asselin-Williams) Filter to damp the computational mode if using Leapfrog
    if self._scheme_for_step(state.n + 1) == "LF":
        raw = numerics.RAW_filt
        state.u[0] += raw * (state.u[-1] - 2 * state.u[0] + state.u[1] - 0.5 * (state.u[1] - state.u[-1]))
        state.w[0] += raw * (state.w[-1] - 2 * state.w[0] + state.w[1] - 0.5 * (state.w[1] - state.w[-1]))
        state.pot[0] += raw * (state.pot[-1] - 2 * state.pot[0] + state.pot[1] - 0.5 * (state.pot[1] - state.pot[-1]))
        state.qt[0] += raw * (state.qt[-1] - 2 * state.qt[0] + state.qt[1] - 0.5 * (state.qt[1] - state.qt[-1]))
