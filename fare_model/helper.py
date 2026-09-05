"""Small numerical helpers used by the FARE solver."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.fft import fft, ifft

#============================================================================================================================================================================================
#              HELPER FUNCTIONS                  HELPER FUNCTIONS                 HELPER FUNCTIONS                   HELPER FUNCTIONS                   HELPER FUNCTIONS                  
#============================================================================================================================================================================================

# general helper functions which do not require FARE inputs (tridiagonal matrix (Thomas) solvers, interpolation)
# njit is used to let the functions compile directly in C; explicit for loops are otherwise excruciatingly slow compiled in Python

@njit
#stores an LU-decomposed matrix given lower, center, and upper diagonal 1D arrays a, b, and c. The benefit of this is that it doesn't need to be done at every iteration.
def thomas_factor(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the LU factors of a tridiagonal matrix represented by its three diagonals."""

    n = len(b)
    bp = b.copy()
    cp = c.copy()
    ap = a.copy()
    for i in range(1, n):
        m = ap[i - 1] / bp[i - 1]
        bp[i] -= m * cp[i - 1]
        ap[i - 1] = m   # store multiplier (L factor)
    return ap, bp, cp


@njit
# solves the LU-decomposed lower, center, and upper diagonal arrays ap, bp, and cp given a right-hand side.
def thomas_solve(rhs: np.ndarray, ap: np.ndarray, bp: np.ndarray, cp: np.ndarray) -> np.ndarray:
    """Solve a factored tridiagonal system for every horizontal Fourier mode."""

    nx, n = rhs.shape
    sol = np.empty_like(rhs)
    for i in range(nx):
        # Forward substitution (Ly = rhs)
        y = rhs[i].copy()
        for j in range(1, n):
            y[j] -= ap[j - 1] * y[j - 1]
        # Back substitution (Ux = y)
        x = np.empty(n, dtype=rhs.dtype)
        x[n - 1] = y[n - 1] / bp[n - 1]
        for j in range(n - 2, -1, -1):
            x[j] = (y[j] - cp[j] * x[j + 1]) / bp[j]
        sol[i] = x
    return sol


@njit
# a special thomas solver for the PPE to account for the lack of pressure constraint (which is why the 0th wave-mode is explicitly set to 0)
def thomas_solve_PPE(rhs: np.ndarray, ap: np.ndarray, bp: np.ndarray, cp: np.ndarray) -> np.ndarray:
    """Solve the pressure Poisson tridiagonal systems while pinning the zero Fourier mode."""

    nx, n = rhs.shape
    sol = np.empty_like(rhs)
    for i in range(nx):
        # kx = 0 mode
        if i == 0:
            sol[i] = 0
            continue
        # Forward substitution
        y = rhs[i].copy()
        for j in range(1, n):
            y[j] -= ap[i, j - 1] * y[j - 1]
        # Back substitution
        sol[i, n - 1] = y[n - 1] / bp[i, n - 1]
        for j in range(n - 2, -1, -1):
            sol[i, j] = (y[j] - cp[i, j] * sol[i, j + 1]) / bp[i, j]
    return sol


# ----------------------------------------------  advection helper functions  --------------------------------------------------

#dealiasing inverse fft, zero-pads the array by 3/2 (1/4 on each side) when transforming to physical space
def dea_ifftx(arr: np.ndarray, I: int) -> np.ndarray:
    """Return the 3/2-padded inverse Fourier transform used for dealiased nonlinear products."""

    mid = (I - 1) // 2
    pad = np.zeros(((I - 1) * 3 // 2, arr.shape[1]), dtype=complex)
    pad[:mid] = arr[:mid]
    pad[-mid:] = arr[-mid:]
    return ifft(pad, axis=0).real


#dealiasing fft, takes in a physical array that is 3/2-padded and
def dea_fftx(arr: np.ndarray, I: int) -> np.ndarray:
    """Return the de-padded Fourier transform of a 3/2-padded physical-space field."""

    mid = (I - 1) // 2
    transformed = fft(arr, axis=0)
    depad = np.zeros((I - 1, arr.shape[1]), dtype=complex)
    depad[:mid] = transformed[:mid]
    depad[-mid:] = transformed[-mid:]
    return depad


# ----------------------------------------------  misc helpers  --------------------------------------------------

#interpolates an arbitrary array and returns an interpolated array 1 smaller along z
def zinterp(arr: np.ndarray) -> np.ndarray:
    """Interpolate a field halfway between adjacent vertical levels."""

    return (arr[:, :-1] + arr[:, 1:]) / 2


def var_update(self) -> None:
    """Promote the newly calculated state and RHS values into the rolling time histories."""

    state = self.state
    for field in (state.P, state.pot, state.u, state.w, state.qt):
        field[2] = field[-1]
        field[-2] = field[2]
        field[2] = field[0]
        field[-1] = field[2]
        field[0] = field[1]
    for rhs in (state.uRHS, state.wRHS, state.potRHS, state.qtRHS):
        rhs[2] = rhs[-1]
        rhs[-2] = rhs[2]
        rhs[2] = rhs[0]
        rhs[-1] = rhs[2]
