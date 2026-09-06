"""Data containers used by the FARE model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Profile = Callable[[np.ndarray, np.ndarray], np.ndarray]
HeightProfile = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
#Physical constants (defined here just in case)
class PhysicalParameters:
    """Store universal physical constants used in  the FARE formulation."""

    g: float = 9.8 #gravity
    theta0: float = 300.0 # reference potential temperature
    cp: float = 1000.0 # specific heat at constant pressure
    LL: float = 2.5e6 # latent heat of vaporization of water
    Rv: float = 462.0 # gas constant for water vapor
    Rd: float = 287.0 # gas constant for dry air
    p0: float = 1.0e5 # reference pressure
    epsilon0: float = 0.6 # approximately Rv/Rd - 1  
    qvso: float = 0.02 # saturation specific humidity at reference temperature and pressure

    @property
    def CpRd(self) -> float:
        """Ratio of specific heat at constant pressure to dry-air gas constant."""

        return self.cp / self.Rd

    @property
    def LCp(self) -> float:
        """Latent heat divided by specific heat at constant pressure."""

        return self.LL / self.cp


@dataclass
class NumericalParameters:
    """Store grid, time-integration, and artificial-diffusion settings."""

    L: tuple[float, float] # domain lengths in horizontal and vertical directions
    grid: tuple[int, int] # number of grid points in horizontal and vertical directions. It will include both boundaries, so the number of interior points is grid[0]-1 in horizontal (since periodic) and grid[1]-2 in vertical (since bounded).
    dt: float # physical time step
    T: float # final time
    s: int # saves every s time steps
    explicit_scheme: str = "AB3" # explicit time-integration scheme
    vis_art: float | None = None # artificial 2nd order viscosity coefficient in vertical direction
    gam_art: float | None = None # artificial 4th order hyperviscosity coefficient in horizontal
    RAW_filt: float = 0.01 # Robert-Asselin-Williams filter coefficient for leapfrog integration (accounts for computational mode)

    @property
    def I(self) -> int:
        """Return the horizontal grid-point count including the repeated periodic boundary."""

        return self.grid[0]

    @property
    def J(self) -> int:
        """Return the vertical grid-point count including physical boundaries."""

        return self.grid[1]

    @property
    def N(self) -> int:
        """Return the number of inclusive time samples implied by T and dt."""

        return int(self.T / self.dt + 1)


@dataclass
class Scenario:
    """Store scenario-dependent profiles, masks, and microphysical parameters."""

    poti: Profile # initial potential temperature profile
    qvi: Profile # initial specific humidity profile
    pot0: Profile # background potential temperature profile
    f0: Profile  # Exner function profile
    u_relax: Profile | None = None
    moistening: Profile | None = None
    cooling: Profile | None = None
    spongelayer: Profile | None = None
    terrainmask: Profile | None = None
    terrain_height: HeightProfile | None = None
    tauu: float = 14400.0 # relaxation time scale for horizontal velocity
    tauM: float = 50.0 # relaxation time scale for mountain
    tauS: float = 200.0 # relaxation time scale for sponge layer
    Vt: float = 5.5 # terminal velocity of rain
    name: str = "custom"


@dataclass
class State:
    """Store live fourier-space fields, RHS histories, and continuation metadata."""

    P: np.ndarray # pressure
    pot: np.ndarray # rainy potential temperature
    qt: np.ndarray # total water specific humidity
    qr: np.ndarray # rainy water specific humidity
    u: np.ndarray # horizontal (zonal) velocity
    w: np.ndarray # vertical velocity
    uRHS: np.ndarray 
    wRHS: np.ndarray
    potRHS: np.ndarray
    qtRHS: np.ndarray
    n: int = 0 # number of time steps completed
    m: int = 0 # number of time steps saved to disk (for continuation)


@dataclass
class Stats:
    """Store sampled conservation, extrema, and stability diagnostics."""

    cons_potr: np.ndarray # rainy potential temperature integrated through the grid
    cons_pote: np.ndarray # equivalent potential temperature integrated through the grid
    cons_pot: np.ndarray # potential temperature integrated through the grid
    cons_qt: np.ndarray # total water specific humidity integrated through the grid
    cons_qr: np.ndarray # rainy water specific humidity integrated through the grid
    cons_qv: np.ndarray # vapor water integrated through the grid
    cons_K: np.ndarray   # kinetic energy term integrated through the grid
    cons_B: np.ndarray   # buoyant potential term  integrated through the grid
    cons_R: np.ndarray   # rainwater potential term integrated through the grid
    C: np.ndarray       #  maximum Courant number
    w_max: np.ndarray   # maximum vertical velocity
    temp_max: np.ndarray # maximum temperature
    count: int = 0


@dataclass
class Solution:
    """Store sampled physical-space solution fields and their spatial/temporal coordinates."""

    t_s: np.ndarray # saved times
    x: np.ndarray 
    z: np.ndarray 
    P_s: np.ndarray
    pot_s: np.ndarray
    pot_r_s: np.ndarray
    u_s: np.ndarray
    w_s: np.ndarray
    qv_s: np.ndarray
    qr_s: np.ndarray
    count: int = 0
