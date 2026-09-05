"""Built-in FARE scenario constructors and their profile functions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from .data import PhysicalParameters, Scenario


@lru_cache(maxsize=2)
def _profile_interpolator(filename: str) -> PchipInterpolator:
    """Load one notebook profile table and return its height-to-forcing PCHIP interpolator."""

    path = Path(__file__).resolve().parents[1] / "profiles" / filename
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    values = values[np.argsort(values[:, 1])]
    heights, indices = np.unique(values[:, 1], return_index=True)
    return PchipInterpolator(heights, values[indices, 0])


def moistrate(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return the CSV-prescribed background moistening profile in kg/(kg s)."""

    return _profile_interpolator("moistening_profile.xls")(z / 1000.0) / (86400.0 * 1000.0)


def coolrate(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return the CSV-prescribed background cooling profile in K/s."""

    return _profile_interpolator("cooling_profile.xls")(z / 1000.0) / 86400.0


def shallow_convection(seed: int | None = 0, param: PhysicalParameters | None = None) -> Scenario:
    """Return the notebook's scattered-convection configuration with analytic forcing profiles."""

    param = param or PhysicalParameters()
    B = 0.003
    DALR = param.g / param.cp

    def pot0(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the linear background potential-temperature profile."""

        return param.theta0 + B * z

    def f0(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the hydrostatic Exner-function profile for the linear background."""

        return 1 - DALR / B * np.log1p(B * z / param.theta0)

    def qvi(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the initially saturated specific-humidity profile."""

        exner = f0(x, z)
        return param.qvso / exner**param.CpRd * np.exp(-param.LL / param.Rv * (1 / (exner * pot0(x, z)) - 1 / param.theta0))

    def poti(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the background temperature plus the near-surface random perturbation."""

        field = np.array(pot0(x, z), copy=True)
        x_mask = np.zeros(field.shape[0], dtype=bool)
        x_mask[15:-15] = True
        z_mask = z < 2000
        mask = x_mask[:, np.newaxis] & z_mask
        rng = np.random.default_rng(seed)
        field[mask] += rng.uniform(-0.1, 0.1, size=np.count_nonzero(mask))
        return field

    return Scenario(poti=poti, qvi=qvi, pot0=pot0, f0=f0, moistening=moistrate, cooling=coolrate, Vt=5.0, name="shallow_convection")


def squall_line(seed: int | None = 0, param: PhysicalParameters | None = None) -> Scenario:
    """Return the notebook's squall-line configuration with vertical wind shear."""

    scenario = shallow_convection(seed=seed, param=param)

    def vertical_shear(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the background horizontal wind profile used for squall-line organization."""

        H0 = 12000.0
        a = 11.11
        return np.where(z < H0, a * (np.cos(np.pi * z / H0) - np.cos(2 * np.pi * z / H0)), -2 * a)

    scenario.u_relax = vertical_shear
    scenario.name = "squall_line"
    return scenario


def atmospheric_river(seed: int | None = 0, param: PhysicalParameters | None = None) -> Scenario:
    """Return the notebook's atmospheric-river configuration with terrain and boundary sponges."""

    param = param or PhysicalParameters()
    DALR = param.g / param.cp
    Lx = 256000.0

    def pot0(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the smooth marine-boundary-layer and free-troposphere temperature profile."""

        theta_surf = 288.0
        gamma_mbl = 0.001
        gamma_trop = 0.004
        z_inv = 1250.0
        width = 200.0
        smooth_step = width * np.log(np.cosh((z - z_inv) / width)) - width * np.log(np.cosh(-z_inv / width))
        return theta_surf + (gamma_mbl + gamma_trop) * z / 2 + (gamma_trop - gamma_mbl) * smooth_step / 2

    def f0(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the Exner profile obtained by numerical hydrostatic integration."""

        integrand = -DALR / pot0(x, z)
        return 1.0 + cumulative_trapezoid(integrand, z, axis=-1, initial=0)

    def qvs(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the saturation specific-humidity profile for the scenario background state."""

        exner = f0(x, z)
        return param.qvso / exner**param.CpRd * np.exp(-param.LL / param.Rv * (1 / (exner * pot0(x, z)) - 1 / param.theta0))

    def qvi(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the initially subsaturated marine-boundary-layer humidity profile."""

        relative_humidity = 0.10 + 0.85 * 0.5 * (1 - np.tanh((z - 1250.0) / 200.0))
        return qvs(x, z) * relative_humidity

    def poti(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the background temperature plus the smaller atmospheric-river perturbation."""

        field = np.array(pot0(x, z), copy=True)
        x_mask = np.zeros(field.shape[0], dtype=bool)
        x_mask[30:-30] = True
        mask = x_mask[:, np.newaxis] & (z < 2000.0)
        rng = np.random.default_rng(seed)
        field[mask] += rng.uniform(-0.02, 0.02, size=np.count_nonzero(mask))
        return field

    def moistrate2(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return background plus coastal moisture forcing for the atmospheric river."""

        background = moistrate(x, z)
        xmask = 0.5 * (1 + np.tanh((100000.0 - x) / 10000.0))
        zmask = 0.5 * (1 - np.tanh((z - 1250.0) / 160.0))
        return background + 20.0 / (86400.0 * 1000.0) * xmask * zmask

    def coolrate2(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return a vertically localized cooling profile for the atmospheric river."""

        return coolrate(x, z)

    def surface_jet(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the lower-tropospheric background wind and embedded surface jet."""

        base_wind = 12.0 * 0.5 * (1 - np.tanh((z - 3000.0) / 1000.0))
        jet_core = 8.0 * np.exp(-((z - 1100.0) ** 2) / (2 * 1000.0**2))
        return base_wind + jet_core

    def mountainmask(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the smooth terrain sponge mask from the notebook configuration."""

        h = mountainshape(x)
        z_sigma = 100.0 + 100.0 * h / 1250.0
        raw_mask = 0.5 * (1 - np.tanh((z - h) / z_sigma)) * np.tanh(h / 50.0)
        return np.where(raw_mask > 0.02, raw_mask, 0.0)

    def boundarymask(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return the side and upper sponge-layer mask from the notebook configuration."""

        width = 15000.0
        leftmask = np.where(x < width, 0.5 * (1 + np.cos(np.pi * x / width)), 0.0)
        rightmask = np.where(x > Lx - width, 0.5 * (1 + np.cos(np.pi * (Lx - x) / width)), 0.0)
        topmask = 0.5 * (1 + np.tanh((z - 9000.0) / 1000.0))
        return 1 - (1 - leftmask) * (1 - rightmask) * (1 - topmask)

    def mountainshape(x: np.ndarray) -> np.ndarray:
        """Return the terrain-height profile used for atmospheric-river visualizations."""

        return 1500.0 * np.exp(-((x - 130000.0) ** 2) / (2 * 15000.0**2))

    return Scenario(
        poti=poti,
        qvi=qvi,
        pot0=pot0,
        f0=f0,
        u_relax=surface_jet,
        moistening=moistrate2,
        cooling=coolrate2,
        spongelayer=boundarymask,
        terrainmask=mountainmask,
        terrain_height=mountainshape,
        tauM=50.0,
        tauS=200.0,
        Vt=6.0,
        name="atmospheric_river",
    )
