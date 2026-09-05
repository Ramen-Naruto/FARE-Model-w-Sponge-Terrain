"""Object-oriented implementation of the FARE moist-convection model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import fft

from .data import NumericalParameters, PhysicalParameters, Scenario, Solution, State, Stats
from .diagnostics import save_solution as _save_solution
from .diagnostics import save_state as _save_state
from .diagnostics import save_stats as _save_stats
from .diagnostics import write_output as _write_output
from .helper import dea_fftx, dea_ifftx, var_update as _var_update
from .plotting import animate_rainvapor as _animate_rainvapor
from .plotting import animate_temperature_velocity as _animate_temperature_velocity
from .plotting import plot_conservation_stability as _plot_conservation_stability
from .solver import apply_sponge as _apply_sponge
from .solver import explicit_terms as _explicit_terms
from .solver import implicit_diffusion as _implicit_diffusion
from .solver import incremental_PPE as _incremental_PPE
from .solver import _prepare_static_fields as _prepare_static_fields_impl


# A generalized FARE solver with kwargs for background profiles, forcing, sponge layers, and sponge terrain. 
# Note that for all function inputs (basically all initials/backgrounds/masks), they must be functions of at least x or z so that arrays can be constructed (if they are 0 or constant, you can just do 0*z + c)
class FARE:
    """Configure, solve, continue, diagnose, visualize, and save a FARE simulation."""

    # unpack imported functions into the class namespace to be called as methods
    explicit_terms = _explicit_terms
    implicit_diffusion = _implicit_diffusion
    incremental_PPE = _incremental_PPE
    apply_sponge = _apply_sponge
    save_stats = _save_stats
    save_solution = _save_solution
    save_state = _save_state
    write_output = _write_output
    var_update = _var_update
    animate_temperature_velocity = _animate_temperature_velocity
    animate_rainvapor = _animate_rainvapor
    plot_conservation_stability = _plot_conservation_stability
    _prepare_static_fields = _prepare_static_fields_impl

    def __init__(self, param: PhysicalParameters | None = None) -> None:
        """Create an unconfigured model with universal physical parameters."""

        self.param = param or PhysicalParameters()
        self.numerics: NumericalParameters | None = None
        self.scenario: Scenario | None = None
        self.state: State | None = None
        self.stats: Stats | None = None
        self.sol: Solution | None = None
        self._cache: dict[str, Any] = {}
        self._static_ready = False
        self._numerics_locked = False
        self._scratch: tuple[np.ndarray, np.ndarray] | None = None
        self._latest_b_ph: np.ndarray | None = None



    #     ================================================   COMPUTATIONAL SET UP    =================================================


    def set_grid(self, L: tuple[float, float] | list[float], grid: tuple[int, int] | list[int], dt: float, T: float, s: int) -> "FARE":
        """Set the computational domain, resolution, integration duration, timestep, and save stride."""

        self._ensure_numerics_mutable()
        Lx, Lz = map(float, L)
        I, J = map(int, grid)
        if Lx <= 0 or Lz <= 0 or I < 3 or J < 3 or dt <= 0 or T < 0 or s < 1:
            raise ValueError("L, grid, dt, T, and s must define a positive, resolvable simulation.")
        if self.numerics is None:
            self.numerics = NumericalParameters((Lx, Lz), (I, J), float(dt), float(T), int(s))
        else:
            self.numerics.L = (Lx, Lz)
            self.numerics.grid = (I, J)
            self.numerics.dt = float(dt)
            self.numerics.T = float(T)
            self.numerics.s = int(s)
        self._invalidate_static()
        return self

    def set_comp(self, explicit_scheme: str = "AB3", vis_art: float | None = None, gam_art: float | None = None, RAW_filt: float = 0.01) -> "FARE":
        """Set the explicit scheme, artificial diffusion, hyperdiffusion, and RAW-filter coefficient."""

        self._ensure_numerics_mutable()
        if self.numerics is None:
            raise RuntimeError("Call set_grid before set_comp.")
        explicit_scheme = explicit_scheme.upper()
        if explicit_scheme not in {"EF", "LF", "AB3"}:
            raise ValueError("explicit_scheme must be 'EF', 'LF', or 'AB3'.")
        if vis_art is not None and vis_art < 0:
            raise ValueError("vis_art must be non-negative.")
        if gam_art is not None and gam_art < 0:
            raise ValueError("gam_art must be non-negative.")
        if RAW_filt < 0:
            raise ValueError("RAW_filt must be non-negative.")
        self.numerics.explicit_scheme = explicit_scheme
        self.numerics.vis_art = vis_art
        self.numerics.gam_art = gam_art
        self.numerics.RAW_filt = float(RAW_filt)
        self._invalidate_static()
        return self



    #     ================================================    SCENARIO  SET-UP  (see scenarios.py for examples)  =================================================

    def set_scenario(self, scenario: Scenario) -> "FARE":
        """Apply a complete scenario's physical profiles, sponge layer, and terrain mask."""

        self.scenario = scenario
        self.set_phys(scenario.poti, scenario.qvi, scenario.pot0, scenario.f0, scenario.u_relax, scenario.moistening, scenario.cooling, scenario.tauu, scenario.Vt)
        self.add_sponge(spongelayer=scenario.spongelayer, tauS=scenario.tauS)
        self.add_terrain(terrainmask=scenario.terrainmask, terrain_height=scenario.terrain_height, tauM=scenario.tauM)
        return self

    def set_phys(self, poti: Any | None = None, qvi: Any | None = None, pot0: Any | None = None, f0: Any | None = None, u_relax: Any | None = None, moistening: Any | None = None, cooling: Any | None = None, tauu: float | None = None, Vt: float | None = None) -> "FARE":
        """Set or replace initial/background profiles, forcing profiles, and physical relaxation parameters."""

        if self.scenario is None:
            if any(value is None for value in (poti, qvi, pot0, f0)):
                raise ValueError("poti, qvi, pot0, and f0 are required when creating a scenario.")
            self.scenario = Scenario(poti=poti, qvi=qvi, pot0=pot0, f0=f0)
        scenario = self.scenario
        for name, value in (("poti", poti), ("qvi", qvi), ("pot0", pot0), ("f0", f0), ("u_relax", u_relax), ("moistening", moistening), ("cooling", cooling)):
            if value is not None:
                setattr(scenario, name, value)
        if tauu is not None:
            scenario.tauu = float(tauu)
        if Vt is not None:
            scenario.Vt = float(Vt)
        self._invalidate_static()
        return self

    def add_sponge(self, spongelayer: Any | None = None, add: bool = False, side: float = 0.0, upper: float = 0.0, width: float = 0.0, tauS: float | None = None) -> "FARE":
        """Set an arbitrary sponge mask or construct a side/upper sponge mask from convenience inputs."""

        self._require_scenario()
        if spongelayer is not None:
            self.scenario.spongelayer = spongelayer
        elif add:
            if self.numerics is None or width <= 0:
                raise ValueError("A positive width and configured grid are required for a constructed sponge layer.")
            Lx, Lz = self.numerics.L

            def constructed_sponge(x: np.ndarray, z: np.ndarray) -> np.ndarray:
                """Return a cosine side sponge combined with a smooth upper sponge."""

                left = np.where(x < width, 0.5 * side * (1 + np.cos(np.pi * x / width)), 0.0)
                right = np.where(x > Lx - width, 0.5 * side * (1 + np.cos(np.pi * (Lx - x) / width)), 0.0)
                top = upper * 0.5 * (1 + np.tanh((z - (Lz - width)) / max(width / 4, 1.0)))
                return 1 - (1 - left) * (1 - right) * (1 - top)

            self.scenario.spongelayer = constructed_sponge
        elif spongelayer is None:
            self.scenario.spongelayer = None
        if tauS is not None:
            self.scenario.tauS = float(tauS)
        self._invalidate_static()
        return self

    def add_terrain(self, terrainmask: Any | None = None, terrain_height: Any | None = None, add: bool = False, tauM: float | None = None) -> "FARE":
        """Set or clear the terrain sponge mask and optional terrain-height profile used for visualization."""

        self._require_scenario()
        if terrainmask is not None:
            self.scenario.terrainmask = terrainmask
        elif not add:
            self.scenario.terrainmask = None
        if terrain_height is not None:
            self.scenario.terrain_height = terrain_height
        elif not add and terrainmask is None:
            self.scenario.terrain_height = None
        if tauM is not None:
            self.scenario.tauM = float(tauM)
        self._invalidate_static()
        return self



    #  ====================================     INITIAL CONDITIONS   =========================================    
     
    def set_state(self) -> "FARE":
        """Create the initial rolling model state from the configured grid and scenario initial conditions."""

        self._require_configuration()
        self._prepare_static_fields()
        numerics = self.numerics
        cache = self._cache
        I, J = numerics.I, numerics.J

        #calculation arrays, these all occur in fourier space (hence dtype=complex)
        shape = (5, I - 1, J + 2)
        P = np.zeros(shape, dtype=complex)
        pot = np.zeros(shape, dtype=complex)
        qt = np.zeros(shape, dtype=complex)
        u = np.zeros(shape, dtype=complex)
        w = np.zeros(shape, dtype=complex)

        #Initial potential temperature
        pot[0] = fft(self._field(self.scenario.poti, cache["x"], cache["z"]), axis=0)

        #Initial vapor specific humidity (rain assumed to be 0)
        qt[0] = fft(self._field(self.scenario.qvi, cache["x"], cache["z"]), axis=0)

        #Background u, the system will attempt to relax back to this initial on a time scale (default 4 hours)
        u[0] = cache["ubg"]
        DpotilCS = cache["DpotilCS"]
        DqvtilCS = cache["DqvtilCS"]
        u_bc = cache["u_bc"]
        pot[0, :, 0], pot[0, :, J + 1] = pot[0, :, 1] - DpotilCS[:, 0] * cache["dz"], pot[0, :, J] + DpotilCS[:, -1] * cache["dz"]
        qt[0, :, 0], qt[0, :, J + 1] = qt[0, :, 1] - DqvtilCS[:, 0] * cache["dz"], qt[0, :, J] + DqvtilCS[:, -1] * cache["dz"]

        # IMPORTANT: SETTING THE HORIZONTAL VELOCITY [U] LOWER BOUNDARY CONDITIONS
        #  -  If the average background horizontal velocity near the surface is near 0, we use Dirichlet 0 (no slip) B.C. there as a crude parametrization of turbulence. 
        #  -  If the horizontal velocity is significantly greater than 0, we instead use Neumann (free slip) B.C. If we try to use Dirichlet u_surface B.C. here, it will inject unwanted energy into the system.
        u[0, :, 0], u[0, :, J + 1] = u_bc * u[0, :, 1], u[0, :, J - 1]
        qr = dea_fftx(np.maximum(dea_ifftx(qt[0], I) - cache["qvs_ph"], 0), I)

        #helpful temporary arrays for implicit-explicit operator splitting
        #temporary storage for previous time steps (necessary for AB3 and to some extent LF)
        self.state = State(P=P, pot=pot, qt=qt, qr=qr, u=u, w=w, uRHS=np.zeros((5, I - 1, J), dtype=complex), wRHS=np.zeros((5, I - 1, J - 1), dtype=complex), potRHS=np.zeros((5, I - 1, J), dtype=complex), qtRHS=np.zeros((5, I - 1, J), dtype=complex))

        #statistics initialization
        self._allocate_records(numerics.N)
        self.save_stats(slot=0, time=0.0)

        #save initial conditions
        self.save_solution(slot=0, time=0.0)
        self._numerics_locked = True
        return self


    #     ================================================    SOLVING (see solver.py and diagnostics.py)   ===================================================

    def solve(self, save: bool = True, output_dir: str | Path | None = None, progress: bool = True) -> "FARE":
        """Advance from the current state to the configured final time and optionally write results to output_dir."""
        self._require_ready()
        if save and output_dir is None:
            raise ValueError("output_dir is required when save=True.")
        self._run(self.numerics.N - 1, progress=progress)
        if save:
            self.write_output(output_dir)
        return self

    def continue_solve(self, additional_steps: int, save: bool = True, output_dir: str | Path | None = None, progress: bool = True) -> "FARE":
        """Advance an already initialized model by additional_steps while retaining state and multistep histories."""
        self._require_ready()
        if additional_steps < 1:
            raise ValueError("additional_steps must be at least one.")
        if save and output_dir is None:
            raise ValueError("output_dir is required when save=True.")
        end_step = self.state.n + int(additional_steps)
        self._grow_records(end_step // self.numerics.s + 1)
        self.numerics.T = end_step * self.numerics.dt
        self._run(end_step, progress=progress)
        if save:
            self.write_output(output_dir)
        return self

    def _run(self, end_step: int, progress: bool = True) -> None:
        """Execute all solver steps through the inclusive target iteration number without writing disk output."""

        total_saved_steps = end_step // self.numerics.s
        printed_progress = False
        for n in range(self.state.n + 1, end_step + 1):
            scheme = self._scheme_for_step(n)
            u_st, w_st = self.explicit_terms(scheme)
            self.implicit_diffusion()
            self.incremental_PPE(u_st, w_st)
            self.apply_sponge()

            # ---------  SAVING THE SOLUTION ------------
            if n % self.numerics.s == 0:
                self.save_stats(slot=1, time=self._time_for_step(n))
                self.save_solution(slot=1, time=self._time_for_step(n))
                if progress:
                    saved_step = self.sol.count - 1
                    percent_complete = 100 * n / end_step
                    print(f"\rSave {saved_step}/{total_saved_steps} | {percent_complete:6.2f}% complete", end="", flush=True)
                    printed_progress = True
            self.var_update()
            self.state.n = n
        if printed_progress:
            print()



     #     ===============================    Time-stepping / Explicit scheme Helpers    =====================================

    def _scheme_for_step(self, n: int) -> str:
        """Return the startup or multistep explicit scheme required for iteration n; implicit time scheme is always Crank-Nicolson 2."""

        scheme = self.numerics.explicit_scheme
        if scheme == "EF":
            return "EF"
        if scheme == "LF":
            return "EF" if n < 2 else "LF"
        return "EF" if n < 2 else "LF" if n < 3 else "AB3"

    def _time_for_step(self, n: int) -> float:
        """Return the saved time coordinate using the notebook's inclusive linspace convention."""

        return float(np.linspace(0.0, self.numerics.T, self.numerics.N)[n])
    


    #     ===================================    FIELD / RECORD SAVE HELPERS    =======================================

    def _allocate_records(self, N: int) -> None:
        """Allocate solution and statistics storage for an inclusive total of N time samples."""

        I, J = self.numerics.I, self.numerics.J
        capacity = (N - 1) // self.numerics.s + 1
        x = self._cache["x"][:, 1:-1].copy()
        z = self._cache["z"][:, 1:-1].copy()
        shape = (capacity, I - 1, J)

        #saved solution arrays
        self.sol = Solution(np.empty(capacity), x, z, *(np.empty(shape) for _ in range(7)))

        #saved statistic arrays
        self.stats = Stats(*(np.zeros(capacity) for _ in range(12)))

    def _grow_records(self, required_capacity: int) -> None:
        """Expand sampled solution and statistics arrays without altering existing records."""

        if required_capacity <= self.sol.P_s.shape[0]:
            return
        capacity = max(required_capacity, 2 * self.sol.P_s.shape[0])

        def grow(array: np.ndarray) -> np.ndarray:
            """Return a larger leading-dimension array containing the original values."""

            expanded = np.empty((capacity, *array.shape[1:]), dtype=array.dtype)
            expanded[: array.shape[0]] = array
            return expanded

        for name in ("t_s", "P_s", "pot_s", "pot_r_s", "u_s", "w_s", "qv_s", "qr_s"):
            setattr(self.sol, name, grow(getattr(self.sol, name)))
        for name in ("cons_potr", "cons_pote", "cons_pot", "cons_qt", "cons_qr", "cons_qv", "cons_K", "cons_B", "cons_R", "C", "w_max", "temp_max"):
            setattr(self.stats, name, grow(getattr(self.stats, name)))

    def _field(self, profile: Any | None, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate a profile or substitute a zero field, broadcasting scalar results onto the full grid."""

        if profile is None:
            return np.zeros_like(x)
        value = np.asarray(profile(x, z))
        return np.broadcast_to(value, x.shape).copy()

    def _invalidate_static(self) -> None:
        """Mark profiles, masks, and linear operators for reconstruction before the next solve step."""

        self._static_ready = False


    #  ===================================    ERROR MESSAGE HELPERS ======================================

    def _ensure_numerics_mutable(self) -> None:
        """Reject numerical changes after a state has been initialized because histories become incompatible."""

        if self._numerics_locked:
            raise RuntimeError("Numerical parameters cannot change after set_state; create a new FARE instance with the current scenario and state.")

    def _require_scenario(self) -> None:
        """Raise a clear error when a scenario-dependent method is called before scenario setup."""

        if self.scenario is None:
            raise RuntimeError("Configure a scenario before setting physical profiles or masks.")

    def _require_configuration(self) -> None:
        """Raise a clear error when grid, numerics, or scenario setup is incomplete."""

        if self.numerics is None or self.scenario is None:
            raise RuntimeError("Call set_grid, set_comp, and set_scenario before initializing state.")

    def _require_ready(self) -> None:
        """Raise a clear error when solving, saving, or plotting is requested before state initialization."""

        self._require_configuration()
        if self.state is None or self.sol is None or self.stats is None:
            raise RuntimeError("Call set_state before solving, saving, or plotting.")
        self._prepare_static_fields()
