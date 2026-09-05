"""Diagnostics, in-memory recorders, and disk-output helpers for FARE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import FARE


def save_stats(self: "FARE", slot: int = 0, time: float | None = None) -> None:
    """Record conservation, extrema, and CFL diagnostics from one live state time level."""

    state = self.state
    stats = self.stats
    cache = self._cache
    param = self.param
    b_ph = self._latest_b_ph
    index = stats.count
    dx = cache["dx"]
    dz = cache["dz"]
    dZ = cache["dZ"]
    z = cache["z"]
    qvs = cache["qvs"]
    P_phys = np.fft.ifft(state.P[slot], axis=0).real
    u_phys = np.fft.ifft(state.u[slot], axis=0).real
    w_phys = np.fft.ifft(state.w[slot], axis=0).real
    pot_phys = np.fft.ifft(state.pot[slot], axis=0).real
    qt_phys = np.fft.ifft(state.qt[slot], axis=0).real
    qvs_phys = np.fft.ifft(qvs, axis=0).real
    qv_phys = np.minimum(qt_phys, qvs_phys)
    qr_phys = np.maximum(qt_phys - qvs_phys, 0)
    pot_real = pot_phys + param.LCp * (qt_phys - qv_phys)
    area = dx * dz
    stats.cons_potr[index] = np.sum(pot_phys[:, 1:-1]) * area
    stats.cons_pote[index] = np.sum(pot_phys[:, 1:-1] + param.LCp * qv_phys[:, 1:-1]) * area
    stats.cons_pot[index] = np.sum(pot_real[:, 1:-1]) * area
    stats.cons_qt[index] = np.sum(qt_phys[:, 1:-1]) * area
    stats.cons_qr[index] = np.sum(qr_phys[:, 1:-1]) * area
    stats.cons_qv[index] = np.sum(qv_phys[:, 1:-1]) * area
    stats.cons_K[index] = 0.5 * np.sum(u_phys[:, 1:-1] ** 2 + w_phys[:, 1:-1] ** 2) * area
    stats.cons_B[index] = 0.0 if b_ph is None else -np.sum(b_ph[:, 1:-1]) * area
    stats.cons_R[index] = self.scenario.Vt * param.g * np.sum(-qr_phys[:, 1:-1] + (z[:, 2:] * qr_phys[:, 2:] - z[:, :-2] * qr_phys[:, :-2]) / dZ)
    stats.C[index] = np.max(np.abs(u_phys) * self.numerics.dt / dx + np.abs(w_phys) * self.numerics.dt / dz)
    stats.w_max[index] = np.max(np.abs(w_phys))
    stats.temp_max[index] = np.max(pot_phys)
    stats.count += 1


def save_solution(self: "FARE", slot: int = 0, time: float | None = None) -> None:
    """Record one physical-space solution snapshot from a live Fourier-space time level."""

    state = self.state
    sol = self.sol
    cache = self._cache
    param = self.param
    index = sol.count
    qvs = cache["qvs"]
    sol.t_s[index] = self.state.n * self.numerics.dt if time is None else time
    P_phys = np.fft.ifft(state.P[slot], axis=0).real
    u_phys = np.fft.ifft(state.u[slot], axis=0).real
    w_phys = np.fft.ifft(state.w[slot], axis=0).real
    pot_phys = np.fft.ifft(state.pot[slot], axis=0).real
    qt_phys = np.fft.ifft(state.qt[slot], axis=0).real
    qvs_phys = np.fft.ifft(qvs, axis=0).real
    qv_phys = np.minimum(qt_phys, qvs_phys)
    qr_phys = np.maximum(qt_phys - qvs_phys, 0)
    pot_real = pot_phys + param.LCp * (qt_phys - qv_phys)
    sol.P_s[index] = P_phys[:, 1:-1]
    sol.pot_s[index] = pot_real[:, 1:-1]
    sol.pot_r_s[index] = pot_phys[:, 1:-1]
    sol.u_s[index] = u_phys[:, 1:-1]
    sol.w_s[index] = w_phys[:, 1:-1]
    sol.qv_s[index] = qv_phys[:, 1:-1]
    sol.qr_s[index] = qr_phys[:, 1:-1]
    sol.count += 1
    self.state.m = self.sol.count


def save_state(self: "FARE", output_dir: str | Path) -> Path:
    """Write the current rolling state and RHS histories to a compressed NumPy archive."""

    self._require_ready()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = self.state
    path = output / "state.npz"
    np.savez_compressed(
        path,
        P=state.P,
        pot=state.pot,
        qt=state.qt,
        qr=state.qr,
        u=state.u,
        w=state.w,
        uRHS=state.uRHS,
        wRHS=state.wRHS,
        potRHS=state.potRHS,
        qtRHS=state.qtRHS,
        n=np.array(state.n),
        m=np.array(state.m),
    )
    return path


def write_output(self: "FARE", output_dir: str | Path) -> Path:
    """Write all in-memory solution and statistics records plus the live continuation state."""

    self._require_ready()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sol = self.sol
    stats = self.stats
    np.savez_compressed(
        output / "solution.npz",
        t_s=sol.t_s[: sol.count],
        x=sol.x,
        z=sol.z,
        P_s=sol.P_s[: sol.count],
        pot_s=sol.pot_s[: sol.count],
        pot_r_s=sol.pot_r_s[: sol.count],
        u_s=sol.u_s[: sol.count],
        w_s=sol.w_s[: sol.count],
        qv_s=sol.qv_s[: sol.count],
        qr_s=sol.qr_s[: sol.count],
    )
    np.savez_compressed(
        output / "stats.npz",
        cons_potr=stats.cons_potr[: stats.count],
        cons_pote=stats.cons_pote[: stats.count],
        cons_pot=stats.cons_pot[: stats.count],
        cons_qt=stats.cons_qt[: stats.count],
        cons_qr=stats.cons_qr[: stats.count],
        cons_qv=stats.cons_qv[: stats.count],
        cons_K=stats.cons_K[: stats.count],
        cons_B=stats.cons_B[: stats.count],
        cons_R=stats.cons_R[: stats.count],
        C=stats.C[: stats.count],
        w_max=stats.w_max[: stats.count],
        temp_max=stats.temp_max[: stats.count],
    )
    metadata = {
        "scenario": self.scenario.name,
        "L": self.numerics.L,
        "grid": self.numerics.grid,
        "dt": self.numerics.dt,
        "T": self.numerics.T,
        "s": self.numerics.s,
        "explicit_scheme": self.numerics.explicit_scheme,
        "completed_steps": self.state.n,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_state(self, output)
    return output
