"""Public package interface for the Fast Autoconversion Rain Evaporation model."""

from .data import NumericalParameters, PhysicalParameters, Scenario, Solution, State, Stats
from .model import FARE
from .scenarios import atmospheric_river, shallow_convection, squall_line

__all__ = [
    "FARE",
    "PhysicalParameters",
    "NumericalParameters",
    "Scenario",
    "State",
    "Stats",
    "Solution",
    "shallow_convection",
    "squall_line",
    "atmospheric_river",
]
