"""Command-line entry point for running a configured FARE simulation."""

from __future__ import annotations

import argparse

from .model import FARE
from .scenarios import atmospheric_river, shallow_convection, squall_line


def main() -> None:
    """Parse command-line inputs, construct a preset simulation, and run it through the FARE public API."""

    parser = argparse.ArgumentParser(description="Run a Fast Autoconversion Rain Evaporation simulation.")
    parser.add_argument("scenario", choices=("shallow_convection", "squall_line", "atmospheric_river"))
    parser.add_argument("--lx", type=float, default=256000.0)
    parser.add_argument("--lz", type=float, default=None)
    parser.add_argument("--nx", type=int, default=257)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--T", type=float, default=None)
    parser.add_argument("--s", type=int, default=None)
    parser.add_argument("--scheme", choices=("EF", "LF", "AB3"), default="AB3")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    factory = {"shallow_convection": shallow_convection, "squall_line": squall_line, "atmospheric_river": atmospheric_river}[args.scenario]
    atmospheric = args.scenario == "atmospheric_river"
    Lz = args.lz if args.lz is not None else 12000.0 if atmospheric else 15000.0
    nz = args.nz if args.nz is not None else 200 if atmospheric else 100
    dt = args.dt if args.dt is not None else 1.1 if atmospheric else 2.2
    T = args.T if args.T is not None else 80000.0 if atmospheric else 79200.0
    s = args.s if args.s is not None else 40 if atmospheric else 20
    model = FARE().set_grid((args.lx, Lz), (args.nx, nz), dt, T, s).set_comp(args.scheme).set_scenario(factory()).set_state()
    model.solve(save=not args.no_save, output_dir=None if args.no_save else args.output_dir)


if __name__ == "__main__":
    main()
