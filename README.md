# FARE-Model-w-Sponge-Terrain
This repository contains a 2D minimal numerical model for simulating moist, precipitating turbulent convection, including scattered convection, squall lines, and atmospheric rivers. It was developed as part of a final project for the UCLA graduate course A&O SCI 212A - Numerical Methods for Geophysical Fluid Dynamics. This codebase implements the Fast Autoconversion Rain Evaporation (FARE) formulation, where water vapor above the saturation vapor pressure is assumed to instantly condense into rainwater, and rainwater quickly evaporates back into water vapor as it falls. Most parameters and schemes are as described in:

Hernandez-Duenas, Gerardo, Andrew J. Majda, Leslie M. Smith, and Samuel N. Stechmann. “Minimal Models for Precipitating Turbulent Convection.” Journal of Fluid Mechanics 717 (February 2013): 576–611. https://doi.org/10.1017/jfm.2012.597.

This model can appropriately simulate nonlinear advection and phase changes associated with moist atmospheric systems, directly iterating velocities (u, w), pressure (P), and scalar pertubations such as rainy potential temperature ($\theta_r$) and total specific humidity ($q_t$). For more information on the FARE equations and numerical schemes used, see ```docs/FORMULATION.md```.

Three key scenarios are presented as examples in the package:

- Scattered Convection: Baseline turbulent simulation with horizontally uniform moisture/cooling forcing, no terrain modifications, and initial near-surface temperature pertubation as described in Hernandez-Duenas et al 2013. 
- Squall Line: Introduces background wind shear to organize convection into fronts as specified in Hernandez-Duenas et al 2013.
- Atmospheric River: Introduces a horizontally non-uniform moisture forcing, a Rayleigh-damped sponge mountain, and sponge layers in the upper and side boundaries to dampen gravity waves.

For more details on these scenarios, see  ```docs/SCENARIOS.md```.

[![Atmospheric River Simulation Video](https://img.youtube.com/vi/SKqMBoDVNAw/maxresdefault.jpg)](https://youtu.be/SKqMBoDVNAw)

The model can be adapted to 3D with not too much difficulty as both horizontal directions are assumed to be fully periodic, meaning that the FFTs can simply be replaced with 2D FFTs.


## Python package

For an example implementation of this package for FARE simulations, see ```notebooks/demo.ipynb```.

The object-oriented package and code lives in `fare_model/`. Install it from the repository root with:

```bash
pip install -e .
```

The preset scenarios  tables from `profiles/moistening_profile.xls` and `profiles/cooling_profile.xls`.

You can then create, configure, and run a preset or custom simulation by instantiating a `FARE` class object:

```python
from fare_model import FARE, atmospheric_river

model = (
    FARE()
    .set_grid(L=(256000, 12000), grid=(257, 200), dt=1.1, T=80000, s=40)
    .set_comp(explicit_scheme="AB3")
    .set_scenario(atmospheric_river())
    .set_state()
)
model.solve(save=True, output_dir="output/atmospheric_river")
```

Fourier-space states used in the active simulation are available through `model.state`; saved physical-space solutions and diagnostics are available through `model.sol` and `model.stats`. `save=True` writes `solution.npz`, `stats.npz`, `state.npz`, and `metadata.json` to the supplied output directory. 

You can also continue an existing simulation without resetting it using:

```python
model.continue_solve(additional_steps=1000, save=True, output_dir="output/atmospheric_river")
```

The package includes preset `shallow_convection()`, `squall_line()`, and `atmospheric_river()` scenarios, which can be applied onto the FARE object like so:

```python
from fare_model import shallow_convection
model.set_scenario(shallow_convection())
```

Lastly, the package can also be accessed and ran through the command line; see `fare_model\__main__.py` for details.