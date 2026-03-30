# FARE-Model-w-Sponge-Terrain
This repository contains a 2D/3D minimal numerical model for simulating moist, precipitating turbulent convection, including scattered convection, squall lines, and atmospheric rivers. It was developed as part of a final project for the UCLA course A&O SCI 212A - Numerical Methods for Geophysical Fluid Dynamics. This codebase implements the Fast Autoconversion Rain Evaporation (FARE) formulation, where water vapor above the saturation vapor pressure is assumed to instantly condense into rainwater, and rainwater is assumed to evaporate back into  described in:

Hernandez-Duenas, Gerardo, Andrew J. Majda, Leslie M. Smith, and Samuel N. Stechmann. “Minimal Models for Precipitating Turbulent Convection.” Journal of Fluid Mechanics 717 (February 2013): 576–611. https://doi.org/10.1017/jfm.2012.597.

This model can appropriately simulate nonlinear advection and phase changes associated with moist atmospheric systems, directly iterating velocities (u, v, w), pressure (P), and scalar pertubations such as rainy potential temperature ($\theta_r$) and total specific humidity ($q_t$). For more information on the FARE equations and numerical schemes used, see ...

To install this, ...

Three key scenarios are presented as examples:
- Scattered Convection: Baseline turbulent simulation with horizontally uniform moisture/cooling forcing, no terrain modifications, and initial near-surface temperature pertubations.
- Squall Line: Introduces background wind shear to organize convection into fronts.
- Atmospheric River: Introduces a horizontally non-uniform moisture forcing, a Rayleigh-damped sponge mountain, and sponge layers in the upper and side boundaries to dampen gravity waves.
For more details on these scenarios, see ...
