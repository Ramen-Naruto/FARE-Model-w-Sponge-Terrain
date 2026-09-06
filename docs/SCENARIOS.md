# Scenarios

The FARE package provides three idealized two-dimensional experiments: scattered convection, a squall line, and an atmospheric river. They share the FARE dynamics and numerical framework in [FORMULATION.md](FORMULATION.md), but use different background winds, moisture sources, terrain, and damping.

## Shared environmental setup

The presentation baseline uses a 256 km horizontal domain and a 15 km vertical domain with a $256\times100$ grid, $\Delta x=1$ km, $\Delta z=150$ m, and $\Delta t=2.2$ s. Its physical constants include $g=9.8\ \mathrm{m\,s^{-2}}$, $\theta_0=300$ K, $c_p=1000\ \mathrm{J\,kg^{-1}\,K^{-1}}$, $L=2.5\times10^6\ \mathrm{J\,kg^{-1}}$, and $V_T=5.5\ \mathrm{m\,s^{-1}}$.

The baseline state is initialized with a small random potential-temperature perturbation below 2 km, initially saturated or nearly saturated low-level moisture, zero rainwater, and zero vertical velocity (top). The shared cooling (bottom left) and moistening profiles (bottom right) provide dissipation representative of the environment outside the simulation. The zonal wind relaxes toward zero over $\tau_u=14400$ s.
<p align="center">
<img src="../notebooks/useful_visuals/initial-saturated-moisture-prof.png" alt="Initial and saturated moisture profiles" width="40%">

<img src="../notebooks/useful_visuals/forcing-profs.png" alt="Cooling and moistening forcing profiles" width="80%">
</p>

## Scattered convection (shallow_convection)

<p align="center">
<img src="../notebooks/useful_visuals/shallow-convection.png" alt="Scattered convection" width="50%">
</p>

The scattered convection case starts with no imposed background wind and random near-surface temperature perturbations. Horizontally uniform cooling and moistening support moist overturning while leaving the organization to internally generated turbulence.

Create it with:

~~~
from fare_model import shallow_convection

scenario = shallow_convection()
~~~

[![Shallow Convection Simulation Video](https://img.youtube.com/vi/QHOAgpSjFoQ/maxresdefault.jpg)](https://youtu.be/QHOAgpSjFoQ)

## Squall line

<p align="center">
<img src="../notebooks/useful_visuals/squall-line.png" alt="Squall line" width="50%">
</p>

The squall-line case retains the thermodynamic setup of scattered convection and relaxes towards a background zonal wind with significant vertical shear over $\tau_u=14400$ s. The background wind is

<p align="center">
<img src="../notebooks/useful_visuals/background-wind-shear.png" alt="Squall-line background wind profile" width="50%">
</p>

$$
u_{\mathrm{bg}}(z)=
\begin{cases}
a\left[\cos\left(\dfrac{\pi z}{H_0}\right)-\cos\left(\dfrac{2\pi z}{H_0}\right)\right], & z \lt H_0,\\
-2a, & z \ge H_0.
\end{cases}
$$

where $H_0=12$ km and $a=11.11\ \mathrm{m\,s^{-1}}$. This shear organizes the initially scattered convection into a line-like system.

Create it with:

~~~
from fare_model import squall_line

scenario = squall_line()
~~~

[![Squall Line Simulation Video](https://img.youtube.com/vi/Bq_d9D17F0s/maxresdefault.jpg)](https://youtu.be/Bq_d9D17F0s)

## Atmospheric river

<p align="center">
<img src="../notebooks/useful_visuals/atmospheric-river.png" alt="Atmospheric river" width="50%">
</p>

The atmospheric-river configuration represents moist onshore flow over a 1.5 km mountain in a 12 km-deep domain. It combines a marine lower-tropospheric thermodynamic profile, a low-level jet, a coastal moisture source concentrated in roughly the lowest 1-2 km, and free-slip lower-boundary behavior for $u$.

To keep gravity waves and moisture from contaminating the interior, the case includes:

- an upper boundary sponge layer with a 200 s damping time;
- side sponge layers that extend 15 km inward from each lateral boundary;
- an upper sponge beginning near 8 km; and
- a terrain-adjacent mountain damping region with a 50 s time scale.

The background state combines a shallow marine boundary layer, an inversion near 1.25 km, low-level moisture, and a lower-tropospheric wind with an embedded jet.

<p align="center">
<img src="../notebooks/useful_visuals/background-atmospheric-river.png" alt="Atmospheric river vertical profiles" width="50%">
<img src="../notebooks/useful_visuals/atmospheric-river-set-up.png" alt="Atmospheric river sponge, terrain, and coastal moisture setup" width="100%">
</p>

Create it with:

~~~
from fare_model import atmospheric_river

scenario = atmospheric_river()
~~~

[![Atmospheric River Simulation Video](https://img.youtube.com/vi/SKqMBoDVNAw/maxresdefault.jpg)](https://youtu.be/SKqMBoDVNAw)

## Diagnostics and interpretation

The presentation tracks area-integrated temperature and water measures together with the Courant number and maximum vertical velocity. Equivalent potential temperature, $\theta_e$, is comparatively well conserved when rainwater remains small relative to vapor. CFL stability is a central practical constraint; the displayed squall-line integration eventually loses stability. Kinetic and potential energy are not expected to be tightly conserved because the experiments include external forcing, relaxation, and damping.

![Saved Diagnostics](../notebooks/useful_visuals/diagnostics.png)
