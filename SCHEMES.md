# Mathematical Formulation

The model is built on the Moist Boussinesq Approximation, utilizing the Fast Autoconversion Rain Evaporation (FARE) limit for bulk microphysics. 

## 1. Governing Dynamics
The core system evolves the velocity vector $\mathbf{u} = (u, w)$, pressure $P$, rainy potential temperature $\theta_r$, and total water mixing ratio $q_t$. 

$$
\begin{align}
    \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} &= -\nabla P + B\mathbf{\hat{k}} + \nu \nabla^2 \mathbf{u} \\
    \nabla \cdot \mathbf{u} &= 0 \\
    \frac{\partial \theta_e}{\partial t} + \nabla \cdot (\mathbf{u} \theta_e) &= \mu_\theta \nabla^2 \theta_e + F_{\theta} \\
    \frac{\partial q_t}{\partial t} + \nabla \cdot (\mathbf{u} q_t) &= \frac{\partial}{\partial z} (V_T q_r) + \mu_q \nabla^2 q_t + F_{q}
\end{align}
$$

*Note: The $\frac{\partial}{\partial z} (V_T q_r)$ term in the moisture equation represents the removal of water from the domain via rain fallout at terminal velocity $V_T$.*

## 2. FARE Microphysics & Buoyancy
In the FARE approximation, the timescale for the conversion of cloud water ($q_c$) to rain ($q_r$) is instantaneous. Any moisture exceeding the saturation mixing ratio $q_{vs}(z)$ immediately converts to precipitating rain.

$$
\begin{align}
    q_v &= \min(q_t, q_{vs}(z)) \\
    q_r &= \max(q_t - q_{vs}(z), 0) \\
    \theta &= \theta_e - \frac{L_v}{c_p} q_v \\
    B &= g \left( \frac{\theta - \theta_0}{\theta_0} + \epsilon q_v - q_r \right)
\end{align}
$$

## 3. Discretized Advection Operators
To maintain numerical stability and strict conservation properties on the vertically staggered C-grid, nonlinear advection is separated into distinct forms for momentum and scalars.

**Momentum Advection (Skew-Symmetric Form):**
Used for $u$ and $w$ to discretely conserve kinetic energy, balancing the advective and flux forms.
$$
\mathcal{A}_{m}(\mathbf{u}) = \frac{1}{2} \left[ (\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla \cdot (\mathbf{u}\mathbf{u}) \right]
$$

**Scalar Advection (Flux Form):**
Used for $\theta_e$ and $q_t$ to strictly conserve total mass and thermodynamic energy.
$$
\mathcal{A}_{s}(\phi) = \nabla \cdot (\mathbf{u} \phi) = \frac{\partial (u \phi)}{\partial x} + \frac{\partial (w \phi)}{\partial z}
$$
