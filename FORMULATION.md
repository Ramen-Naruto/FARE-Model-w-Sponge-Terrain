# Physical Formulation

The model is built on the Moist Boussinesq Approximation, utilizing the Fast Autoconversion (FARE) limit for bulk microphysics. 

## 1. Governing Dynamics
The core system evolves the velocity vector $\vec{u}$, kinematic pressure $\phi$, rainy potential temperature $\theta_r$, and total water mixing ratio $q_t$. The equations are expressed using the material derivative $\frac{\mathrm{D}}{\mathrm{D}t} = \frac{\partial}{\partial t} + \vec{u} \cdot \nabla$.

$$
\begin{align}
\frac{\mathrm{D}\vec{u}}{\mathrm{D}t} = -\nabla \phi + \hat{k} b(\theta_r, q_t, z) \\
\nabla \cdot \vec{u} = 0 \\
\frac{\mathrm{D}\theta_r}{\mathrm{D}t} + \frac{L}{c_p} V_T \frac{\partial q_r}{\partial z} = 0 \\
\frac{\mathrm{D}q_t}{\mathrm{D}t} - V_T \frac{\partial q_r}{\partial z} = 0
\end{align}
$$

## 2. Diagnostic Relations
The model relies on the following diagnostic relationships to relate potential temperature ($\theta$), equivalent potential temperature ($\theta_e$), rainy potential temperature ($\theta_r$), and the various water phases ($q_t, q_v, q_r$):

$$
\begin{align}
\theta_e &= \theta + \frac{L}{c_p} q_v \\
\theta_r &= \theta - \frac{L}{c_p} q_r \\
q_t &= q_v + q_r
\end{align}
$$

## 3. Buoyancy Formulation
Buoyancy ($b$) is determined by a piecewise function depending on whether the total water mixing ratio ($q_t$) has reached the saturation threshold ($q_{vs}$). It relies on the background states for potential temperature $\tilde{\theta}(z)$ and water vapor $\tilde{q}_v(z)$.

$$
\begin{equation}
b = g
\begin{cases}
\frac{\theta_r - \tilde{\theta}(z)}{\theta_o} + \varepsilon_o(q_t - \tilde{q}_v(z)) & \text{if } q_t < q_{vs} \\
\frac{\theta_r - \tilde{\theta}(z)}{\theta_o} + \left(\frac{L}{c_p\theta_o} - 1\right)(q_t - q_{vs}(z)) + \varepsilon_o(q_{vs}(z) - \tilde{q}_v(z)) & \text{if } q_t \geq q_{vs}.
\end{cases}
\end{equation}
$$


# Numerical Scheme

## Spatial Implementation

### Pseudo-spectral Fourier Galerkin (Horizontal, 3/2 rule)
Horizontal derivatives are computed in spectral space, where $k$ is the wavenumber:

$$
\begin{align}
\partial_x \hat{u}_k &= i k \hat{u}_k \\
\partial_x^2 \hat{u}_{k,j} &= -k^2 \hat{u}_{k,j}
\end{align}
$$

### 2nd-order Centered Differences (Vertical, Staggered)
Vertical derivatives are computed using finite differences on a staggered C-grid:

$$
\begin{align}
\partial_z \hat{u}_{k,j} &\approx \frac{\hat{u}_{k,j+1} - \hat{u}_{k,j-1}}{2\Delta z} = \frac{\hat{u}_{k,j+1/2} - \hat{u}_{k,j-1/2}}{\Delta z} \\
\partial_z^2 \hat{u}_{k,j} &\approx \frac{\hat{u}_{k,j+1} - 2\hat{u}_{k,j} + \hat{u}_{k,j-1}}{\Delta z^2}
\end{align}
$$

### Nonlinear Advection
To ensure stability and conservation, advection is formulated differently depending on the variable:

$$
\begin{align}
u \cdot \nabla \theta &= \nabla \cdot (u\theta) \quad \text{(Flux form for } \theta_r, q_t \text{)} \\
u \cdot \nabla v &= \frac{1}{2}(\nabla \cdot (uv) + u \cdot \nabla v) \quad \text{(Skew-symmetric form for } u, w \text{)}
\end{align}
$$

### Pressure Poisson Equation
With the hybrid spatial discretization, the continuous PPE reduces to a tridiagonal matrix system for each wavenumber, removing the main computational bottleneck:

$$
\nabla^2 \widehat{P}^{*} \approx (-k^2 + \partial_z^{2})\widehat{P}^{*}_{k,j} = (-k^2 I + D_{zz})\widehat{P}^{*}
$$

---

## Time Implementation

The model integrates forward in time using a Semi-Implicit Incremental Projection method. The integration operators are split based on the physical process:

### Explicit AB3 (Advection, Sources, Forcing)
A 3rd-order Adams-Bashforth scheme is used for the fully nonlinear terms:

$$
\frac{u^* - u^n}{\Delta t} = \frac{23}{12}f(u^n) - \frac{16}{12}f(u^{n-1}) + \frac{5}{12}f(u^{n-2})
$$

### Explicit EF (Pressure Forcing)

An Euler Forward step handles the intermediate pressure updates:
$$
\frac{u^* - u^n}{\Delta t} = f(u^n)
$$

### Implicit CN (Vertical Diffusion)
A Crank-Nicolson scheme is applied to vertical diffusion to maintain stability without overly restricting the time step:

$$
\frac{u^{n+1} - u^{*}}{\Delta t} = \frac{1}{2}(f(u^{*}) + f(u^{n+1}))
$$

### Exact (Horizontal Hyperdiffusion)
Horizontal hyperdiffusion is solved exactly in spectral space to eliminate high-frequency noise:

$$
\hat{u}_k^{n+1} = \exp(-\gamma k^4 \Delta t)\hat{u}_k^*
$$
