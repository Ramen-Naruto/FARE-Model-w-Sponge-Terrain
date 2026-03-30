# Physical Formulation

The model is built on the Moist Boussinesq Approximation, utilizing the Fast Autoconversion (FARE) limit for bulk microphysics. 

## Governing Dynamics
The core system evolves the velocity vector $\vec{u}$, pressure $P$, rainy potential temperature $\theta_r$, and total water mixing ratio $q_t$. The equations are expressed using the material derivative $\frac{\mathrm{D}}{\mathrm{D}t} = \frac{\partial}{\partial t} + \vec{u} \cdot \nabla$.

$$
\begin{align}
\frac{\mathrm{D}\vec{u}}{\mathrm{D}t} = -\frac{1}{\rho}\nabla P + \hat{k} b(\theta_r, q_t, z) \\
\nabla \cdot \vec{u} = 0 \\
\frac{\mathrm{D}\theta_r}{\mathrm{D}t} + \frac{L}{c_p} V_T \frac{\partial q_r}{\partial z} = 0 \\
\frac{\mathrm{D}q_t}{\mathrm{D}t} - V_T \frac{\partial q_r}{\partial z} = 0
\end{align}
$$

### Diagnostic Relations
The model relies on the following diagnostic relationships to relate potential temperature ($\theta$), equivalent potential temperature ($\theta_e$), rainy potential temperature ($\theta_r$), and the water vapor content ($q_t, q_v, q_r$).

$$
\begin{align}
\theta_e &= \theta + \frac{L}{c_p} q_v \\
\theta_r &= \theta - \frac{L}{c_p} q_r \\
q_t &= q_v + q_r \\
q_r &= max(q_t - q_{vs}, 0) \\
q_v &= min(q_t, q_{vs})
\end{align}
$$

### Buoyancy Formulation
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

### Horizontal Pseudo-spectral Fourier Galerkin with 3/2 dealiasing
Horizontal derivatives are computed in spectral space, where $k$ is the wavenumber:

$$
\begin{align}
\partial_x \hat{u}_k &= i k \hat{u}_k \\
\partial_x^2 \hat{u}_{k,j} &= -k^2 \hat{u}_{k,j}
\end{align}
$$

### Vertical 2nd-order Staggered Centered Differences
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
With the hybrid spatial discretization, the continuous PPE reduces to a tridiagonal matrix system along the vertical dimension for each horizontal wavenumber, removing the main computational bottleneck:

$$
\nabla^2 \widehat{P}^\ast \approx (-k^2 I + D_{zz})\widehat{P}^\ast \approx \frac{\rho_{0}}{\Delta t} \nabla \cdot (\widehat{\vec{u}}^\ast)
$$

Note: For the horizontal mean wavemode ($k=0$), the $-k^2$ term in the Poisson equation vanishes. Combined with Neumann boundary conditions at the vertical limits, the resulting tridiagonal matrix is singular. 

We bypass this singularity by solving the $k=0$ pressure analytically. Because mass conservation and rigid boundaries require the mean vertical velocity to be zero at all heights ($w^{n+1}_{k=0} = 0$), the projection step for the mean mode simplifies to:

$$
\frac{\partial \widehat{P}^\ast_0}{\partial z} = \rho_0 \frac{\widehat{w}^\ast_0}{\Delta t}
$$

We integrate this vertical pressure gradient analytically from the surface upward using a cumulative sum. This recovers the mean pressure profile while explicitly pinning the surface pressure to a reference value.

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
\frac{u^{n+1} - u^\ast}{\Delta t} = \frac{1}{2}(f(u^\ast) + f(u^{n+1}))
$$

### Exact (Horizontal Hyperdiffusion)
Horizontal hyperdiffusion is solved exactly in spectral space to eliminate high-frequency noise:

$$
\hat{u}_k^{n+1} = \exp(-\gamma k^4 \Delta t)\hat{u}_k^*
$$

## Overall Procedure

### Semi-Implicit Advection-Diffusion ($\hat{\theta}_r, \hat{q}_t$)

1. **Apply all explicit terms with 3/2 dealiasing, excluding diffusion:**
   
$$
\hat{\theta}^n \rightarrow \hat{\theta}^\ast \quad \text{and} \quad \hat{q}^n \rightarrow \hat{q}^\ast
$$

2. **Apply implicit diffusion and exact hyperdiffusion:**

$$
\hat{\theta}^\ast \rightarrow \hat{\theta}^{n+1} \quad \text{and} \quad \hat{q}^\ast \rightarrow \hat{q}^{n+1}
$$

---

### Semi-Implicit Incremental Projection ($\hat{u}, \hat{w}, \hat{P}$)

1. **Apply all explicit terms with 3/2 dealiasing, including the explicit half of diffusion:**
   
$$
\hat{u}^n \rightarrow \hat{u}^\ast \quad \text{and} \quad \hat{w}^n \rightarrow \hat{w}^\ast
$$

2. **Solve the incremental Pressure Poisson Equation (PPE) for the pressure perturbation:** $\hat{p}^\ast$
   
3. **Update the full pressure field** ($\hat{p}^n + \hat{p}^\ast \rightarrow \hat{p}^{n+1}$) **and apply the boundary conditions for** $\hat{p}^{n+1}$.
4. **Apply the pressure gradient** ($\nabla \hat{p}^{n+1}$) **along with implicit diffusion and exact hyperdiffusion to correct the velocity field:**
   
$$
\hat{u}^\ast \rightarrow \hat{u}^{n+1} \quad \text{and} \quad \hat{w}^\ast \rightarrow \hat{w}^{n+1}
$$

