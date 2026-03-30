# Mathematical Formulation

The model is built on the Moist Boussinesq Approximation, utilizing the Fast Autoconversion (FARE) limit for bulk microphysics. 

## 1. Governing Dynamics
The core system evolves the velocity vector $\mathbf{u}$, kinematic pressure $\phi$, rainy potential temperature $\theta_r$, and total water mixing ratio $q_t$. The equations are expressed using the material derivative $\frac{\mathrm{D}}{\mathrm{D}t} = \frac{\partial}{\partial t} + \mathbf{u} \cdot \nabla$.

$$
\begin{align}
\frac{\mathrm{D}\mathbf{u}}{\mathrm{D}t} = -\nabla \phi + \mathbf{k} b(\theta_r, q_t, z) \\
\nabla \cdot \mathbf{u} = 0 \\
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
