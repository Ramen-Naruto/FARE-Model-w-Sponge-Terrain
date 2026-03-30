## FARE Equations

The FARE model equations are given by:
\begin{align}
    % Momentum Equation
    \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} &= -\nabla P + B\mathbf{\hat{k}} + \nu \nabla^2 \mathbf{u} \\
    % Continuity Equation
    \nabla \cdot \mathbf{u} &= 0 \\
    % Thermodynamic Equation (Equivalent Potential Temp)
    \frac{\partial \theta_e}{\partial t} + \nabla \cdot (\mathbf{u} \theta_e) &= \mu_\theta \nabla^2 \theta_e + F_{\theta} \\
    % Moisture Equation (Total Water)
    \frac{\partial q_t}{\partial t} + \nabla \cdot (\mathbf{u} q_t) &= \frac{\partial}{\partial z} (V_T q_r) + \mu_q \nabla^2 q_t + F_{q}
\end{align}
