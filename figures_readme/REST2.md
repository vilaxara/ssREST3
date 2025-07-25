# Replica exchange with solute scaling (REST2)

## What is REST2?

Hamiltonian of the system is scaled by a factor of $\lambda$ in REST2 as shown below :


$$
E_{n}^{REST2}(X_{n}) = \lambda_{n}^{pp} E_{pp}(X_{n}) + \lambda_{n}^{pw} E_{pw}(X_{n}) + \lambda_{n}^{ww} E_{ww} (X_{n})
$$


Where, 
- $X_{n}\hspace{10mm}$ : Positions of particles.
- $E_{n}^{REST2}\hspace{3.5mm}$ : Scaled Hamiltonian of the system describing potential energy.
- $E_{pp}\hspace{9.5mm}$ : Hamiltonian contribution from solute-solute interactions.
- $E_{pw}\hspace{9mm}$ : Hamiltonian contribution from solute-solvent interactions.
- $E_{ww}\hspace{8.5mm}$ : Hamiltonian contribution from solvent-solvent interactions.


$\lambda_{n}$ is the scaling factor which scales the respective energy contributions as shown below :  


$$
\lambda_{n}^{pp}=\frac{\beta_{n}}{\beta_{0}} \ ; \quad \lambda_{n}^{pw}=\sqrt{\frac{\beta_{n}}{\beta_{0}}} \ ; \quad \lambda_{n}^{ww}=1; 
$$

where as,

$$
\beta_{0}=\frac{1}{k_{B}T_{0}}
$$

Where,
- $T_{0}\hspace{11mm}$ : Temperature of the base or $zeroth$ replica.
- $k_{B}\hspace{10.5mm}$ : Boltzmann constant.
- $\lambda_{n}^{pw}\hspace{9.5mm}$ : Scaling factor for protein:protein energy contributions.
- $\lambda_{n}^{pp}\hspace{10mm}$ : Scaling factor for protein:solvent energy contributions.
- $\lambda_{n}^{ww}\hspace{9mm}$ : Scaling factor for solvent:solvent energy contributions.

Because of this our solute/protein of interest will be influenced by the effective temperature $T_{n}$ while the whole simulation ensemble will be at the base temperature $T_{0}$.
