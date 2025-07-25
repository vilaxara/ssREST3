# Solvent-scaled Replica exchange with solute scaling (ssREST3)

## Where did the solvent scaling come from?

<!-- To circumvent the collapse of polymer kind of proteins while using Replica Exchange with Solute Scaling(REST2) protein-solvent interactions of the simulation are scaled.  -->

As we know Hamiltonian of the system is scaled by a factor of $\lambda$ in REST2 as shown below :


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

To scale the solute-solvent interctions we use another scaling factor $\kappa$ whose value ranges from $1.00 - 1.10$ depending on the number of amino acids of the protien system, structural propensity of the protein etc. In the bio-molecular simuations this is achived by scaling the non-bonded interactions between solute and solvent. By tweaking the LJ parameters of the solvent, in our case tip4p-*disp* water oxygen atom $\epsilon_{OW}$ as shown :  

$$
\large
\epsilon_{OW}^{rescaled} = \kappa_{n}^{2} * \epsilon_{OW}
$$

Where,
- $\epsilon_{OW}\hspace{8.5mm}$ : Unsaled LJ parameter of solvent oxygen atom.
- $\epsilon_{OW}^{rescaled}\hspace{4mm}$ : Scaled LJ parameter of solvent oxygen atom.
- $\kappa_{n}\hspace{11mm}$ : Scaling factor used to increase the solvent properties of $n^{th}$ replica to feel the temperature $T_{n} \ge T_{0}$.

As most of the bio molecular topology files for gromacs uses type-2 combination rules to compute LJ interactions between two particles, by scaling the potential depth of heavy atoms  <p1 style="color: red;">(Is the potential depth good term for epsilon and should we be tweaking heavy atoms only for non-water solvents to be enhanced?)</p1> of the solvent by $\kappa^{2}$, $\epsilon_{protein:solvent}$ was will be scaled to $\kappa$. The scaling factor was exponetially fitted between 1.0 and 1.10 for the respective repica. For $n^{th}$ replica it is as shown below:

$$\kappa_{n} = \kappa_{low} * \exp{ \biggl(n* \frac{\log(\kappa_{high}/\kappa_{low})}{N_{r}-1} \biggr) } \ ; \quad 1.00 \leq \kappa_{n} \leq 1.10$$

$$
\epsilon_{p:OW}=(\epsilon_{p:p}^{rescaled} * \epsilon_{OW:OW}^{rescaled})^{\frac{1}{2}}=\lambda_{n}^{pw}\kappa_{n}*(\epsilon_{p:p} * \epsilon_{OW:OW})^{\frac{1}{2}}
$$

Where,
- $\kappa_{high}\hspace{7mm}$ : Maximum value of scaling factor. (1.10 in our case)
- $\kappa_{low}\hspace{8mm}$ : minimum value of scaling factor = 1.0.
- $N_{r}\hspace{10mm}$ : Number of replicas.
- $n\hspace{12mm}$ : Replia number.
- $\epsilon_{p:OW}\hspace{6mm}$ : Protein:water LJ interaction 
- $\epsilon_{p:p}\hspace{9.5mm}$ : Protein:protein self interaction  
- $\epsilon_{OW:OW}\hspace{3mm}$ : water:water self interaction

By tweaking the solvent properties we ensure that the protein of interest will not stay colapse at higher temperatures during . **We made sure solvent self-interactions and solvent:ions intercations are unscaled by mentioning the respective values in the `[non-bonded]` parameters section of the toplogy files.** 

## How to implement ssREST3 for GROMACS?

### Required files :

- processed.top : This file can be obtained by running the following gromacs command :: `gmx grompp -f *mdp -p *top -c *gro -r *gro -o temporary.tpr -pp`
- ssREST3.py

In the python script we implemented, processed gromacs topology file(`processed.top`) was used as input. User can either implement REST2 or ssREST3 enhanced sampling scheme by switching `kappa` on or off. When `kappa` is off, the script implements conventional REST2 where as when it is on solvent scaling is implemented along with REST2.
