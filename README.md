# Solvent-scaled-REST3

This repository contains the files used for solvent scaled REST3 (ssREST3). The periodic boundary corrected [trajectories](https://dartmouth-my.sharepoint.com/:f:/g/personal/f006f50_dartmouth_edu/Eg2RMLY6NJVPsXMUqPKVceIBsh5mVyRxki1qnTEFOwXq-w?e=6moAve) and the [data](https://dartmouth-my.sharepoint.com/:f:/r/personal/f006f50_dartmouth_edu/Documents/ssREST3_data/json_data?csf=1&web=1&e=dxTG8N) for all the simulations addressed in the article are available. We explain the [concept of ssREST3](./ssREST3.ipynb) and how it is an optimize variant of REST2 which helps the polymer kind of proteins like IDP's to stay elongated at higher temperatures. 

## Methodology
What is REST2 : [jump in](./figures_readme/REST2.md)<br>
Then, what is ssREST3 : [to infinity and beyond](./figures_readme/ssREST3.md)

### Overview of setup
Our protein of interest is simulated in a cubic box solvated with water and ions enough to neutralize the system. All the equilibration procedures and checks are explained in the ssREST3 article <p1 style="color: red;">(link for ssREST3 paper)</p1>. A brief overview of the simulations sets and parameters used in the current simulation is : 

- **Protein of interest** : $\alpha$-synuclien.
- __Small molecules/Ligands__ : Fasudil, Ligand-47[^2]
- **Force field** : amber99sb-*disp*
- **Water** : tip4p-*disp* [^1]
- **MD engine** : [GROMACS-2022](https://www.gromacs.org/) patched with [PLUMED-2.8](https://www.plumed.org/)
- **Simulation box type** : Cubic box length of 6.5nm  
- **Enhanced sampling techniques used** : REST2[^3], Parallel-bias Metadynamics[^4], ssREST3
<!-- ![TOF](./figures_readme/figure_a_blend.wy.png) -->
<!-- <img src="./figures_readme/figure_a_blend.wy.png" width="700" height="400" /> -->

<div align="center">
  <img src="./figures_readme/figure_a_blend.wy.png" width="70%" />
</div>
<p align=center>
  Alpha-synuclein solvated in water. Water shell around the protein is shown in the figure.
</p>

<!-- <figure align=center>
  <img src="./figures_readme/figure_a_blend.wy.png" width="700" height="400" alt="protein-water"/>
  <figcaption>Alpha-synuclein solvated in water. Water shell around the protein is shown in the figure.</figcaption>
</figure> -->

We simulate $\alpha$-synuclein in presence and absence of the small molecules. After they are simulated for sufficient amount of time we perform basic analysis on the pbc corrected trajectories.

### File tree

<details>

<summary>File Tree</summary>

```bash
.
├── Apo
│   └── readme.md
├── Fasudil
│   ├── R2_fas_10reps.ipynb
│   ├── R2_fas_20reps.ipynb
│   ├── R3_fas_16reps.ipynb
│   ├── R3_fas_8reps.ipynb
│   ├── agg_rg_fas_1.ipynb
│   ├── agg_rg_fas_2.ipynb
│   ├── readme.md
│   └── temp_plots_fas_10.ipynb
├── Ligand47
│   ├── R2_lig47_10reps.ipynb
│   ├── R2_lig47_20reps.ipynb
│   ├── R3_lig47_16reps.ipynb
│   ├── R3_lig47_8reps.ipynb
│   ├── agg_rg_lig47_1.ipynb
│   ├── agg_rg_lig47_2.ipynb
│   └── readme.md
├── README.md
├── analysis_PBMetad.py
├── analysis_R2.ipynb
├── analysis_apo.ipynb
├── analysis_ssR3.ipynb
├── comparision_plots.ipynb
├── figures_readme
│   ├── REST2.md
│   ├── figure_a_blend.wy.png
│   └── ssREST3.md
├── q_bootstrap_fas.npy
├── q_bootstrap_lig4.npy
└── scripts
    ├── Block_analysis.py
    ├── __init__.py
    ├── __pycache__
    │   ├── Block_analysis.cpython-311.pyc
    │   ├── plot.cpython-311.pyc
    │   ├── plot_external_def.cpython-311.pyc
    │   ├── plot_old.cpython-311.pyc
    │   ├── small_utilities.cpython-311.pyc
    │   └── structure_analysis_changes.cpython-311.pyc
    ├── plot.py
    ├── plot_def.py
    ├── plot_external_def.py
    ├── plot_old.py
    ├── readme.md
    ├── small_utilities.py
    ├── structure_analysis_changes.py
    └── temp_plots_fas_10.ipynb
```
## Authors
- [Jaya Krishna K](https://github.com/vilaxara)
- [Korey Reed](https://github.com/koreyr)
- [Paul Robustelli](https://github.com/paulrobustelli)

## References

[^1]: P. Robustelli, S. Piana, D.E. Shaw, Developing a molecular dynamics force field for both folded and disordered protein states, Proc. Natl. Acad. Sci. U.S.A., 115 (21) E4758-E4766, https://doi.org/10.1073/pnas.1800690115 (2018).
[^2]: Paul Robustelli, Alain Ibanez-de-Opakua, Cecily Campbell-Bezat, Fabrizio Giordanetto, Stefan Becker, Markus Zweckstetter, Albert C. Pan, and David E. Shaw, Molecular Basis of Small-Molecule Binding to α-Synuclein, Journal of the American Chemical Society 144 (6), 2501-2510, [DOI: 10.1021/jacs.1c07591](https://doi.org/10.1021/jacs.1c07591) (2022)
[^3]: Lingle Wang, Richard A. Friesner, and B. J. Berne, Replica Exchange with Solute Scaling: A More Efficient Version of Replica Exchange with Solute Tempering (REST2), The Journal of Physical Chemistry B 115 (30), 9431-9438, [DOI: 10.1021/jp204407d](https://doi.org/10.1021/jp204407d) (2011)
[^4]: Jim Pfaendtner and Massimiliano Bonomi, Efficient Sampling of High-Dimensional Free-Energy Landscapes with Parallel Bias Metadynamics, Journal of Chemical Theory and Computation 11 (11), 5062-5067, [DOI: 10.1021/acs.jctc.5b00846](https://doi.org/10.1021/acs.jctc.5b00846) (2015)

