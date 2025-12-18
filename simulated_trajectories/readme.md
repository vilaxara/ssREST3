## Overview

Strided trajectories of REST2 and ssREST3 simulations are presented here. One can extract them using the following command which creates a directory `trajectories`:

`cat trajectories.tar.xz.part-* | tar -xJf -`

This directory contains the following folder structure:

<details>

<summary>Trajectories</summary>

```bash
.
├── pbc_apo_ions.pdb
├── pbc_apo.pdb
├── pbc_fasudil_ions.pdb
├── pbc_fasudil.pdb
├── pbc_lig47_ions.pdb
├── pbc_lig47.pdb
├── REST2
│   ├── apo_10
│   │   ├── demux_replica
│   │   └── temperature_replica
│   ├── apo_20
│   │   ├── demux_replica
│   │   └── temperature_replica
│   ├── fas_10
│   │   ├── demux_replica
│   │   └── temperature_replica
│   ├── fas_20
│   │   ├── demux_replica
│   │   └── temperature_replica
│   ├── lig47_10
│   │   ├── demux_replica
│   │   └── temperature_replica
│   └── lig47_20
│       ├── demux_replica
│       └── temperature_replica
└── ssREST3
    ├── apo_16
    │   ├── demux_replica
    │   └── temperature_replica
    ├── apo_8
    │   ├── demux_replica
    │   └── temperature_replica
    ├── apo_8_1
    │   ├── demux_replica
    │   └── temperature_replica
    ├── fas_16
    │   ├── demux_replica
    │   └── temperature_replica
    ├── fas_8
    │   ├── demux_replica
    │   └── temperature_replica
    ├── lig47_16
    │   ├── demux_replica
    │   └── temperature_replica
    └── lig47_8
        ├── demux_replica
        └── temperature_replica
```
