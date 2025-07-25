# %%
import numpy as np
import sys
import os
sys.path.insert(0, '/dartfs-hpc/rc/home/0/f006f50/labhome/JKrishna/scripts')
from importlib import *


# %%
import structure_analysis
reload(structure_analysis)

# %%
ligand_rings = { 'fas' :  [[304,305,306,307,308,309,310,311,312,313]], 'lig47' : [[307, 308, 309, 311, 313, 314]]}
lig_hbond_donors= {'fas' : [[296,318],[296,331]], 'lig47' : [[329,330],[329,331]]}
#Ligand_Pos_Charges={'fas': [329], 'lig47': [296] }
Ligand_Pos_Charges={'fas': [296], 'lig47': [329] }
Ligand_Neg_Charges={ 'fas': [], 'lig47': []}

xtc_n=sys.argv[1]

# %%
print(os.getcwd())

# %%
w_dir=os.getcwd()

# %%

dir_t1=f'{w_dir}/xtc'

nreps=1
colvar_file=f'{w_dir}/colvar/colvar_{xtc_n}.dat'
out_dir=f'./out_{xtc_n}/'
tpr=None
t_in=f'pbc_{xtc_n}'

pbc={'pbc' : True, 'traj_tag_in' : t_in ,'traj_tag_out' : None, 'traj_type' : 'xtc'}


pdb=f'{w_dir}/prot_lig.pdb'

# helix='/data/Jaya_Krishna/analysis_trajectories/async/rest/asyn_sa.pdb'
# pp='/data/Jaya_Krishna/analysis_trajectories/async/rest/asyn_pp.pdb'

lig_list=['fas', 'lig47']
lig=lig_list[1]

# analysis_l=[['kd' , 'charge' , 'aro', 'hyphob'], ['kd' , 'hbond'],['l_cm'],['p_cm']]
analysis_l=[['l_cm']]

for l in analysis_l :
        
    A = structure_analysis.ligand_interactions(rep_trajectory=dir_t1, 
                                                        out_dir=out_dir,
                                                        pbc_tpr=tpr ,
                                                        nreps=nreps,
                                                        pdb=pdb, 
                                                        # helix_pdb=helix, 
                                                        # pp_pdb=pp, 
                                                        ligand_rings=ligand_rings[lig], 
                                                        ligand_residue_index=20, 
                                                        prot_end_res_num=19, 
                                                        prot_start_res_num=0, 
                                                        stride=1, 
                                                        offset=121, 
                                                        ligand_positive_charge_atoms=Ligand_Pos_Charges[lig], 
                                                        ligand_negative_charge_atoms=Ligand_Neg_Charges[lig], 
                                                        lig_hbond_donors=lig_hbond_donors[lig],
                                                        rep_demux_switch="off",
                                                        # trj_break=2,
                                                        apo=False,
                                                        pbc=pbc,
                                                        analysis_list=l,
                                                        colvar_file=colvar_file)


# %%



