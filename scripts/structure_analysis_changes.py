# Class for computing interatons and structural properties

import mdtraj as md
from mdtraj.utils import ensure_type
from mdtraj.geometry import compute_distances, compute_angles
from mdtraj.geometry import _geometry
import os
import sys
import numpy as np
import scipy as sp
from scipy import optimize
from scipy.optimize import leastsq
import math
from numpy import log2, zeros, mean, var, sum, loadtxt, arange, array, cumsum, dot, transpose, diagonal, floor
from numpy.linalg import inv, lstsq
import pyblock
from numpy import copy
from multiprocessing import Pool
import multiprocessing
import time

import json

import MDAnalysis as mda
from MDAnalysis import transformations

from Block_analysis import *

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    
def json_dump(out_dir:str, json_in:dict, json_out:str, flag:str, indent:int=1):

    if flag == 'w':

        with open(f"{out_dir}/{json_out}", 'w') as file_:
            json.dump(json_in, file_,cls=NpEncoder, indent=indent)

    elif flag == 'a':

        with open(f"{out_dir}/{json_out}",'r') as file_in:
            data = json.load(file_in)

        # data.update(json_in)

        for key, value in json_in.items():
            if key not in data:
                data[key]=value

        with open(f"{out_dir}/{json_out}", 'w') as file_out:
            json.dump(data, file_out,cls=NpEncoder, indent=indent)

# def random_quote():

#     from quoters import Quote
#     print(f'REMEMBER : {Quote.print()}')
        
def convert_to_array(x):

    if isinstance(x, np.ndarray):
        return x
    
    elif isinstance(x, list) :
        return np.array(x)

    elif isinstance(x, (int, float, str)) :
        return np.array([x])
    
    elif isinstance(x, tuple) :
        return np.array(x)

def center_wrap(top:str, traj:str, out_file:str):
    u = mda.Universe(top,traj)
    protein = u.select_atoms('protein')
    system = u.select_atoms('all')
    not_protein = u.select_atoms('not protein')
    transforms = [transformations.unwrap(system), \
            transformations.center_in_box(protein, wrap=False),\
            transformations.wrap(not_protein,compound='residues')]
    u.trajectory.add_transformations(*transforms)
    with mda.Writer(out_file, system.n_atoms) as W:
        for ts in u.trajectory:
            W.write(system)

    del u


def load_traj(args_dict:dict) :

    args_dict_in = {'trajectory':None, 'pdb':None, 'stride':1, 'p_sel':None}
    
    for key, value in args_dict.items() :

        if key in args_dict_in.keys() : 

            args_dict_in[key] = value

    a = md.load(args_dict_in['trajectory'], top=args_dict_in['pdb'], stride=args_dict_in['stride'])
    a = a.atom_slice(a.topology.select(args_dict_in['p_sel']))
    a.center_coordinates()

    return a

def wrap_traj(args_dict:dict) :

    args_dict_in = {'trajectory':None, 'out_dir':None, 'tpr':None, 'traj_num':None, 'out_file_tag':'pbc_', 'out_file_fmt':'xtc' }
    
    for key, value in args_dict.items() :

        if key in args_dict_in.keys() : 

            args_dict_in[key] = value

    out_file=f"{args_dict_in['out_dir']}/{args_dict_in['out_file_tag']}{args_dict_in['traj_num']}.{args_dict_in['out_file_fmt']}"
    center_wrap(args_dict_in['tpr'], args_dict_in['trajectory'], out_file)

def load_traj_p(args_dict:dict) :

    args_dict_in = {'trajectory':None, 'pdb':None, 'stride':1, 'p_sel':None, 'sel':None}
    
    for key, value in args_dict.items() :

        if key in args_dict_in.keys() : 

            args_dict_in[key] = value 
    
    b=md.load(args_dict_in['trajectory'], top=args_dict_in['pdb'], stride=args_dict_in['stride'])
    prot=b.topology.select(args_dict_in['sel'])
    b = b.atom_slice(b.topology.select(args_dict_in['p_sel']))
    b.restrict_atoms(prot)
    b.center_coordinates()

    return b

def restrict_atoms(args_dict:dict):

    import copy

    args_dict_in = {'trajectory':None, 'selection' : None }

    for key, value in args_dict.items() :

        if key in args_dict_in.keys() : 

            args_dict_in[key] = value

    t_out = copy.deepcopy(args_dict_in['trajectory'])
    t_out.restrict_atoms(t_out.top.select(args_dict_in['selection']))
    t_out.center_coordinates()

    return t_out

def norm_weights(file_name):

    colvar=np.loadtxt(file_name,comments=['#','@'])
    num_cvs=len(colvar[0])-1

    kt=2.494339
    w=np.exp((colvar[:,num_cvs]/kt))

    max_=np.sum(w)
    w_norm=w/max_

    return num_cvs, w_norm



class initial():
    
    def __init__(self, rep_trajectory:str, pdb:str, prot_start_res_num:int, prot_end_res_num:int, out_dir:str, analysis_list:list=[], pbc_tpr:str = None, helix_pdb:str = None, pp_pdb:str = None, ligand_rings:list = [], ligand_residue_index:int = None ,nreps:int = 1, stride:int =1, offset:int = 0,
                 ligand_positive_charge_atoms:list = [], ligand_negative_charge_atoms:list = [], lig_hbond_donors:list=[], colvar_file:str = None, pbc:dict = {'pbc' : True, 'traj_in_tag' : 'prod_','traj_out_tag' : 'pbc_', 'traj_type' : 'xtc'}, trj_break:int=8, rep_demux_switch:str = "off", demux_trajectory:str=None, apo:bool=False, **kwargs):

        self.analysis_dict={ 'salpha_rmsd' : False, 'pp_rmsd' : False, 'rg' : False, 'ss' : False, 'ba' : False, 'kd' : False, 'charge' : False, 'aro' : False, 'hyphob' : False, 'hbond' : False, 'p_cm' : False, 'l_cm' : False, 'kd_bf_time' : False}

        self.out_file_dict =  {'bf' : 'bound_fraction.json' , 'kd' : 'kd.json', 'kd_bf_time' : 'kd_bf_time.json', 'sa' : 'sa.json', 'pp_rmsd' : 'pp.json' , 'rg' : 'rg.json', 'aright' : 'alphabeta_alpharight.json','sheet' :  'alphabeta_betasheet.json',
                              'pp' : 'alphabeta_ppII.json', 'ba' : 'bend_angle.json', 'ss_h' : 'helix_contant.json', 'ss_s' : 'sheet_contant.json',
                              'lc' :  'avg_ligand_contacts.json', 'lc_rw' : 'avg_ligand_contacts_rw.json', 'cm' : 'contact_matrix.json',
                              'charge:all' : 'charge_contact_fraction.json', 'charge_rw:all' :  'charge_re.json', 'charge_rw:bf' :  'charge_re_bf.json', 'charge:bf' : 'charge_contact_fraction_bf.json',
                              'aro:all' : 'aro_interactions.json', 'aro_rw:all' :  'aro_interactions_re.json', 'aro_rw:bf' :  'aro_interactions_re_bf.json', 'aro:bf' : 'aro_interactions_bf.json', 'aro:binary' : 'aro_binary.json',
                              'hyph:all' : 'hyphob_interactions.json', 'hyph_rw:all' :  'hyphob_interactions_re.json', 'hyph_rw:bf' :  'hyphob_interactions_re_bf.json', 'hyph:bf' : 'hyphob_interactions_bf.json', 'hyph:binary' : 'hyphob_binary.json',
                              'hb:all' : 'hbond_interactions.json', 'hb_rw:all' :  'hbond_interactions_re.json', 'hb_rw:bf' :  'hbond_interactions_re_bf.json', 'hb:bf' : 'hbond_interactions_bf.json', 'hb:binary' : 'hbond_binary.json',
                              'p_cm' : 'p_contact_map.json', 'p_cm_rw' : 'p_contact_map_re.json', 'p_cd' : 'p_contact_distance_map.json', 'p_cd' : 'p_contact_distance_map_re.json', 'l_cm' : 'l_contact_map.json', 'l_cm_rw' : 'l_contact_map_re.json'}
        
        for i in analysis_list : 

            if i in self.analysis_dict.keys(): self.analysis_dict[i] = True

        self.trj_brk=trj_break

        self.pdb_p = pdb
        self.pdb = md.load(pdb)

        self.stride = stride

        self.pbc = pbc['pbc']
        self.traj_tag_in, self.traj_tag_out, self.traj_type = pbc['traj_tag_in'], pbc['traj_tag_out'], pbc['traj_type']

        if rep_demux_switch=="on": self.rest=True
        elif rep_demux_switch=="off": self.rest=False

        self.apo = apo
        self.pbc_tpr = pbc_tpr
        self.out_dir=out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.residue_offset = offset
        self.ligand_rings = ligand_rings

        self.weights = []

        if colvar_file :

            self.w=True
            self.num_cvs, self.weights = norm_weights(colvar_file)
            self.weights = self.weights[::stride]

            print("No of CV's :", self.num_cvs-1)
            print(f"Sum of combined xtc : {np.sum(self.weights)}\n")

            self.out_file_dict['kd_raw']='kd_raw.json'

        else: 
            self.w=False

        self.Ligand_Pos_Charges = ligand_positive_charge_atoms
        self.Ligand_Neg_Charges = ligand_negative_charge_atoms
        self.lig_hbond_donors = lig_hbond_donors
        self.ligand_residue_index = ligand_residue_index
        self.prot_start_res_num = prot_start_res_num
        self.prot_end_res_num = prot_end_res_num

        assert (self.prot_end_res_num>0) & (self.prot_end_res_num>0) , "Give positive protein start and ending residue numbers starting from 0!"
        self.prot_res_range = convert_to_array([i for i in range(self.prot_start_res_num+self.residue_offset, self.prot_end_res_num+self.residue_offset+1)])

        self.nreps = nreps
        assert self.nreps > 0 , "Number of replicas should be > 0; If only one give nreps=1 !"

        if helix_pdb : 
            self.helix = md.load(helix_pdb)
            sel=f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"
            self.helix.restrict_atoms(self.helix.topology.select(sel))
            self.helix.center_coordinates()
        if pp_pdb:
            self.pp = md.load(pp_pdb)
            self.pp.restrict_atoms(self.pp.topology.select(sel))
            self.pp.center_coordinates()

        self.trj_dict={}
        self.p_trj={}
        self.n_frames={}
        self.simulation_time={}

        # p_sel=f"resid {self.prot_start_res_num} to {self.ligand_residue_index}"

        print("Initiating memory...")

        if not self.rest :

            if not self.apo : self.p_sel=f"resid {self.prot_start_res_num} to {self.ligand_residue_index}"
            elif self.apo : self.p_sel=f"resid {self.prot_start_res_num} to {self.prot_end_res_num}" 

            self.sel=f"resid {self.prot_start_res_num} to {self.prot_end_res_num}"

            if self.pbc : 
                
                if self.nreps==1 : 
                
                    if os.path.exists(f"{rep_trajectory}/{self.traj_tag_in}.{self.traj_type}") : self.trj_dict[self.nreps-1]=f"{rep_trajectory}/{self.traj_tag_in}.{self.traj_type}"
                    else : print(f"{self.traj_tag_in}.{self.traj_type} is missing...\n")
                
                else:

                    for i in range(self.nreps):

                        if os.path.exists(f"{rep_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}") : trajectory=f"{rep_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}"
                        elif os.path.exists(f"{rep_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}") : trajectory=f"{rep_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}"
                        else : print(f"{self.traj_tag_in} {self.traj_type} is missing...\n")

                        self.trj_dict[i] = trajectory

                    # print(f"Loading trajectory {i}!...\n")

                    # self.trj[i] = load_traj(args_dict={'trajectory':trajectory,
                    #                         'pdb':pdb,
                    #                         'stride':stride, 
                    #                         'p_sel':p_sel})                    

                    # self.p_trj[i] = load_traj_p(args_dict={'trajectory':trajectory,
                    #                         'pdb':pdb,
                    #                         'stride':stride,
                    #                         'p_sel':p_sel,
                    #                         'sel':sel})
                    
                    # self.n_frames[i] = self.trj[i].n_frames
                    # self.simulation_time[i] = self.trj[i].timestep * self.n_frames[i]/(10**6)

            elif not self.pbc : 
                    
                # pool = Pool(processes=min(self.trj_brk, multiprocessing.cpu_count()))
                    
                if self.nreps == 1 : 
                    if os.path.exists(f"{rep_trajectory}/{self.traj_tag_in}.{self.traj_type}") : self.trj_dict[self.nreps-1]=f"{rep_trajectory}/{self.traj_tag_in}.{self.traj_type}"
                    else : print(f"{self.traj_tag_in}.{self.traj_type} is missing...\n")

                else :

                    for i in range(self.nreps):
                        if os.path.exists(f"{rep_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}") : trajectory=f"{rep_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}" 
                        elif os.path.exists(f"{rep_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}") : trajectory=f"{rep_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}"
                        else : print(f"{self.traj_tag_in} {self.traj_type} is missing...\n")



                        args_dict={'trajectory':trajectory,
                                    'pdb':pdb,
                                    'out_dir':rep_trajectory,
                                    'tpr':self.pbc_tpr, 
                                    'traj_num':i,
                                    'out_file_tag':self.traj_tag_out,
                                    'out_file_fmt':self.traj_type}

                        print(f"Wrapping trajectory {i}!...\n")

                        # pool.apply_async(wrap_traj, args=args_dict)
                        wrap_traj(args_dict)

                # pool.close()
                # pool.join()

                if self.nreps == 1 :  self.trj_dict[self.nreps-1]=f"{rep_trajectory}/{self.traj_tag_out}.{self.traj_type}"
                else : 
                    
                    for i in range(self.nreps):
                        trajectory=f"{rep_trajectory}/{self.traj_tag_out}{i}.{self.traj_type}"

                        self.trj_dict[i]=trajectory

                    # if i == 0 : trajectory=f"{rep_trajectory}/{self.traj_tag_out}.{self.traj_type}"
                    # else : trajectory=f"{rep_trajectory}/{self.traj_tag_out}{i}.{self.traj_type}"


                    # print(f"Loading trajectory {i}!...\n")

                    # self.trj[i] = load_traj(args_dict={'trajectory':trajectory,
                    #                         'pdb':pdb,
                    #                         'stride':stride, 
                    #                         'p_sel':p_sel})
                
                    # self.p_trj[i] = load_traj_p(args_dict={'trajectory':trajectory,
                    #                         'pdb':pdb,
                    #                         'stride':stride,
                    #                         'p_sel':p_sel,
                    #                         'sel':sel})

                    # self.n_frames[i] = self.trj[i].n_frames
                    # self.simulation_time[i] = self.trj[i].timestep * self.n_frames[i]/(10**6)
                    
        elif self.rest :

            if not self.apo : self.p_sel=f"resid {self.prot_start_res_num} to {self.ligand_residue_index}"
            elif self.apo : self.p_sel=f"resid {self.prot_start_res_num} to {self.prot_end_res_num}" 

            self.sel=f"resid {self.prot_start_res_num} to {self.prot_end_res_num}"
            
            for trj_type in ["rep", "demux"]:
                
                if self.pbc : 

                    for i in range(self.nreps):

                        if trj_type=="rep" : 
                            if os.path.exists(f"{rep_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}") : trajectory=f"{rep_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}"
                            elif os.path.exists(f"{rep_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}") : trajectory=f"{rep_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}"
                            else : print(f"{self.traj_tag_in} {self.traj_type} is missing...\n")

                        elif trj_type=="demux" : 
                            if os.path.exists(f"{demux_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}") : trajectory=f"{demux_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}"
                            elif os.path.exists(f"{demux_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}") : trajectory=f"{demux_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}"
                            else : print(f"{self.traj_tag_in} {self.traj_type} is missing...\n")

                        k=f"{trj_type}:{i}"

                        self.trj_dict[k] = trajectory
                        # print(f"Loading {trj_type} trajectory {i}!...\n")

                        # self.trj[k] = load_traj(args_dict={'trajectory':trajectory,
                        #                         'pdb':pdb,
                        #                         'stride':stride, 
                        #                         'p_sel':p_sel})                    

                        # self.p_trj[k] = load_traj_p(args_dict={'trajectory':trajectory,
                        #                         'pdb':pdb,
                        #                         'stride':stride,
                        #                         'p_sel':p_sel,
                        #                         'sel':sel})
                        
                        # self.n_frames[k] = self.trj[k].n_frames
                        # self.simulation_time[k] = self.trj[k].timestep * self.n_frames[k]/(10**6)


                elif not self.pbc :

                    # pool = Pool(processes=min(self.trj_brk, multiprocessing.cpu_count()))

                    for i in range(self.nreps):


                        if trj_type=="rep" : 
                            if os.path.exists(f"{rep_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}") : trajectory=f"{rep_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}" ; out_dir=rep_trajectory
                            elif os.path.exists(f"{rep_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}") : trajectory=f"{rep_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}" ; out_dir=rep_trajectory
                            else : print(f"{self.traj_tag_in} {self.traj_type} is missing...\n")

                        elif trj_type=="demux" : 
                            if os.path.exists(f"{demux_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}") : trajectory=f"{demux_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}" ; out_dir=demux_trajectory
                            elif os.path.exists(f"{demux_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}") : trajectory=f"{demux_trajectory}/{i}{self.traj_tag_in}.{self.traj_type}" ; out_dir=demux_trajectory
                            else : print(f"{self.traj_tag_in} {self.traj_type} is missing...\n")

                        # if trj_type=="rep" : trajectory=f"{rep_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}" ; out_dir=rep_trajectory
                        # elif trj_type=="demux" : trajectory=f"{demux_trajectory}/{self.traj_tag_in}{i}.{self.traj_type}" ; out_dir = demux_trajectory

                        args_dict={'trajectory':trajectory,
                                    'pdb':pdb,
                                    'out_dir':out_dir,
                                    'tpr':self.pbc_tpr, 
                                    'traj_num':i,
                                    'out_file_tag':self.traj_tag_out,
                                    'out_file_fmt':self.traj_type}


                        print(f"Wrapping {trj_type} trajectory {i}!...\n")

                        # pool.apply_async(wrap_traj, args=args_dict)
                        wrap_traj(args_dict)

                    # pool.close()
                    # pool.join()

                    for i in range(self.nreps):

                        if trj_type=="rep" : trajectory=f"{rep_trajectory}/{self.traj_tag_out}{i}.{self.traj_type}"
                        elif trj_type=="demux" : trajectory=f"{demux_trajectory}/{self.traj_tag_out}{i}.{self.traj_type}"

                        k=f"{trj_type}:{i}"

                        self.trj_dict[k] = trajectory

                        # print(f"Loading {trj_type} trajectory {i}!...\n")

                        # self.trj[k] = load_traj(args_dict={'trajectory':trajectory,
                        #                         'pdb':pdb,
                        #                         'stride':stride, 
                        #                         'p_sel':p_sel})
                    
                        # self.p_trj[k] = load_traj_p(args_dict={'trajectory':trajectory,
                        #                         'pdb':pdb,
                        #                         'stride':stride,
                        #                         'p_sel':p_sel,
                        #                         'sel':sel})

                        # self.n_frames[k] = self.trj[k].n_frames
                        # self.simulation_time[k] = self.trj[k].timestep * self.n_frames[k]/(10**6)

        self.sa = {}
        self.pp_rmsd={}
        self.rg = {}
        self.alphabeta_alpharight = {} 
        self.alphabeta_betasheet = {} 
        self.alphabeta_ppII = {}
        self.bend_angle = {}
        self.helix_contant = {}
        self.sheet_contant = {}

        self.contact_matrix = {}
        self.average_ligand_contacts = {}
        self.average_ligand_contacts_bf = {}
        self.average_ligand_contacts_rw = {}
        self.average_ligand_contacts_rw_bf = {}
        self.bound_fraction = {}
        self.kd = {}
        self.kd_bf_time = {}

        self.charge_fraction = {}
        self.charge_fraction_bf = {}
        self.charge_re = {}
        self.charge_re_bf = {}

        self.aro_binary_contacts = {}
        self.aro_interactions={}
        self.aro_interactions_bf={}
        self.aro_interactions_re={}
        self.aro_interactions_re_bf={}
        self.stackparams = {}

        self.hydro_binary={}
        self.hydro_interactions={}
        self.hydro_interactions_bf={}
        self.hydro_interactions_re={}
        self.hydro_interactions_re_bf={}

        self.hbond_binary={}
        self.hbond_interactions={}
        self.hbond_interactions_bf={}
        self.hbond_interactions_re={}
        self.hbond_interactions_re_bf={}

        self.p_contact_map = {}
        self.p_contact_distance_map = {}
        self.l_contact_map = {}

        # self.num_residues = self.pdb.topology.n_residues
        self.temp = self.pdb.atom_slice(self.pdb.topology.select(self.p_sel))
        self.num_residues = self.temp.top.n_residues

        self.box_size={}

        # if self.rest :

            
        #     self.box_size = self.trj["rep:0"].unitcell_lengths[0][0]

        # elif not self.rest :

        #     self.box_size = self.trj[0].unitcell_lengths[0][0]


        # self.contacts = self.contact_matrix()
        # self.aromatic_interactions = self.aromatic()



    
######################################-Definations for computing structure parameters-###############################    

def calc_SA(trj, helix,  RMS_start, RMS_stop, r0:float=0.10):

    RMS = []
    for i in range(RMS_start, RMS_stop):
        sel = helix.topology.select("resid %s to %s and name CA" % (i, i+5))
        if len(sel) < 6 : break
        rmsd = md.rmsd(trj, helix, atom_indices=sel)
        RMS.append(rmsd)
    RMS = np.asarray(RMS)
    # Sa_sum = np.zeros((trj.n_frames))
    Sa = (1.0-(RMS/r0)**8)/(1-(RMS/r0)**12)

    return Sa

def calc_rg(trj):
    
    mass = []
    for at in trj.topology.atoms:
        mass.append(at.element.mass)
    mass_CA = len(mass)*[0.0]
    # put the CA entries equal to 1.0
    for i in trj.topology.select("name CA"):
        mass_CA[i] = 1.0
    # calculate CA radius of gyration
    rg_CA = md.compute_rg(trj, masses=np.array(mass_CA))
    
    return rg_CA


def calc_phipsi(trj):

    # Compute Phi and Psi
    indices_phi, phis = md.compute_phi(trj)
    indices_psi, psis = md.compute_psi(trj)
    
    # psis=psis.T[:19].T
    
    phi_label = []
    for i_phi in range(0, indices_phi.shape[0]):
        resindex = trj.topology.atom(indices_phi[i_phi][3]).residue.resSeq
        # resindex = trj.topology.atom(indices_phi[i_phi][2]).residue.resSeq

        phi_label.append(resindex)
    phi_label = np.array(phi_label)
    psi_label = []
    for i_psi in range(0, indices_psi.shape[0]):
        resindex = trj.topology.atom(indices_psi[i_psi][3]).residue.resSeq
        # resindex = trj.topology.atom(indices_psi[i_psi][2]).residue.resSeq

        psi_label.append(resindex)
    psi_label = np.array(psi_label)
    # psi_label = np.array(psi_label[:19])
    
    phipsi = []
    for i in range(0, len(psi_label)-1):
        current_phipsi = np.column_stack((phis[:, i+1], psis[:, i]))
        phipsi.append(current_phipsi)
    phipsi_array = np.array(phipsi)

    def alphabeta_rmsd(phi, psi, phi_ref, psi_ref):
        alphabetarmsd = np.sum(0.5*(1+np.cos(psi-psi_ref)),
                            axis=1)+np.sum(0.5*(1+np.cos(phi-phi_ref)), axis=1)
        return alphabetarmsd


    Phi_all = phis
    Psi_all = psis
    alphabeta_alpharight = alphabeta_rmsd(Phi_all, Psi_all, -1.05, -0.79)
    alphabeta_betasheet = alphabeta_rmsd(Phi_all, Psi_all, 2.36, -2.36)
    alphabeta_ppII = alphabeta_rmsd(Phi_all, Psi_all, -1.31, 2.71)
    
    return alphabeta_alpharight, alphabeta_betasheet, alphabeta_ppII

def vec_angles(trj, atom_indices:list):
    
    #Get xyz co-ordinates
    xyz=[]
    for  atom_idx in atom_indices :
        a=[]
        for frame_idx in range(trj.n_frames):

            a.append(trj.xyz[frame_idx, atom_idx,:].astype(float))
        xyz.append(a)    

    xyz=np.array(xyz)
    xyz.shape

    #Define vectors with 2nd atom as starting point
    V=[]
    v1=xyz[0]-xyz[1]
    v2=xyz[2]-xyz[1]
    V.append(v1)
    V.append(v2)

    #Compute angles between two vectors
    angles=[]
    for i in range(trj.n_frames):
    
        a=np.rad2deg(np.arccos(np.dot(V[0][i],V[1][i])/
                            (np.sqrt(np.dot(V[0][i],V[0][i])*np.dot(V[1][i],V[1][i])))))
        
        angles.append(a)
    
    angles=np.array(angles)
        
    return angles

def ave_angle(trj, res_id:list):
    atom_id=[]
    print(f"Will compute the average bend angle between the below protien residues. Please CHECK!")

    for j in res_id:
            
        sel="name CA and resid "+str(j)
        a=trj.topology.atom(int(trj.topology.select(sel)))
        atom_id.append(int(trj.topology.select(sel)))


        print(f"{a}, {int(trj.topology.select(sel))}")

    angle=vec_angles(trj, atom_id)
    out=get_blockerror_pyblock_nanskip(angle)

    print("\n")
    return [ out[0], out[1] ] 

def dssp_convert(dssp):
    dsspH = np.copy(dssp)
    dsspE = np.copy(dssp)
    dsspH[dsspH == 'H'] = 1
    dsspH[dsspH == 'E'] = 0
    dsspH[dsspH == 'C'] = 0
    dsspH[dsspH == 'NA'] = 0
    dsspH = dsspH.astype(int)
    TotalH = np.sum(dsspH, axis=1)
    SE_H = np.zeros((len(dssp[0]), 2))

    for i in range(0, len(dssp[0])):
        data = dsspH[:, i].astype(float)
        if(np.mean(data) > 0):
            SE_H[i] = [np.mean(data), (block(x=data))**.5]

    dsspE[dsspE == 'H'] = 0
    dsspE[dsspE == 'E'] = 1
    dsspE[dsspE == 'C'] = 0
    dsspE[dsspE == 'NA'] = 0
    dsspE = dsspE.astype(int)
    TotalE = np.sum(dsspE, axis=1)
    Eprop = np.sum(dsspE, axis=0).astype(float)/len(dsspE)
    SE_E = np.zeros((len(dssp[0]), 2))

    for i in range(0, len(dssp[0])):
        data = dsspE[:, i].astype(float)
        if(np.mean(data) > 0):
            SE_E[i] = [np.mean(data), (block(x=data))**.5]
    return SE_H, SE_E

def ss(trj):

    dssp=md.compute_dssp(trj, simplified=True)
    H1_H,H1_E=dssp_convert(dssp)
    
    return H1_H, H1_E


######################################-Class for computing ligand contacts and Kd-###############################    

def bound_frac_kd_rw_(box_size:float, contact_matrix:np.array, weights:list = [] ):

    def Kd_calc(bound, conc):
        return((1-bound)*conc/bound)
    
    Box_L = box_size
    # Convert nM to meters for Box_V in M^3
    Box_V = (Box_L*10**-9)**3
    # Convert Box_V to L
    Box_V_L = Box_V*1000
    #Concentraion in Mols/L
    Concentration = 1/(Box_V_L*(6.023*10**23))
    #print("L:", Box_L, "V:", Box_V, "Conc:", Concentration)

    contact_rows = np.sum(contact_matrix, axis=1)

    if len(weights)>0 :

        bf, bf_be = get_blockerror_pyblock_nanskip_rw_(np.where(contact_rows > 0, 1, 0),weights)
    
    else :

        bf, bf_be = get_blockerror_pyblock_nanskip(np.where(contact_rows > 0, 1, 0))

    upper = bf+bf_be
    KD = Kd_calc(bf, Concentration)
    KD_upper = Kd_calc(upper, Concentration)
    KD_error = KD-KD_upper

    kd=np.round(KD*1000,4)
    kde=np.round(KD_error*1000,4)

    return [bf, bf_be], [kd, kde]


def kd_timecourse_rw_(box_size:float, contact_matrix:np.array, simulation_time:float, weights:list = [], stride:int = 1):
    
    def Kd_calc(bound, conc):
        return((1-bound)*conc/bound)

    Box_L = box_size
    # Convert nM to meters for Box_V in M^3
    Box_V = (Box_L*10**-9)**3
    # Convert Box_V to L
    Box_V_L = Box_V*1000
    #Concentraion in Mols/L
    Concentration = 1/(Box_V_L*(6.023*10**23))
    #print("L:", Box_L, "V:", Box_V, "Conc:", Concentration)
    OUT=[]

    contact_rows = np.sum(contact_matrix, axis=1)
    contact_binary=np.where(contact_rows > 0, 1, 0)
    
    time = np.linspace(0, simulation_time, len(contact_binary))
    boundfrac_by_frame = []
    t2 = []
    err_by_frame = []
    err_upper = []
    err_lower = []
        #stride = 100
        
    for j in range(stride, len(contact_binary), stride):
        #Data = np.asarray(contact_binary[0:j])
        if len(weights)>0 :
            
            bf, be = get_blockerror_pyblock_nanskip_rw_(np.asarray(contact_binary[0:j]),weights[0:j])
        else :
            
            bf, be = get_blockerror_pyblock_nanskip(np.asarray(contact_binary[0:j]))
            
        boundfrac_by_frame.append(bf)
        err_by_frame.append(be)
        err_upper.append(bf-be)
        err_lower.append(bf+be)
        t2.append(time[j])

    Kd = Kd_calc(np.asarray(boundfrac_by_frame), Concentration)*1000
    Kd_upper = Kd_calc(np.asarray(err_upper), Concentration)*1000
    Kd_lower = Kd_calc(np.asarray(err_lower), Concentration)*1000
        
        
    return np.column_stack((t2, Kd, Kd_upper, Kd_lower, boundfrac_by_frame)) 
    

class contact_matrix(initial):

    def __init__(self, rep_trajectory: str, pdb: str, prot_start_res_num: int, prot_end_res_num: int, out_dir: str, analysis_list: list = [], pbc_tpr: str = None, helix_pdb: str = None, pp_pdb: str = None, ligand_rings: list = [], ligand_residue_index: int = None, nreps: int = 1, stride: int = 1, offset: int = 0, ligand_positive_charge_atoms: list = [], ligand_negative_charge_atoms: list = [], lig_hbond_donors: list = [], colvar_file: str = None, pbc: dict = { 'pbc': True,'traj_in_tag': 'prod_','traj_out_tag': 'pbc_','traj_type': 'xtc' }, trj_break: int = 8, rep_demux_switch: str = "off", demux_trajectory: str = None, apo: bool = False, **kwargs):
        super().__init__(rep_trajectory, pdb, prot_start_res_num, prot_end_res_num, out_dir, analysis_list, pbc_tpr, helix_pdb, pp_pdb, ligand_rings, ligand_residue_index, nreps, stride, offset, ligand_positive_charge_atoms, ligand_negative_charge_atoms, lig_hbond_donors, colvar_file, pbc, trj_break, rep_demux_switch, demux_trajectory, apo, **kwargs)

        

    def contact_matrix_rw_(self, trj, cutoff:float = 0.6):
        
        lig_sel=f"resid {self.ligand_residue_index}"
        prot_sel=f"resid {self.prot_start_res_num} to {self.prot_end_res_num}"
        
        ligand=trj.topology.select(lig_sel)
        protein=trj.topology.select(prot_sel)
        
        ligand_atomid = []
        for atom in ligand:
            indices = []
            indices.append(atom)
            indices.append(trj.topology.atom(atom))
            ligand_atomid.append(indices)
            
        protein_atomid = []
        for atom in protein:
            indices = []
            indices.append(atom)
            indices.append(trj.topology.atom(atom))
            protein_atomid.append(indices)
            
        contact_pairs = np.zeros((self.num_residues-1, 2))
    #    ligand_residue_index = 20

        for i in range(0, self.num_residues-1):
            contact_pairs[i] = [i, self.ligand_residue_index]
        contact = md.compute_contacts(trj, contact_pairs, scheme='closest-heavy')
        
        contacts = np.asarray(contact[0]).astype(float)
    #    cutoff = cut_off
        contact_matrix = np.where(contacts < cutoff, 1, 0)
        
    #    contact_rows = contact_rows = np.sum(contact_matrix, in_reps, rest=restxis=1)
        contact_ave, contact_pyb_be = get_blockerrors_pyblock_nanskip(contact_matrix, 1.0)
        
        if self.w :
            
            cre=[]
            for i in range(0,len(contact_matrix[0])):
                cre.append(get_blockerror_pyblock_nanskip_rw_(contact_matrix[:,i],self.weights))
                
            return np.column_stack((self.prot_res_range, cre)), np.column_stack((self.prot_res_range, contact_ave, contact_pyb_be)), convert_to_array(contact_matrix)
        
        else :
        
            return np.column_stack((self.prot_res_range, contact_ave, contact_pyb_be)), convert_to_array(contact_matrix)

        

        

######################################-Class for computing charge contacts between ligand and protein-###############################    


class charge(initial):

    def __init__(self, rep_trajectory: str, pdb: str, prot_start_res_num: int, prot_end_res_num: int, out_dir: str, analysis_list: list = [], pbc_tpr: str = None, helix_pdb: str = None, pp_pdb: str = None, ligand_rings: list = [], ligand_residue_index: int = None, nreps: int = 1, stride: int = 1, offset: int = 0, ligand_positive_charge_atoms: list = [], ligand_negative_charge_atoms: list = [], lig_hbond_donors: list = [], colvar_file: str = None, pbc: dict = { 'pbc': True,'traj_in_tag': 'prod_','traj_out_tag': 'pbc_','traj_type': 'xtc' }, trj_break: int = 8, rep_demux_switch: str = "off", demux_trajectory: str = None, apo: bool = False, **kwargs):
        super().__init__(rep_trajectory, pdb, prot_start_res_num, prot_end_res_num, out_dir, analysis_list, pbc_tpr, helix_pdb, pp_pdb, ligand_rings, ligand_residue_index, nreps, stride, offset, ligand_positive_charge_atoms, ligand_negative_charge_atoms, lig_hbond_donors, colvar_file, pbc, trj_break, rep_demux_switch, demux_trajectory, apo, **kwargs)


        

    def charge_contacts_rw_(self, trj, cutoff=0.5):
        
        def add_charge_pair(pairs,pos,neg,contact_prob):
            if pos not in pairs: 
                pairs[pos] = {} 
            if neg not in pairs[pos]:
                pairs[pos][neg] = {}
            pairs[pos][neg] = contact_prob
            
            
        Protein_Pos_Charges=trj.topology.select("(resname ARG and name CZ) or (resname LYS and name NZ) or (resname HIE and name NE2) or (resname HID and name ND1)")
        Protein_Neg_Charges=trj.topology.select("(resname ASP and name CG) or (resname GLU and name CD) or (name OXT) or (resname NASP and name CG)")
        
        neg_res=[]
        pos_res=[]
        
        for i in Protein_Neg_Charges:
            neg_res.append(trj.topology.atom(i).residue.resSeq)

        for i in Protein_Pos_Charges:
            pos_res.append(trj.topology.atom(i).residue.resSeq)
            
        charge_pairs_ligpos=[]                      
        for i in self.Ligand_Pos_Charges:
            for j in Protein_Neg_Charges:              
                charge_pairs_ligpos.append([i,j])
                pos=trj.topology.atom(i)
                neg=trj.topology.atom(j) 

        charge_pairs_ligneg=[]                      
        for i in self.Ligand_Neg_Charges:
            for j in Protein_Pos_Charges:              
                charge_pairs_ligneg.append([i,j])
                pos=trj.topology.atom(i)
                neg=trj.topology.atom(j)
                
        if len(charge_pairs_ligpos) != 0:
            contact  = md.compute_distances(trj, charge_pairs_ligpos)
            contacts = np.asarray(contact).astype(float)
    #        cutoff=0.5
            neg_res_contact_frames=np.where(contacts < cutoff, 1, 0)
            
            
        if len(charge_pairs_ligneg) != 0:
            contact  = md.compute_distances(trj, charge_pairs_ligneg)
            contacts = np.asarray(contact).astype(float)
    #        cutoff=0.5
            pos_res_contact_frames=np.where(contacts < cutoff, 1, 0)
            
            
        neg_res_index=np.array(neg_res)-self.residue_offset
        
        if self.w :
        
            Charge_Contacts_re=np.zeros((trj.n_frames,trj.topology.n_residues))
            for i in range(0,len(neg_res)):
                Charge_Contacts_re[:,neg_res[i]-self.residue_offset]=neg_res_contact_frames[:,i]
                
            charge_re, charge_be_re = get_blockerrors_pyblock_nanskip_rw_(Charge_Contacts_re,1.0,self.weights)
            
                
            Charge_Contacts=np.zeros((trj.n_frames,trj.topology.n_residues))
            for i in range(0,len(neg_res)):
                Charge_Contacts[:,neg_res[i]-self.residue_offset]=neg_res_contact_frames[:,i]
                
            charge_fraction, charge_fraction_be = get_blockerrors_pyblock_nanskip(Charge_Contacts,1.0)
            
                
            return np.column_stack((self.prot_res_range, charge_re[:self.prot_end_res_num+1], charge_be_re[:self.prot_end_res_num+1])), \
                np.column_stack((self.prot_res_range, charge_fraction[:self.prot_end_res_num+1], charge_fraction_be[:self.prot_end_res_num+1]))

            
        else :

            Charge_Contacts=np.zeros((trj.n_frames,trj.topology.n_residues))
            for i in range(0,len(neg_res)):
                Charge_Contacts[:,neg_res[i]-self.residue_offset]=neg_res_contact_frames[:,i]
                
            charge_fraction, charge_fraction_be = get_blockerrors_pyblock_nanskip(Charge_Contacts,1.0)
            
            return np.column_stack((self.prot_res_range, charge_fraction[:self.prot_end_res_num+1], charge_fraction_be[:self.prot_end_res_num+1]))

        
        

        




######################################-Class for aromatic interactions -###############################  

def find_plane_normal(points):

    N = points.shape[0]
    A = np.concatenate((points[:, 0:2], np.ones((N, 1))), axis=1)
    B = points[:, 2]
    out = lstsq(A, B, rcond=-1)
    na_c, nb_c, d_c = out[0]
    if d_c != 0.0:
        cu = 1./d_c
        bu = -nb_c*cu
        au = -na_c*cu
    else:
        cu = 1.0
        bu = -nb_c
        au = -na_c
    normal = np.asarray([au, bu, cu])
    normal /= math.sqrt(dot(normal, normal))

    return normal

def find_plane_normal2(positions):
    # Alternate approach used to check sign - could the sign check cause descrepency with desres?
    # Use Ligand IDs 312, 308 and 309 to check direction
    # [304 305 306 307 308 309 310 311 312 313]
    v1 = positions[0]-positions[1]
    v1 /= np.sqrt(np.sum(v1**2))
    v2 = positions[2]-positions[1]
    v2 /= np.sqrt(np.sum(v2**2))
    normal = np.cross(v1, v2)

    return normal    

def find_plane_normal2_assign_atomid(positions, id1, id2, id3):
    # Alternate approach used to check sign - could the sign check cause descrepency with desres?
    v1 = positions[id1]-positions[id2]
    v1 /= np.sqrt(np.sum(v1**2))
    v2 = positions[id3]-positions[id1]
    v2 /= np.sqrt(np.sum(v2**2))
    normal = np.cross(v1, v2)

    return normal

def get_ring_center_normal_assign_atomid(positions, id1, id2, id3):
    center = np.mean(positions, axis=0)
    normal = find_plane_normal(positions)
    normal2 = find_plane_normal2_assign_atomid(positions, id1, id2, id3)
    # check direction of normal using dot product convention
    comp = np.dot(normal, normal2)
    if comp < 0:
        normal = -normal

    return center, normal

def get_ring_center_normal_(positions):
    center = np.mean(positions, axis=0)
    normal = find_plane_normal(positions)
    normal2 = find_plane_normal2(positions)
    # check direction of normal using dot product convention
    comp = np.dot(normal, normal2)
    if comp < 0:
        normal = -normal

    return center, normal

def normvector_connect(point1, point2):
    vec = point1-point2
    vec = vec/np.sqrt(np.dot(vec, vec))
    return vec

def angle(v1, v2):
    return np.arccos(np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1))*np.sqrt(np.dot(v2, v2))))

def get_ring_center_normal_trj_assign_atomid(position_array, id1, id2, id3):
    length = len(position_array)
    centers = np.zeros((length, 3))
    normals = np.zeros((length, 3))
    centers_normals = np.zeros((length, 2, 3))
    # print(np.shape(length), np.shape(centers), np.shape(normals))
    for i in range(0, len(position_array)):
        center, normal = get_ring_center_normal_assign_atomid(
            position_array[i], id1, id2, id3)
        centers_normals[i][0] = center
        centers_normals[i][1] = normal

    return centers_normals

class aromatic(initial):

    def __init__(self, rep_trajectory: str, pdb: str, prot_start_res_num: int, prot_end_res_num: int, out_dir: str, analysis_list: list = [], pbc_tpr: str = None, helix_pdb: str = None, pp_pdb: str = None, ligand_rings: list = [], ligand_residue_index: int = None, nreps: int = 1, stride: int = 1, offset: int = 0, ligand_positive_charge_atoms: list = [], ligand_negative_charge_atoms: list = [], lig_hbond_donors: list = [], colvar_file: str = None, pbc: dict = { 'pbc': True,'traj_in_tag': 'prod_','traj_out_tag': 'pbc_','traj_type': 'xtc' }, trj_break: int = 8, rep_demux_switch: str = "off", demux_trajectory: str = None, apo: bool = False, **kwargs):
        super().__init__(rep_trajectory, pdb, prot_start_res_num, prot_end_res_num, out_dir, analysis_list, pbc_tpr, helix_pdb, pp_pdb, ligand_rings, ligand_residue_index, nreps, stride, offset, ligand_positive_charge_atoms, ligand_negative_charge_atoms, lig_hbond_donors, colvar_file, pbc, trj_break, rep_demux_switch, demux_trajectory, apo, **kwargs)

    
    
    def aro_contacts(self, trj):
        
        residues=trj.topology.n_residues
        
        n_rings = len(self.ligand_rings)
        # print("Ligand Aromatics Rings:", n_rings)

        ligand_ring_params = []
        for i in range(0, n_rings):
            ring = np.array(self.ligand_rings[i])

            positions = trj.xyz[:, ring, :]

            ligand_centers_normals = get_ring_center_normal_trj_assign_atomid(positions, 0, 1, 2)
            ligand_ring_params.append(ligand_centers_normals)


        #Find Protein Aromatic Rings
        #Add Apropriate HIS name if there is a charged HIE OR HIP in the structure 
        prot_rings = []
        aro_residues = []
        prot_ring_name = []
        prot_ring_index = []

        aro_select = trj.topology.select("resname TYR PHE HIS TRP and name CA")
        for i in aro_select:
            atom = trj.topology.atom(i)
            resname = atom.residue.name

            if resname == "TYR":
                ring = trj.topology.select(
                    "resid %s and name CG CD1 CD2 CE1 CE2 CZ" % atom.residue.index)

            if resname == "TRP":
                ring = trj.topology.select(
                    "resid %s and name CG CD1 NE1 CE2 CD2 CZ2 CE3 CZ3 CH2" % atom.residue.index)

            if resname == "HIS":
                ring = trj.topology.select("resid %s and name CG ND1 CE1 NE2 CD2" %
                                atom.residue.index)

            if resname == "PHE":
                ring = trj.topology.select(
                    "resid %s and name CG CD1 CD2 CE1 CE2 CZ" % atom.residue.index)

            prot_rings.append(ring)
            prot_ring_name.append(atom.residue)
            prot_ring_index.append(atom.residue.index+self.residue_offset)



        prot_ring_params = []
        for i in range(0, len(prot_rings)):
            ring = np.array(prot_rings[i])
            positions = trj.xyz[:, ring, :]
            ring_centers_normals = get_ring_center_normal_trj_assign_atomid(positions, 0, 1, 2)
            prot_ring_params.append(ring_centers_normals)

    #    frames = trj.n_frames
        sidechains = len(prot_rings)
        ligrings = len(self.ligand_rings)

        # print(trj.n_frames, sidechains)

        Ringstacked_old = {}
        Ringstacked = {}
        Quadrants = {}
        Stackparams = {}
        Aro_Contacts = {}
        Pstack = {}
        Tstack = {}


        # def normvector_connect(point1, point2):
        #     vec = point1-point2
        #     vec = vec/np.sqrt(np.dot(vec, vec))
        #     return vec


        # def angle(v1, v2):
        #     return np.arccos(np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1))*np.sqrt(np.dot(v2, v2))))

        '''
        print("q1: alpha<=45 and beta>=135")
        print("q2: alpha>=135 and beta>=135")
        print("q3: alpha<=45 and beta<=45")
        print("q4: alpha>=135 and beta<=135")
        '''

        for l in range(0, ligrings):
            name = "Lig_ring.%s" % l

            Stackparams[name] = {}
            Pstack[name] = {}
            Tstack[name] = {}
            Aro_Contacts[name] = {}
            alphas = np.zeros(shape=(trj.n_frames, sidechains))
            betas = np.zeros(shape=(trj.n_frames, sidechains))
            dists = np.zeros(shape=(trj.n_frames, sidechains))
            thetas = np.zeros(shape=(trj.n_frames, sidechains))
            phis = np.zeros(shape=(trj.n_frames, sidechains))
            pstacked_old = np.zeros(shape=(trj.n_frames, sidechains))
            pstacked = np.zeros(shape=(trj.n_frames, sidechains))
            tstacked = np.zeros(shape=(trj.n_frames, sidechains))
            stacked = np.zeros(shape=(trj.n_frames, sidechains))
            aro_contacts = np.zeros(shape=(trj.n_frames, sidechains))
            quadrant=np.zeros(shape=(trj.n_frames,sidechains))

            for i in range(0, trj.n_frames):
                ligcenter = ligand_ring_params[l][i][0]
                lignormal = ligand_ring_params[l][i][1]
                for j in range(0, sidechains):
                    protcenter = prot_ring_params[j][i][0]
                    protnormal = prot_ring_params[j][i][1]
                    dists[i, j] = np.linalg.norm(ligcenter-protcenter)
                    connect = normvector_connect(protcenter, ligcenter)
                    # alpha is the same as phi in gervasio/Procacci definition
                    alphas[i, j] = np.rad2deg(angle(connect, protnormal))
                    betas[i, j] = np.rad2deg(angle(connect, lignormal))
                    theta = np.rad2deg(angle(protnormal, lignormal))
                    thetas[i, j] = np.abs(theta)-2*(np.abs(theta)
                                                    > 90.0)*(np.abs(theta)-90.0)
                    phi = np.rad2deg(angle(protnormal, connect))
                    phis[i, j] = np.abs(phi)-2*(np.abs(phi) > 90.0)*(np.abs(phi)-90.0)

            for j in range(0, sidechains):
                name2 = prot_ring_index[j]
                res2 = prot_ring_name[j]
                # print('====>',name2, res2)
                
                Ringstack = np.column_stack((dists[:, j], alphas[:, j], betas[:, j], thetas[:, j], phis[:, j]))
                stack_distance_cutoff = 0.65
                r = np.where(dists[:, j] <= stack_distance_cutoff)[0]
                aro_contacts[:, j][r] = 1

                # New Definitions
                # p-stack: r < 6.5 Å, θ < 60° and ϕ < 60°.
                # t-stack: r < 7.5 Å, 75° < θ < 90° and ϕ < 60°.
                p_stack_distance_cutoff = 0.65
                t_stack_distance_cutoff = 0.75
                r_pstrict = np.where(dists[:, j] <= p_stack_distance_cutoff)[0]
                r_tstrict = np.where(dists[:, j] <= t_stack_distance_cutoff)[0]

                a=np.where(alphas[:,j] >= 135)
                b=np.where(alphas[:,j] <= 45)
                c=np.where(betas[:,j] >= 135)
                d=np.where(betas[:,j] <= 45)
                e=np.where(dists[:,j] <= 0.5)
                q1=np.intersect1d(np.intersect1d(b,c),e)
                q2=np.intersect1d(np.intersect1d(a,c),e)
                q3=np.intersect1d(np.intersect1d(b,d),e)
                q4=np.intersect1d(np.intersect1d(a,d),e)
                stacked[:,j][q1]=1
                stacked[:,j][q2]=1
                stacked[:,j][q3]=1
                stacked[:,j][q4]=1
                quadrant[:,j][q1]=1
                quadrant[:,j][q2]=2
                quadrant[:,j][q3]=3
                quadrant[:,j][q4]=4
                total_stacked=len(q1)+len(q2)+len(q3)+len(q4)
                
                # print("q1:",len(q1),"q2:",len(q2),"q3:",len(q3),"q4:",len(q4))
                # print("q1:",len(q1)/total_stacked,"q2:",len(q2)/total_stacked,"q3:",len(q3)/total_stacked,"q4:",len(q4)/total_stacked)
                # print(max(len(q1),len(q2),len(q3),len(q4))/min(len(q1),len(q2),len(q3),len(q4)))
                
                Stackparams[name][name2]=Ringstack

                # print(np.average(Ringstack, in_reps, rest=restxis=0))
                
                f = np.where(thetas[:, j] <= 45)
                g = np.where(phis[:, j] <= 60)
                h = np.where(thetas[:, j] >= 75)

                pnew = np.intersect1d(np.intersect1d(f, g), r_pstrict)
                tnew = np.intersect1d(np.intersect1d(h, g), r_tstrict)
                pstacked[:, j][pnew] = 1
                tstacked[:, j][tnew] = 1
                stacked[:, j][pnew] = 1
                stacked[:, j][tnew] = 1
                total_stacked = len(pnew)+len(tnew)
                
                # print("===>Contacts:", len(r), "Total:", total_stacked,"P-stack:", len(pnew), "T-stack:", len(tnew))
                
                Stackparams[name][name2] = Ringstack
            Pstack[name] = pstacked
            Tstack[name] = tstacked
            Aro_Contacts[name] = aro_contacts
            Ringstacked[name] = stacked
            Quadrants[name]=quadrant
        
        
        aro_res_index = np.array(prot_ring_index)-self.residue_offset

        aromatic_stacking_contacts_r0 = np.zeros((trj.n_frames, residues))

        for i in range(0, len(aro_res_index)):
            aromatic_stacking_contacts_r0[:, aro_res_index[i]] = Ringstacked['Lig_ring.0'][:, i]
            
 
        # self.aro_binary_contacts = aromatic_stacking_contacts_r0
        # self.stackparams = Stackparams
        

        aro_r0_ave, aro_r0_pyb_be = get_blockerrors_pyblock(np.array(aromatic_stacking_contacts_r0), 1.0)
        
        if self.w : 
            
            aro_r0_ave_re, aro_r0_pyb_be_re = get_blockerrors_pyblock_nanskip_rw_(np.array(aromatic_stacking_contacts_r0), 1.0, self.weights)
                
            return np.column_stack((self.prot_res_range, aro_r0_ave_re[:self.prot_end_res_num+1], aro_r0_pyb_be_re[:self.prot_end_res_num+1])),\
                  np.column_stack((self.prot_res_range, aro_r0_ave[:self.prot_end_res_num+1], aro_r0_pyb_be[:self.prot_end_res_num+1])), aromatic_stacking_contacts_r0, Stackparams
        
        else:
            
            return np.column_stack((self.prot_res_range, aro_r0_ave[:self.prot_end_res_num+1], aro_r0_pyb_be[:self.prot_end_res_num+1])), aromatic_stacking_contacts_r0, Stackparams

        


    
######################################- Class for hydrophobic interactions -###############################    

class hydrophobic(initial):

    def __init__(self, rep_trajectory: str, pdb: str, prot_start_res_num: int, prot_end_res_num: int, out_dir: str, analysis_list: list = [], pbc_tpr: str = None, helix_pdb: str = None, pp_pdb: str = None, ligand_rings: list = [], ligand_residue_index: int = None, nreps: int = 1, stride: int = 1, offset: int = 0, ligand_positive_charge_atoms: list = [], ligand_negative_charge_atoms: list = [], lig_hbond_donors: list = [], colvar_file: str = None, pbc: dict = { 'pbc': True,'traj_in_tag': 'prod_','traj_out_tag': 'pbc_','traj_type': 'xtc' }, trj_break: int = 8, rep_demux_switch: str = "off", demux_trajectory: str = None, apo: bool = False, **kwargs):
        super().__init__(rep_trajectory, pdb, prot_start_res_num, prot_end_res_num, out_dir, analysis_list, pbc_tpr, helix_pdb, pp_pdb, ligand_rings, ligand_residue_index, nreps, stride, offset, ligand_positive_charge_atoms, ligand_negative_charge_atoms, lig_hbond_donors, colvar_file, pbc, trj_break, rep_demux_switch, demux_trajectory, apo, **kwargs)




    def hphob_contacts_rw_(self ,trj, cutoff=0.4):

        residues=trj.topology.n_residues
        
        def add_contact_pair(pairs, a1, a2, a1_id, a2_id, prot_res, contact_prob):
            if prot_res not in pairs:
                pairs[prot_res] = {}
            if a2 not in pairs[prot_res]:
                pairs[prot_res][a2] = {}
            if a1_id not in pairs[prot_res][a2]:
                pairs[prot_res][a2][a1_id] = contact_prob
        
        s1=f"resid {self.ligand_residue_index} and element C"
        s2=f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and element C and not name CA"

        ligand_hphob = trj.topology.select(s1)
        protein_hphob = trj.topology.select(s2)
        
        ligand_hphob_atoms = []
        for atom in ligand_hphob:
            ligand_hphob_atoms.append(trj.topology.atom(atom))

        protein_hphob_atoms = []
        for atom in protein_hphob:
            protein_hphob_atoms.append(trj.topology.atom(atom))
            
        hphob_pairs = []
        for i in ligand_hphob:
            for j in protein_hphob:
                hphob_pairs.append([i, j])
                
        contact = md.compute_distances(trj, hphob_pairs)
        contacts = np.asarray(contact).astype(float)

        contact_frames = np.where(contacts < cutoff, 1, 0)
        
        # Cast hydrophobic contacts as per residue in each frame
        
        Hphob_res_contacts = np.zeros((trj.n_frames, residues))
        for frame in range(trj.n_frames):
            if np.sum(contact_frames[frame]) > 0:
                contact_pairs = np.where(contact_frames[frame] == 1)
                for j in contact_pairs[0]:
                    residue = trj.topology.atom(hphob_pairs[j][1]).residue.resSeq
                    Hphob_res_contacts[frame][residue-self.residue_offset] = 1
                    
        hphob_ave, hphob_pyb_be = get_blockerrors_pyblock_nanskip(Hphob_res_contacts, 1.0)
        
        # self.hydro_binary = Hphob_res_contacts
        
        if self.w :
            
            hphob_ave_re, hphob_pyb_be_re = get_blockerrors_pyblock_nanskip_rw_(Hphob_res_contacts, 1.0, self.weights)
            
                    
            return np.column_stack((self.prot_res_range, hphob_ave_re[:self.prot_end_res_num+1], hphob_pyb_be_re[:self.prot_end_res_num+1])), \
                np.column_stack((self.prot_res_range, hphob_ave[:self.prot_end_res_num+1], hphob_pyb_be[:self.prot_end_res_num+1])), Hphob_res_contacts
        
        else :
                    
            return np.column_stack((self.prot_res_range, hphob_ave[:self.prot_end_res_num+1], hphob_pyb_be[:self.prot_end_res_num+1])), Hphob_res_contacts


######################################- Class for hydrogen bond interactions -###############################    

def baker_hubbard2(traj, freq=0.1, exclude_water=True, periodic=True, sidechain_only=False,
                   distance_cutoff=0.35, angle_cutoff=150, lig_donor_index=[]):

    angle_cutoff = np.radians(angle_cutoff)

    if traj.topology is None:
        raise ValueError('baker_hubbard requires that traj contain topology '
                         'information')

    # Get the possible donor-hydrogen...acceptor triplets

    # ADD IN LIGAND HBOND DONORS
    add_donors = lig_donor_index

    bond_triplets = _get_bond_triplets(traj.topology,
                                       exclude_water=exclude_water, lig_donors=add_donors, sidechain_only=sidechain_only)

    mask, distances, angles = _compute_bounded_geometry(traj, bond_triplets,
                                                        distance_cutoff, [1, 2], [0, 1, 2], freq=freq, periodic=periodic)

    # Find triplets that meet the criteria
    presence = np.logical_and(
        distances < distance_cutoff, angles > angle_cutoff)
    mask[mask] = np.mean(presence, axis=0) > freq
    return bond_triplets.compress(mask, axis=0)

def _compute_bounded_geometry(traj, triplets, distance_cutoff, distance_indices,
                              angle_indices, freq=0.0, periodic=True):
    """
    Returns a tuple include (1) the mask for triplets that fulfill the distance
    criteria frequently enough, (2) the actual distances calculated, and (3) the
    angles between the triplets specified by angle_indices.
    """
    # First we calculate the requested distances
    distances = md.compute_distances(
        traj, triplets[:, distance_indices], periodic=periodic)

    # Now we discover which triplets meet the distance cutoff often enough
    prevalence = np.mean(distances < distance_cutoff, axis=0)
    mask = prevalence > freq

    # Update data structures to ignore anything that isn't possible anymore
    triplets = triplets.compress(mask, axis=0)
    distances = distances.compress(mask, axis=1)

    # Calculate angles using the law of cosines
    abc_pairs = zip(angle_indices, angle_indices[1:] + angle_indices[:1])
    abc_distances = []

    # Calculate distances (if necessary)
    for abc_pair in abc_pairs:
        if set(abc_pair) == set(distance_indices):
            abc_distances.append(distances)
        else:
            abc_distances.append(md.compute_distances(traj, triplets[:, abc_pair],
                                                      periodic=periodic))

    # Law of cosines calculation
    a, b, c = abc_distances
    cosines = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    np.clip(cosines, -1, 1, out=cosines)  # avoid NaN error
    angles = np.arccos(cosines)
    return mask, distances, angles

def _get_bond_triplets(topology, lig_donors, exclude_water=True, sidechain_only=False):
    def can_participate(atom):
        # Filter waters
        if exclude_water and atom.residue.is_water:
            return False
        # Filter non-sidechain atoms
        if sidechain_only and not atom.is_sidechain:
            return False
        # Otherwise, accept it
        return True

    def get_donors(e0, e1):
        # Find all matching bonds
        elems = set((e0, e1))
        atoms = [(one, two) for one, two in topology.bonds
                 if set((one.element.symbol, two.element.symbol)) == elems]
        # Filter non-participating atoms
        atoms = [atom for atom in atoms
                 if can_participate(atom[0]) and can_participate(atom[1])]
        # Get indices for the remaining atoms
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            # make sure to get the pair in the right order, so that the index
            # for e0 comes before e1
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)

        return indices

    # Check that there are bonds in topology
    nbonds = 0
    for _bond in topology.bonds:
        nbonds += 1
        break  # Only need to find one hit for this check (not robust)
    if nbonds == 0:
        raise ValueError('No bonds found in topology. Try using '
                         'traj._topology.create_standard_bonds() to create bonds '
                         'using our PDB standard bond definitions.')

    nh_donors = get_donors('N', 'H')
    oh_donors = get_donors('O', 'H')
    sh_donors = get_donors('S', 'H')
    xh_donors = np.array(nh_donors + oh_donors + sh_donors+lig_donors)

    if len(xh_donors) == 0:
        # if there are no hydrogens or protein in the trajectory, we get
        # no possible pairs and return nothing
        return np.zeros((0, 3), dtype=int)

    acceptor_elements = frozenset(('O', 'N', 'S'))
    acceptors = [a.index for a in topology.atoms
                 if a.element.symbol in acceptor_elements and can_participate(a)]
    # Make acceptors a 2-D numpy array
    acceptors = np.array(acceptors)[:, np.newaxis]

    # Generate the cartesian product of the donors and acceptors
    xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
    acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
    bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))

    # Filter out self-bonds
    self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
    return bond_triplets[np.logical_not(self_bond_mask), :]



class hbond(initial):

    def __init__(self, rep_trajectory: str, pdb: str, prot_start_res_num: int, prot_end_res_num: int, out_dir: str, analysis_list: list = [], pbc_tpr: str = None, helix_pdb: str = None, pp_pdb: str = None, ligand_rings: list = [], ligand_residue_index: int = None, nreps: int = 1, stride: int = 1, offset: int = 0, ligand_positive_charge_atoms: list = [], ligand_negative_charge_atoms: list = [], lig_hbond_donors: list = [], colvar_file: str = None, pbc: dict = { 'pbc': True,'traj_in_tag': 'prod_','traj_out_tag': 'pbc_','traj_type': 'xtc' }, trj_break: int = 8, rep_demux_switch: str = "off", demux_trajectory: str = None, apo: bool = False, **kwargs):
        super().__init__(rep_trajectory, pdb, prot_start_res_num, prot_end_res_num, out_dir, analysis_list, pbc_tpr, helix_pdb, pp_pdb, ligand_rings, ligand_residue_index, nreps, stride, offset, ligand_positive_charge_atoms, ligand_negative_charge_atoms, lig_hbond_donors, colvar_file, pbc, trj_break, rep_demux_switch, demux_trajectory, apo, **kwargs)



    def hbond_rw_(self, trj):


        lig_sel=f"resid {self.ligand_residue_index}"
        prot_sel=f"resid {self.prot_start_res_num} to {self.prot_end_res_num}"
        # Select Ligand Residues
        ligand = trj.topology.select(lig_sel)
        # Select Protein Residues
        protein = trj.topology.select(prot_sel)


        HBond_PD = np.zeros((trj.n_frames, trj.topology.n_residues))
        HBond_LD = np.zeros((trj.n_frames, trj.topology.n_residues))
        Hbond_pairs_PD = {}
        Hbond_pairs_LD = {}


        def add_hbond_pair(donor, acceptor, hbond_pairs, donor_res):
            if donor_res not in hbond_pairs:
                hbond_pairs[donor_res] = {}
            if donor not in hbond_pairs[donor_res]:
                hbond_pairs[donor_res][donor] = {}
            if acceptor not in hbond_pairs[donor_res][donor]:
                hbond_pairs[donor_res][donor][acceptor] = 0
            hbond_pairs[donor_res][donor][acceptor] += 1

        # Donor & Acceptors Definitions from DESRES paper:
        # ligdon = mol.select('chain B and (nitrogen or oxygen or sulfur) and (withinbonds 1 of hydrogen)')
        # ligacc = mol.select('chain B and (nitrogen or oxygen or sulfur)')
        # protdon = mol.select('chain A and (nitrogen or oxygen or sulfur) and (withinbonds 1 of hydrogen)')
        # protacc = mol.select('chain A and (nitrogen or oxygen or sulfur)')


        for frame in range(trj.n_frames):
            hbonds = baker_hubbard2(trj[frame], angle_cutoff=150, distance_cutoff=0.35, 
                                    lig_donor_index=self.lig_hbond_donors)

            for hbond in hbonds:
                if ((hbond[0] in protein) and (hbond[2] in ligand)):
                    donor = trj.topology.atom(hbond[0])
                    donor_id = hbond[0]
                    donor_res = trj.topology.atom(hbond[0]).residue.resSeq
                    acc = trj.topology.atom(hbond[2])
                    acc = trj.topology.atom(hbond[2])
                    acc_res = trj.topology.atom(hbond[2]).residue.resSeq
                    HBond_PD[frame][donor_res-self.residue_offset] = 1
                    add_hbond_pair(donor, acc, Hbond_pairs_PD, donor_res)
                if ((hbond[0] in ligand) and (hbond[2] in protein)):
                    donor = trj.topology.atom(hbond[0])
                    donor_id = hbond[0]
                    donor_res = trj.topology.atom(hbond[0]).residue.resSeq
                    acc = trj.topology.atom(hbond[2])
                    acc_id = hbond[2]
                    acc_res = trj.topology.atom(hbond[2]).residue.resSeq
                    HBond_LD[frame][acc_res-self.residue_offset] = 1
                    add_hbond_pair(donor, acc, Hbond_pairs_LD, acc_res)


        HB_Total = HBond_PD+HBond_LD
        
        HBond_ave, HBond_pyb_be = get_blockerrors_pyblock(HB_Total, 1.0)

        # self.hbond_binary = HB_Total
        
        if self.w :
            
            HBond_ave_re, HBond_pyb_be_re = get_blockerrors_pyblock_nanskip_rw_(HB_Total, 1.0, self.weights)
            
            return np.column_stack((self.prot_res_range, HBond_ave_re[:self.prot_end_res_num+1], HBond_pyb_be_re[:self.prot_end_res_num+1])), \
                np.column_stack((self.prot_res_range, HBond_ave[:self.prot_end_res_num+1], HBond_pyb_be[:self.prot_end_res_num+1])), HB_Total
        
        else :
            
            return np.column_stack((self.prot_res_range, HBond_ave[:self.prot_end_res_num+1], HBond_pyb_be[:self.prot_end_res_num+1])), HB_Total

######################################- Functions for protein:protein and Protein:Ligand:Protein contact matrix-###############################  

## TOMMY rocks ! 


def contact_map_protein_rw(trj, weights:list=[], cutoff:float=1.2, apo:bool=False):

    import numpy as np
    import mdtraj as md
    
    """
    Compute a reweighted contact map and distance matrix for protein-protein interactions.
    
    Parameters:
    trj : mdtraj.Trajectory
        The trajectory containing the protein atoms.
    weights : list or np.ndarray
        Normalized weights for reweighting the contacts (if provided).
    cutoff : float
        Distance cutoff for contact definition in nm.
    apo : bool
        If True, uses all residues; if False, excludes the last residue.
        
    Returns:
    np.ndarray
        Reweighted contact map.
    """
    # Determine the number of residues
    p_residues = trj.topology.n_residues - 1 if not apo else trj.topology.n_residues

    # Generate upper triangle indices for residue pairs
    indices = np.stack(np.triu_indices(p_residues, 1), axis=1)

    # Compute distances between residue pairs across all frames
    dist_array = np.array(md.compute_contacts(trj, indices)[0]).astype(float)

    # Identify contacts based on cutoff
    contact_array = np.where(dist_array < cutoff, 1, 0)

    # Initialize contact and distance matrices
    distance_matrix = np.zeros((p_residues, p_residues))
    contact_matrix = np.zeros((p_residues, p_residues))

    if len(weights) > 0:
        # Ensure weights are normalized
        weights = np.array(weights) / np.sum(weights)

        # Reweighting contacts and distances
        reweighted_distances = np.dot(weights, dist_array)  # Reweight distances across frames
        reweighted_contacts = np.dot(weights, contact_array)  # Reweight contacts across frames

        # Fill the upper triangle of the matrices with the reweighted values
        distance_matrix[indices[:, 0], indices[:, 1]] = reweighted_distances
        contact_matrix[indices[:, 0], indices[:, 1]] = reweighted_contacts
    else:
        # Compute mean values without reweighting
        distance_matrix[indices[:, 0], indices[:, 1]] = dist_array.mean(axis=0)
        contact_matrix[indices[:, 0], indices[:, 1]] = contact_array.mean(axis=0)

    # Make matrices symmetric
    distance_matrix += distance_matrix.T
    contact_matrix += contact_matrix.T

    return np.array(contact_matrix).astype(float)  # , np.array(distance_matrix).astype(float)



def contact_map_ligand_rw(trj, ps:int, pe:int, ligand_res_index:int, weights:list=[], cutoff=1.2):

    contact_maps = []
    for i in range(ps,pe):
        # print(i)
        contact_map = []
        for j in range(ps,pe):
            dist1 = md.compute_contacts(trj, [[i, ligand_res_index]], scheme='closest-heavy')
            dist2 = md.compute_contacts(trj, [[j, ligand_res_index]], scheme='closest-heavy')
            array1 = np.asarray(dist1[0]).astype(float)
            array2 = np.asarray(dist2[0]).astype(float)
            contact1 = np.where(array1 < cutoff, 1, 0)
            contact2 = np.where(array2 < cutoff, 1, 0)
            sum = contact1 + contact2
            contact = np.where(sum == 2, 1, 0)

            if len(weights)>0:
                contacts = np.dot(contact[:,0],weights)
            else:
                contacts = np.average(contact)

            contact_map.append(contacts)
        contact_maps.append(contact_map)
        
    return np.asarray(contact_maps).astype(float)

# def contact_map_ligand_rw_1(trj, ps:int, pe:int, ligand_res_index:int, weights:list=[], cutoff=1.2):
#     # Create pairs for compute_contacts
#     pairs = [[i, ligand_res_index] for i in range(ps, pe+1)]
#     pairs = np.array(pairs)

#     # Compute distances
#     dists = md.compute_contacts(trj, pairs, scheme='closest-heavy')[0]

#     # Convert to float
#     dists = dists.astype(float)

#     # Compute contact maps
#     contacts = np.where(dists < cutoff, 1, 0)

#     # Compute pairwise contacts
#     pairwise_contacts = np.dot(contacts, contacts.T)

#     # Compute final contact map
#     final_contacts = np.where(pairwise_contacts == 2, 1, 0)

#     # Apply weights if provided
#     if len(weights) > 0:
#         final_contacts = np.dot(final_contacts, weights)
#     else:
#         final_contacts = np.average(final_contacts, axis=1)

#     return final_contacts.astype(float)

def contact_map_ligand_2(trj, ps:int, pe:int, ligand_res_index:int, weights:list=[], cutoff=1.2):
    # Create pairs for compute_contacts
    pairs = [[i, ligand_res_index] for i in range(ps, pe+1)]
    pairs = np.array(pairs)

    # Compute distances
    dists = np.asarray(md.compute_contacts(trj, pairs, scheme='closest-heavy')[0]).astype(float)

    contacts = np.where(dists < cutoff, 1, 0)

    dual = (contacts.T @ contacts) / len(contacts)

    return dual

## TOMMY rocks !

def contact_map_ligand_rw_2(trj, ps:int, pe:int, ligand_res_index:int, weights:list=[], cutoff=0.6):
    import numpy as np
    import mdtraj as md 
    
    """
    Compute a reweighted dual contact map for ligand-protein interactions.
    
    Parameters:
    trj : mdtraj.Trajectory
        The trajectory containing protein and ligand atoms.
    ps : int
        Starting residue index for the protein.
    pe : int
        Ending residue index for the protein.
    ligand_res_index : int
        Residue index of the ligand.
    weights : list or np.ndarray
        Normalized weights for reweighting the contacts (if provided).
    cutoff : float
        Distance cutoff for contact definition in nm.
        
    Returns:
    np.ndarray
        Reweighted dual contact map.
    """
    
    # Create pairs for computing contacts (protein residues with ligand)
    pairs = np.array([[i, ligand_res_index] for i in range(ps, pe+1)])

    # Compute distances between protein-ligand pairs for all frames
    dists = np.asarray(md.compute_contacts(trj, pairs, scheme='closest-heavy')[0]).astype(float)

    # Identify contacts (1 if within cutoff distance, otherwise 0)
    contacts = np.where(dists < cutoff, 1, 0)

    if len(weights) > 0:
        # Normalize weights if not already normalized
        weights = np.array(weights) / np.sum(weights)
        
        # Reweight contacts by applying weights to each frame
        reweighted_contacts = contacts * weights[:, np.newaxis]
        
        # Compute reweighted dual contact map
        dual = (reweighted_contacts.T @ contacts) / np.sum(weights)
    else:
        # If no weights provided, compute dual contact map without reweighting
        dual = (contacts.T @ contacts) / len(contacts)

    return dual




######################################- Class for user initiation -###############################    

def dict_conv(input, keys_list:list=[], start:int=0, rest:bool=False):

    if not rest : return dict(enumerate(input, start))
    else : 
        out_put={}
        a=0
        for k in keys_list:
            out_put[k]=input[a]
            a+=1
        
        return out_put
    
def reset_dict(to_reset:dict) :
    
    if not len(to_reset) : 
        to_reset={}
    elif len(to_reset) : 

        for i in to_reset : i={}
    


# def dict_conv1(input, keys_list:list):
#     out_put={}
#     a=0
#     for k in keys_list:
#         out_put[k]=input[a]
#         a+=1
    
#     return out_put

class ligand_interactions(contact_matrix, charge, aromatic, hydrophobic, hbond):

    def __init__(self, rep_trajectory: str, pdb: str, prot_start_res_num: int, prot_end_res_num: int, out_dir: str, analysis_list: list = [], pbc_tpr: str = None, helix_pdb: str = None, pp_pdb: str = None, ligand_rings: list = [], ligand_residue_index: int = None, nreps: int = 1, stride: int = 1, offset: int = 0, ligand_positive_charge_atoms: list = [], ligand_negative_charge_atoms: list = [], lig_hbond_donors: list = [], colvar_file: str = None, pbc: dict = { 'pbc': True,'traj_in_tag': 'prod_','traj_out_tag': 'pbc_','traj_type': 'xtc' }, trj_break: int = 8, rep_demux_switch: str = "off", demux_trajectory: str = None, apo: bool = False, **kwargs):
        super().__init__(rep_trajectory, pdb, prot_start_res_num, prot_end_res_num, out_dir, analysis_list, pbc_tpr, helix_pdb, pp_pdb, ligand_rings, ligand_residue_index, nreps, stride, offset, ligand_positive_charge_atoms, ligand_negative_charge_atoms, lig_hbond_donors, colvar_file, pbc, trj_break, rep_demux_switch, demux_trajectory, apo, **kwargs)
        
        s_time = time.perf_counter()
        # if not self.rest : self.sequence = np.array([f'{residue}' for residue in self.trj[0].topology.residues])
        # elif self.rest : self.sequence = np.array([f'{residue}' for residue in self.trj["rep:0"].topology.residues])
        self.sequence = np.array([f'{residue}' for residue in self.temp.topology.residues])

        print(f"Residues in the trjectory : \n{self.sequence}\n")

        print(f"Total number of residues excluding ions: {self.num_residues}\n")
        if not self.apo : print(f"Number of protein residues: {self.num_residues-1}\n")
        elif self.apo : print(f"Number of protein residues: {self.num_residues}\n")

        if not self.apo : print(f"Protien residues : \n{self.sequence[:-1 or None]}\n")
        elif self.apo : print(f"Protien residues : \n{self.sequence}\n")

        self.temp=None

        write=True

        if not self.rest :

            if self.nreps < self.trj_brk : 

                s_time = time.perf_counter()

                in_reps = [i for i in range(self.nreps)]
                self.trj={}


                for i in in_reps :

                    print(f"Loading trajectory {self.traj_tag_in}...\n")


                    self.trj[i] = load_traj(args_dict={'trajectory':self.trj_dict[i],
                                            'pdb':self.pdb_p,
                                            'stride':self.stride, 
                                            'p_sel':self.p_sel})                    
                    

                    self.n_frames[i] = self.trj[i].n_frames
                    self.simulation_time[i] = self.trj[i].timestep * self.n_frames[i]/(10**6)
                    self.box_size[i] = self.trj[i].unitcell_lengths[0][0]

                e_time = time.perf_counter()

                print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                print (f"Analysis for {in_reps}\n")

                self.analysis(in_reps,write=write, rest=self.rest)

                # self.write_json(write=write)

                write=False

            else :


                for i in range(0,self.nreps,self.trj_brk):
                    s_time = time.perf_counter()
                    in_reps=[]
                    self.trj={}

                    for j in range(i,i+self.trj_brk):

                        if j<self.nreps:
                            
                            in_reps.append(j)

                            print(f"Loading trajectory {j}...\n")

                            self.trj[j] = load_traj(args_dict={'trajectory':self.trj_dict[j],
                                                    'pdb':self.pdb_p,
                                                    'stride':self.stride, 
                                                    'p_sel':self.p_sel})                    
                            
                            self.n_frames[j] = self.trj[j].n_frames
                            self.simulation_time[j] = self.trj[j].timestep * self.n_frames[j]/(10**6)
                            self.box_size[j] = self.trj[j].unitcell_lengths[0][0]

                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    print (f"Analysis for {in_reps}\n")

                    self.analysis(in_reps,write=write, rest=self.rest)

                    # self.write_json(write=write)

                    write=False

        elif self.rest:
            

            keys = list(self.trj_dict.keys())

            for i in range(0,len(keys),self.trj_brk):
                s_time = time.perf_counter()

                batch_keys = keys[i:i + self.trj_brk]
                in_reps=[]
                self.trj={}

                for key in batch_keys:

                    in_reps.append(key[1])

                    print(f"Loading trajectory {key}...\n")

                    self.trj[key] = load_traj(args_dict={'trajectory':self.trj_dict[key],
                                                         'pdb':self.pdb_p,
                                                         'stride':self.stride,
                                                         'p_sel':self.p_sel})                    
                    
                    self.n_frames[key] = self.trj[key].n_frames
                    self.simulation_time[key] = self.trj[key].timestep * self.n_frames[key]/(10**6)
                    self.box_size[key] = self.trj[key].unitcell_lengths[0][0]

                e_time = time.perf_counter()
                print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                print (f"Analysis for {batch_keys}\n")

                self.analysis(batch_keys,write=write, rest=self.rest)

                # self.write_json(write=write)

                write=False

        e_time = time.perf_counter()
        print(f'All Done in {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

        # random_quote()

        del self.trj

        pass

    def contact_matrix_out(self, j):
        
        return self.contact_matrix_rw_(self.trj[j])
    
    def kd_out(self, j):

        return bound_frac_kd_rw_(self.box_size[j],self.contact_matrix[j],self.weights)

    def kd_out_raw(self, j):

        return bound_frac_kd_rw_(self.box_size[j],self.contact_matrix[j])
    
    def kd_time_out(self, j):

        return kd_timecourse_rw_(self.box_size[j],self.contact_matrix[j],self.simulation_time[j],weights=self.weights)

    def charge_out(self, j):

        return self.charge_contacts_rw_(self.trj[j])
    
    def aro_out(self, j):

        return self.aro_contacts(self.trj[j])
    
    def hydro_out(self, j):

        return self.hphob_contacts_rw_(self.trj[j])

    def hbond_out(self, j):

        return self.hbond_rw_(self.trj[j])

    def p_cm(self, j):

        if self.w : return contact_map_protein_rw(self.trj[j],self.weights, apo=self.apo, cutoff=1.2)
        else : return contact_map_protein_rw(self.trj[j], apo=self.apo, cutoff=1.2)
    
    # def l_cm(self, j):

    #     if self.w : return contact_map_ligand_rw(self.trj[j],self.prot_start_res_num,self.prot_end_res_num,self.ligand_residue_index,self.weights)
    #     else : return contact_map_ligand_rw(self.trj[j],self.prot_start_res_num,self.prot_end_res_num,self.ligand_residue_index)

    def l_cm(self, j):

        if self.w : return contact_map_ligand_rw_2(self.trj[j],self.prot_start_res_num,self.prot_end_res_num,self.ligand_residue_index,self.weights, cutoff=0.6)
        else : return contact_map_ligand_rw_2(self.trj[j],self.prot_start_res_num,self.prot_end_res_num,self.ligand_residue_index, cutoff=0.6)

    def write_json_file(self, out_para, out_file:str, write:bool=True ):

        if write : json_dump(self.out_dir, out_para, out_file, 'w') 
        else : json_dump(self.out_dir, out_para, out_file, 'a')

        pass
    

    

    def analysis(self, in_reps, write:bool, rest:bool=False):

        def bf(input, bf):

            assert input.shape[1]==3 , "The input should be 3 column array! "

            a=input.T[1:]/bf
            out=np.column_stack((input.T[0],a.T))

            return out
        
        if self.apo is True : 

            for i in in_reps:

                if self.analysis_dict['salpha_rmsd'] : 

                    s_time = time.perf_counter()

                    print(f"Claculating Alpha-Helix RMSD...\n")
                    ps = restrict_atoms({'trajectory':self.trj[i],
                                         'selection':f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"})
                    # ps = self.trj[i]
                    # ps.restrict_atoms(ps.topology.select(f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"))
                    # ps=self.p_trj[i].restrict_atoms(self.p_trj[i].topology.select(f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"))
                    # ps.center_coordinates()
                    self.sa[i] = calc_SA(ps,self.helix, self.prot_start_res_num, self.prot_end_res_num)
                    
                    self.write_json_file(self.sa, self.out_file_dict['sa'], write=write)
                    del ps
                    reset_dict(self.sa)

                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                if self.analysis_dict['pp_rmsd'] : 

                    s_time = time.perf_counter()
                    print(f"Claculating Poly-prolein RMSD...\n")
                    ps = restrict_atoms({'trajectory':self.trj[i],
                                         'selection':f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"})
                    # ps = self.trj[i]
                    # ps.restrict_atoms(ps.topology.select(f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"))
                    # ps=self.p_trj[i].restrict_atoms(self.p_trj[i].topology.select(f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"))
                    # ps.center_coordinates()
                    self.pp_rmsd[i] = calc_SA(ps,self.pp, self.prot_start_res_num, self.prot_end_res_num)
                    
                    self.write_json_file(self.pp_rmsd, self.out_file_dict['pp_rmsd'], write=write)
                    del ps
                    reset_dict(self.pp)
                    
                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                if self.analysis_dict['rg'] : 
                    s_time = time.perf_counter()
                    print(f"Claculating Rg...\n")
                    self.rg[i] = calc_rg(self.trj[i])
                    
                    self.write_json_file(self.rg, self.out_file_dict['rg'], write=write)
                    reset_dict(self.rg)

                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                if self.analysis_dict['ss'] : 

                    s_time = time.perf_counter()    
                    print(f"Claculating SS propenseties...\n")
                    self.alphabeta_alpharight[i],self.alphabeta_betasheet[i],self.alphabeta_ppII[i] =  calc_phipsi(self.trj[i])
                    ps = restrict_atoms({'trajectory':self.trj[i],
                                         'selection':f"resid {self.prot_start_res_num} to {self.prot_end_res_num}"})
                    # ps.restrict_atoms(ps.topology.select(f"resid {self.prot_start_res_num} to {self.prot_end_res_num}"))
                    # ps.center_coordinates()
                    self.helix_contant[i], self.sheet_contant[i] = ss(ps)
                        
                    self.write_json_file(self.alphabeta_alpharight, self.out_file_dict['aright'], write=write)
                    self.write_json_file(self.alphabeta_betasheet, self.out_file_dict['sheet'], write=write)
                    self.write_json_file(self.alphabeta_ppII, self.out_file_dict['pp'], write=write)

                    self.write_json_file(self.helix_contant, self.out_file_dict['ss_h'], write=write)
                    self.write_json_file(self.sheet_contant, self.out_file_dict['ss_s'], write=write)
                    reset_dict([self.alphabeta_alpharight,self.alphabeta_betasheet,self.alphabeta_ppII,self.helix_contant, self.sheet_contant])
                    del ps
                    # reset_dict(self.alphabeta_betasheet)
                    # reset_dict(self.alphabeta_ppII)
                    # reset_dict(self.helix_contant)
                    # reset_dict(self.sheet_contant)
                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                if self.analysis_dict['ba'] : 

                    s_time = time.perf_counter()
                    print(f"Claculating average bend angle...\n")
                    self.bend_angle[i] = ave_angle(self.trj[i], [0,10,19])
                    self.write_json_file(self.bend_angle, self.out_file_dict['ba'], write=write)
                    reset_dict(self.bend_angle)

                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

            if self.analysis_dict['p_cm'] : 
                s_time = time.perf_counter()
                with Pool() as p:
                    rep_range=[i for i in in_reps] 

                    print(f"Claculating contact matrix...\n")
                    # self.p_contact_map, self.p_contact_distance_map = zip(*p.map(self.p_cm, rep_range))
                    self.p_contact_map = p.map(self.p_cm, rep_range)

                    self.p_contact_map = dict_conv(self.p_contact_map, in_reps, rest=rest)
                    # self.p_contact_distance_map = dict_conv(self.p_contact_distance_map, in_reps, rest=rest)

                    self.write_json_file(self.p_contact_map, self.out_file_dict['p_cm'], write=write)
                    reset_dict([self.p_contact_map,self.p_contact_distance_map])
                    # reset_dict(self.p_contact_distance_map)
                
                e_time = time.perf_counter()
                print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

        else : 

            for i in in_reps:

                if self.analysis_dict['salpha_rmsd'] : 
                    s_time = time.perf_counter()
                    print(f"Claculating Alpha-Helix RMSD...\n")
                    ps = restrict_atoms({'trajectory':self.trj[i],
                                         'selection':f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"})
                    # ps = self.trj[i]
                    # ps.restrict_atoms(ps.topology.select(f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"))
                    # ps=self.p_trj[i].restrict_atoms(self.p_trj[i].topology.select(f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"))
                    # ps.center_coordinates()
                    self.sa[i] = calc_SA(ps,self.helix, self.prot_start_res_num, self.prot_end_res_num)
                    
                    self.write_json_file(self.sa, self.out_file_dict['sa'], write=write)
                    del ps
                    reset_dict(self.sa)
                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                if self.analysis_dict['pp_rmsd'] : 
                    s_time = time.perf_counter()
                    print(f"Claculating Poly-prolein RMSD...\n")
                    ps = restrict_atoms({'trajectory':self.trj[i],
                                         'selection':f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"})
                    # ps = self.trj[i]
                    # ps.restrict_atoms(ps.topology.select(f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"))
                    # ps=self.p_trj[i].restrict_atoms(self.p_trj[i].topology.select(f"resid {self.prot_start_res_num} to {self.prot_end_res_num} and name CA"))
                    # ps.center_coordinates()
                    self.pp_rmsd[i] = calc_SA(ps, self.pp, self.prot_start_res_num, self.prot_end_res_num)
                    
                    self.write_json_file(self.pp_rmsd, self.out_file_dict['pp_rmsd'], write=write)
                    del ps
                    reset_dict(self.pp)
                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                if self.analysis_dict['rg'] : 
                    s_time = time.perf_counter()
                    print(f"Claculating Rg...\n")
                    self.rg[i] = calc_rg(self.trj[i])
                    
                    self.write_json_file(self.rg, self.out_file_dict['rg'], write=write)
                    self.rg[i] = calc_rg(self.trj[i])
                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                if self.analysis_dict['ss'] : 
                    s_time = time.perf_counter()    
                    print(f"Claculating SS propenseties...\n")
                    self.alphabeta_alpharight[i],self.alphabeta_betasheet[i],self.alphabeta_ppII[i] =  calc_phipsi(self.trj[i])
                    ps = restrict_atoms({'trajectory':self.trj[i],
                                         'selection':f"resid {self.prot_start_res_num} to {self.prot_end_res_num}"})
                    # ps.restrict_atoms(ps.topology.select(f"resid {self.prot_start_res_num} to {self.prot_end_res_num}"))
                    self.helix_contant[i], self.sheet_contant[i] = ss(ps)
                        
                    self.write_json_file(self.alphabeta_alpharight, self.out_file_dict['aright'], write=write)
                    self.write_json_file(self.alphabeta_betasheet, self.out_file_dict['sheet'], write=write)
                    self.write_json_file(self.alphabeta_ppII, self.out_file_dict['pp'], write=write)

                    self.write_json_file(self.helix_contant, self.out_file_dict['ss_h'], write=write)
                    self.write_json_file(self.sheet_contant, self.out_file_dict['ss_s'], write=write)
                    reset_dict([self.alphabeta_alpharight,self.alphabeta_betasheet,self.alphabeta_ppII,self.helix_contant, self.sheet_contant])
                    del ps
                    # reset_dict(self.alphabeta_betasheet)
                    # reset_dict(self.alphabeta_ppII)
                    # reset_dict(self.helix_contant)
                    # reset_dict(self.sheet_contant)
                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                if self.analysis_dict['ba'] : 

                    s_time = time.perf_counter()
                    print(f"Claculating average bend angle...\n")
                    self.bend_angle[i] = ave_angle(self.trj[i], [0,10,19])
                    self.write_json_file(self.bend_angle, self.out_file_dict['ba'], write=write)
                    reset_dict(self.bend_angle)
                    e_time = time.perf_counter()
                    print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                # print(f"Claculating Kd...\n")
            
            if self.analysis_dict['kd'] :
                s_time = time.perf_counter()
                with Pool() as p:
                    rep_range=[i for i in in_reps]

                    if self.w : 

                        print(f"Claculating CONTACT MATRIX and Kd...\n")
                        self.average_ligand_contacts_rw, self.average_ligand_contacts, self.contact_matrix = zip(*p.map(self.contact_matrix_out,rep_range))

                        self.average_ligand_contacts_rw = dict_conv(self.average_ligand_contacts_rw, in_reps, rest=rest)
                        self.average_ligand_contacts = dict_conv(self.average_ligand_contacts, in_reps, rest=rest)
                        self.contact_matrix = dict_conv(self.contact_matrix, in_reps, rest=rest)

                        self.bound_fraction, self.kd = zip(*p.map(self.kd_out,rep_range))
                        __, self.kd_raw = zip(*p.map(self.kd_out_raw,rep_range))

                        self.bound_fraction = dict_conv(self.bound_fraction, in_reps, rest=rest)
                        self.kd = dict_conv(self.kd, in_reps, rest=rest)
                        self.kd_raw = dict_conv(self.kd_raw, in_reps, rest=rest)
                    
                        if self.analysis_dict['kd_bf_time'] : 
                            
                            self.kd_bf_time = p.map(self.kd_time_out,rep_range)
                            self.kd_bf_time = dict_conv(self.kd_bf_time, in_reps, rest=rest)

                            self.write_json_file(self.kd_bf_time, self.out_file_dict['kd_bf_time'], write=write)

                        self.write_json_file(self.average_ligand_contacts_rw, self.out_file_dict['lc_rw'], write=write)
                        self.write_json_file(self.average_ligand_contacts, self.out_file_dict['lc'], write=write)
                        self.write_json_file(self.kd, self.out_file_dict['kd'], write=write)
                        self.write_json_file(self.kd_raw, self.out_file_dict['kd_raw'], write=write)

                        self.write_json_file(self.bound_fraction, self.out_file_dict['bf'], write=write)
                        self.write_json_file(self.contact_matrix, self.out_file_dict['cm'], write=write)

                        # self.kd_bf_time = zip(*p.map(self.kd_time_out,rep_range))
                        # self.kd_bf_time = dict_conv(self.kd_bf_time, in_reps, rest=rest)
                        # reset_dict([self.average_ligand_contacts_rw,self.average_ligand_contacts,self.contact_matrix])


                    else: 

                        print(f"Claculating CONTACT MATRIX and Kd...\n")
                        self.average_ligand_contacts, self.contact_matrix = zip(*p.map(self.contact_matrix_out,rep_range))
                        self.average_ligand_contacts = dict_conv(self.average_ligand_contacts, in_reps, rest=rest)
                        self.contact_matrix = dict_conv(self.contact_matrix, in_reps, rest=rest)

                        self.bound_fraction, self.kd = zip(*p.map(self.kd_out,rep_range))
                        self.bound_fraction = dict_conv(self.bound_fraction, in_reps, rest=rest)
                        self.kd = dict_conv(self.kd, in_reps, rest=rest)

                        if self.analysis_dict['kd_bf_time'] :
                                
                            self.kd_bf_time = p.map(self.kd_time_out,rep_range)
                            self.kd_bf_time = dict_conv(self.kd_bf_time, in_reps, rest=rest)

                            self.write_json_file(self.kd_bf_time, self.out_file_dict['kd_bf_time'], write=write)

                        self.write_json_file(self.average_ligand_contacts, self.out_file_dict['lc'], write=write)
                        self.write_json_file(self.kd, self.out_file_dict['kd'], write=write)
                        self.write_json_file(self.bound_fraction, self.out_file_dict['bf'], write=write)
                        self.write_json_file(self.contact_matrix, self.out_file_dict['cm'], write=write)
                        # reset_dict([self.average_ligand_contacts,self.contact_matrix,self.kd_bf_time ])
                
                for i in in_reps : 

                    if self.w  :
                        
                        self.average_ligand_contacts_rw_bf[i] = bf(self.average_ligand_contacts_rw[i],self.bound_fraction[i][0])
                        self.average_ligand_contacts_bf[i] = bf(self.average_ligand_contacts[i],self.bound_fraction[i][0])
                        reset_dict([self.average_ligand_contacts_rw,self.average_ligand_contacts,self.contact_matrix,
                                    self.average_ligand_contacts_rw_bf,self.average_ligand_contacts_bf ])

                    else:

                        self.average_ligand_contacts_bf[i] = bf(self.average_ligand_contacts[i],self.bound_fraction[i][0])
                        reset_dict([self.average_ligand_contacts,self.contact_matrix,self.kd_bf_time,
                                     self.average_ligand_contacts_bf])

                e_time = time.perf_counter()
                print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')




            with Pool() as p:
                rep_range=[i for i in in_reps]
                
                if self.w :

                    if self.analysis_dict['charge'] :

                        s_time = time.perf_counter()
                        print(f"Claculating charge...\n")
                        self.charge_re, self.charge_fraction = zip(*p.map(self.charge_out,rep_range))
                        self.charge_re = dict_conv(self.charge_re, in_reps, rest=rest)
                        self.charge_fraction = dict_conv(self.charge_fraction, in_reps, rest=rest)

                        for i in in_reps : 

                            self.charge_re_bf[i], self.charge_fraction_bf[i] = bf(self.charge_re[i] , self.bound_fraction[i][0]), bf(self.charge_fraction[i] , self.bound_fraction[i][0])

                        self.write_json_file(self.charge_re, self.out_file_dict['charge_rw:all'], write=write)
                        self.write_json_file(self.charge_fraction, self.out_file_dict['charge:all'], write=write)
                        self.write_json_file(self.charge_re_bf, self.out_file_dict['charge_rw:bf'], write=write)
                        self.write_json_file(self.charge_fraction_bf, self.out_file_dict['charge:bf'], write=write)
                        reset_dict([self.charge_re,self.charge_fraction, self.charge_re_bf, self.charge_fraction_bf ])
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    if self.analysis_dict['aro'] : 

                        s_time  = time.perf_counter()
                        print(f"Claculating aro...\n")
                        self.aro_interactions_re, self.aro_interactions, self.aro_binary_contacts, self.stackparams = zip(*p.map(self.aro_out,rep_range))
                        self.aro_interactions_re = dict_conv(self.aro_interactions_re, in_reps, rest=rest)
                        self.aro_interactions = dict_conv(self.aro_interactions, in_reps, rest=rest)
                        # self.aro_binary_contacts = dict_conv(self.aro_binary_contacts, in_reps, rest=rest)
                        self.stackparams = dict_conv(self.stackparams, in_reps, rest=rest)

                        for i in in_reps:

                            self.aro_interactions_re_bf[i], self.aro_interactions_bf[i] = bf(self.aro_interactions_re[i] , self.bound_fraction[i][0]), bf(self.aro_interactions[i] , self.bound_fraction[i][0])

                        self.write_json_file(self.aro_interactions_re, self.out_file_dict['aro_rw:all'], write=write)
                        self.write_json_file(self.aro_interactions, self.out_file_dict['aro:all'], write=write)
                        # self.write_json_file(self.aro_binary_contacts, self.out_file_dict['aro:binary'], write=write)
                        self.write_json_file(self.aro_interactions_re_bf, self.out_file_dict['aro_rw:bf'], write=write)
                        self.write_json_file(self.aro_interactions_bf, self.out_file_dict['aro:bf'], write=write)
                        reset_dict([self.aro_interactions_re,self.aro_interactions, self.aro_interactions_re_bf, self.aro_interactions_bf ,
                                    self.aro_binary_contacts,self.stackparams])
                        e_time = time.perf_counter()

                    if self.analysis_dict['hyphob'] : 
                        
                        s_time = time.perf_counter()
                        print(f"Claculating hphob...\n")
                        self.hydro_interactions_re, self.hydro_interactions, self.hydro_binary = zip(*p.map(self.hydro_out, rep_range))
                        self.hydro_interactions_re = dict_conv(self.hydro_interactions_re, in_reps, rest=rest)
                        self.hydro_interactions = dict_conv(self.hydro_interactions, in_reps, rest=rest)
                        # self.hydro_binary = dict_conv(self.hydro_binary, in_reps, rest=rest)

                        for i in in_reps:

                            self.hydro_interactions_re_bf[i], self.hydro_interactions_bf[i] = bf(self.hydro_interactions_re[i] , self.bound_fraction[i][0]), bf(self.hydro_interactions[i] , self.bound_fraction[i][0])

                        self.write_json_file(self.hydro_interactions_re, self.out_file_dict['hyph_rw:all'], write=write)
                        self.write_json_file(self.hydro_interactions, self.out_file_dict['hyph:all'], write=write)
                        # self.write_json_file(self.hydro_binary, self.out_file_dict['hyph:binary'], write=write)
                        self.write_json_file(self.hydro_interactions_re_bf, self.out_file_dict['hyph_rw:bf'], write=write)
                        self.write_json_file(self.hydro_interactions_bf, self.out_file_dict['hyph:bf'], write=write)
                        reset_dict([self.hydro_interactions_re,self.hydro_interactions, self.hydro_interactions_re_bf, self.hydro_interactions_bf ,
                                    self.hydro_binary])
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    if self.analysis_dict['hbond'] : 

                        s_time = time.perf_counter()
                        print(f"Claculating hbond...\n")
                        self.hbond_interactions_re, self.hbond_interactions, self.hbond_binary =  zip(*p.map(self.hbond_out, rep_range))
                        self.hbond_interactions_re = dict_conv(self.hbond_interactions_re, in_reps, rest=rest)
                        self.hbond_interactions = dict_conv(self.hbond_interactions, in_reps, rest=rest)
                        # self.hbond_binary = dict_conv(self.hbond_binary, in_reps, rest=rest)

                        for i in in_reps:

                            self.hbond_interactions_re_bf[i], self.hbond_interactions_bf[i] = bf(self.hbond_interactions_re[i] , self.bound_fraction[i][0]), bf(self.hbond_interactions[i] , self.bound_fraction[i][0])

                        self.write_json_file(self.hbond_interactions_re, self.out_file_dict['hb_rw:all'], write=write)
                        self.write_json_file(self.hbond_interactions, self.out_file_dict['hb:all'], write=write)
                        # self.write_json_file(self.hbond_binary, self.out_file_dict['hb:binary'], write=write)
                        self.write_json_file(self.hbond_interactions_re_bf, self.out_file_dict['hb_rw:bf'], write=write)
                        self.write_json_file(self.hbond_interactions_bf, self.out_file_dict['hb:bf'], write=write)
                        reset_dict([self.hbond_interactions_re,self.hbond_interactions, self.hbond_interactions_re_bf, self.hbond_interactions_bf ,
                                    self.hbond_binary])
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    if self.analysis_dict['p_cm'] : 

                        s_time = time.perf_counter()
                        print(f"Claculating protein contact matrix...\n")
                        # self.p_contact_map, self.p_contact_distance_map = zip(*p.map(self.p_cm, rep_range))
                        self.p_contact_map = p.map(self.p_cm, rep_range)

                        self.p_contact_map = dict_conv(self.p_contact_map, in_reps, rest=rest)
                        # self.p_contact_distance_map = dict_conv(self.p_contact_distance_map, in_reps, rest=rest)

                        self.write_json_file(self.p_contact_map, self.out_file_dict['p_cm_rw'], write=write)
                        # self.write_json_file(self.p_contact_distance_map, self.out_file_dict['p_cd'], write=write)
                        reset_dict([self.p_contact_map,self.p_contact_distance_map])
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    if self.analysis_dict['l_cm'] : 

                        s_time = time.perf_counter()
                        print(f"Claculating ligand-protein dual contact matrix...\n")
                        self.l_contact_map = p.map(self.l_cm, rep_range)
                        self.l_contact_map = dict_conv(self.l_contact_map, in_reps, rest=rest)

                        self.write_json_file(self.l_contact_map, self.out_file_dict['l_cm_rw'], write=write)
                        reset_dict(self.l_contact_map)
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    pass

                else:
                    
                    if self.analysis_dict['charge'] :

                        s_time = time.perf_counter()
                        print(f"Claculating charge...\n")
                        self.charge_fraction = p.map(self.charge_out,rep_range)
                        self.charge_fraction = dict_conv(self.charge_fraction, in_reps, rest=rest)

                        for i in in_reps : 

                            self.charge_fraction_bf[i] = bf(self.charge_fraction[i] , self.bound_fraction[i][0])
                        
                        self.write_json_file(self.charge_fraction, self.out_file_dict['charge:all'], write=write)
                        self.write_json_file(self.charge_fraction_bf, self.out_file_dict['charge:bf'], write=write)
                        reset_dict([self.charge_fraction, self.charge_fraction_bf ])
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    if self.analysis_dict['aro'] : 

                        s_time = time.perf_counter()
                        print(f"Claculating aro...\n")
                        self.aro_interactions, self.aro_binary_contacts, self.stackparams = zip(*p.map(self.aro_out,rep_range))
                        self.aro_interactions = dict_conv(self.aro_interactions, in_reps, rest=rest)
                        # self.aro_binary_contacts = dict_conv(self.aro_binary_contacts, in_reps, rest=rest)
                        self.stackparams = dict_conv(self.stackparams, in_reps, rest=rest)

                        for i in in_reps:

                            self.aro_interactions_bf[i] = bf(self.aro_interactions[i] , self.bound_fraction[i][0])

                        self.write_json_file(self.aro_interactions, self.out_file_dict['aro:all'], write=write)
                        # self.write_json_file(self.aro_binary_contacts, self.out_file_dict['aro:binary'], write=write)
                        self.write_json_file(self.aro_interactions_bf, self.out_file_dict['aro:bf'], write=write)
                        reset_dict([self.aro_interactions, self.aro_interactions_bf, self.aro_binary_contacts,self.stackparams])
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    if self.analysis_dict['hyphob'] :

                        s_time = time.perf_counter()
                        print(f"Claculating hphob...\n")
                        self.hydro_interactions, self.hydro_binary = zip(*p.map(self.hydro_out, rep_range))
                        self.hydro_interactions = dict_conv(self.hydro_interactions, in_reps, rest=rest)
                        # self.hydro_binary = dict_conv(self.hydro_binary, in_reps, rest=rest)

                        for i in in_reps:

                            self.hydro_interactions_bf[i] = bf(self.hydro_interactions[i] , self.bound_fraction[i][0])

                        self.write_json_file(self.hydro_interactions, self.out_file_dict['hyph:all'], write=write)
                        # self.write_json_file(self.hydro_binary, self.out_file_dict['hyph:binary'], write=write)
                        self.write_json_file(self.hydro_interactions_bf, self.out_file_dict['hyph:bf'], write=write)
                        reset_dict([self.hydro_interactions, self.hydro_interactions_bf ,self.hydro_binary])
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    if self.analysis_dict['hbond'] : 

                        s_time = time.perf_counter()
                        print(f"Claculating hbond...\n")
                        self.hbond_interactions, self.hbond_binary =  zip(*p.map(self.hbond_out, rep_range))
                        self.hbond_interactions = dict_conv(self.hbond_interactions, in_reps, rest=rest)
                        # self.hbond_binary = dict_conv(self.hbond_binary, in_reps, rest=rest)

                        for i in in_reps:

                            self.hbond_interactions_bf[i] = bf(self.hbond_interactions[i] , self.bound_fraction[i][0])

                        self.write_json_file(self.hbond_interactions, self.out_file_dict['hb:all'], write=write)
                        # self.write_json_file(self.hbond_binary, self.out_file_dict['hb:binary'], write=write)
                        self.write_json_file(self.hbond_interactions_bf, self.out_file_dict['hb:bf'], write=write)
                        reset_dict([self.hbond_interactions, self.hbond_interactions_bf ,self.hbond_binary])
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    if self.analysis_dict['p_cm'] : 

                        s_time = time.perf_counter()
                        print(f"Claculating protein contact matrix...\n")
                        # self.p_contact_map, self.p_contact_distance_map = zip(*p.map(self.p_cm, rep_range))
                        self.p_contact_map = p.map(self.p_cm, rep_range)

                        self.p_contact_map = dict_conv(self.p_contact_map, in_reps, rest=rest)
                        # self.p_contact_distance_map = dict_conv(self.p_contact_distance_map, in_reps, rest=rest)

                        self.write_json_file(self.p_contact_map, self.out_file_dict['p_cm'], write=write)
                        # self.write_json_file(self.p_contact_distance_map, self.out_file_dict['p_cd'], write=write)
                        reset_dict([self.p_contact_map,self.p_contact_distance_map])
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

                    if self.analysis_dict['l_cm'] : 

                        s_time = time.perf_counter()
                        print(f"Claculating ligand-protein dual contact matrix...\n")
                        self.l_contact_map = p.map(self.l_cm, rep_range)
                        self.l_contact_map = dict_conv(self.l_contact_map, in_reps, rest=rest)

                        self.write_json_file(self.l_contact_map, self.out_file_dict['l_cm'], write=write)
                        reset_dict(self.l_contact_map)
                        e_time = time.perf_counter()
                        print(f'Time taken : {time.strftime("%H:%M:%S", time.gmtime(e_time - s_time))} seconds\n')

