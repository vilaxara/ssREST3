# Scripts for small utilities
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_json(file:str, flag:str='r'):

    return json.load(open(file,flag))

def norm_weights(file_name):

    import numpy as np

    colvar=np.loadtxt(file_name,comments=['#','@'])
    num_cvs=len(colvar[0])-1

    kt=2.494339
    w=np.exp((colvar[:,num_cvs]/kt))

    max_=np.sum(w)
    w_norm=w/max_

    return num_cvs, w, w_norm

def compute_temperatures(temp_range:tuple, nreps:int):
    import numpy as np
    from math import exp, log
    tlow, thigh = temp_range
    temps = []
    for i in range(nreps):
        temps.append(np.round(tlow*exp((i)*log(thigh/tlow)/(nreps-1)),3))

    return np.array(temps)

def sequence_ticks(pdb:str):

    aa_dict = {'ALA' : ["A", "Alanine"], 'ARG' : ["R", "Arginine"], 'ASN' : ["N", "Asparagine"], 'ASP' : ["D", "Aspartic-acid"], 'CYS' : ["C", "Cysteine"], 
            'GLU' : ["E", "Glutamic-acid"], 'GLN' : ["Q", "Glutamine"], 'GLY' : ["G", "Glycine"], 'HIS' : ["H", "Histidine"], 'ILE' : ["I", "Isoleucine"],
            'LEU' : ["L", "Leucine"], 'LYS' : ["K", "Lysine"], 'MET' : ["M", "Methionine"], 'PHE' : ["F", "Phenylalanine"], 'PRO' : ["P", "Proline"], 
            'SER' : ["S", "Serine"], 'THR' : ["T", "Threonine"], 'TRP' : ["W", "Tryptophan"], 'TYR' : ["Y", "Tyrosine"], 'VAL' : ["V", "Valine"]}

    import numpy as np
    import mdtraj as md

    pdb = md.load(pdb)
    pdb = pdb.atom_slice(pdb.top.select('protein'))

    sequence = np.array([residue for residue in pdb.topology.residues]).astype(str)
    # print(sequence)

    def split_temp(inp:np.ndarray):
        import re

        exp=r"([a-z]+)([0-9]+)"
        out=[]

        for i in range(len(inp)):
            match = re.match(exp, inp[i], re.I)

            if match:
                items = match.groups()

            out.append(list(items))

        return np.array(out)

    a=split_temp(np.array(sequence))

    b=np.zeros(len(a),dtype=object)
    c=np.zeros(len(a),dtype=object)

    for i in range(0,len(a)):

        for k in aa_dict.keys():

            if k in a[i]:
                
                a[i][0] = aa_dict[k][0]

        b[i]=''.join(a[i])
        c[i]=a[i][0]

    return np.array(b) , np.array(c)

def sequence_ticks_1(sequence):

    aa_dict = {'ALA' : ["A", "Alanine"], 'ARG' : ["R", "Arginine"], 'ASN' : ["N", "Asparagine"], 'ASP' : ["D", "Aspartic-acid"], 'CYS' : ["C", "Cysteine"], 
            'GLU' : ["E", "Glutamic-acid"], 'GLN' : ["Q", "Glutamine"], 'GLY' : ["G", "Glycine"], 'HIS' : ["H", "Histidine"], 'ILE' : ["I", "Isoleucine"],
            'LEU' : ["L", "Leucine"], 'LYS' : ["K", "Lysine"], 'MET' : ["M", "Methionine"], 'PHE' : ["F", "Phenylalanine"], 'PRO' : ["P", "Proline"], 
            'SER' : ["S", "Serine"], 'THR' : ["T", "Threonine"], 'TRP' : ["W", "Tryptophan"], 'TYR' : ["Y", "Tyrosine"], 'VAL' : ["V", "Valine"]}

    import numpy as np

    def split_temp(inp:np.ndarray):
        import re

        exp=r"([a-z]+)([0-9]+)"
        out=[]

        for i in range(len(inp)):
            match = re.match(exp, inp[i], re.I)

            if match:
                items = match.groups()

            out.append(list(items))

        return np.array(out)

    a=split_temp(np.array(sequence))

    b=np.zeros(len(a),dtype=object)
    c=np.zeros(len(a),dtype=object)

    for i in range(0,len(a)):

        for k in aa_dict.keys():

            if k in a[i]:
                
                a[i][0] = aa_dict[k][0]

        b[i]=''.join(a[i])
        c[i]=a[i][0]

    return np.array(b) , np.array(c)

def make_dir(dir_name:str):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
