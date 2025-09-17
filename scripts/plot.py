import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
import itables
from itables import init_notebook_mode
init_notebook_mode(all_interactive=True)
import pandas as pd

sys.path.insert(0, './scripts')
from Block_analysis import *
from small_utilities import *

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

def compute_temperatures(temp_range:list, nreps:int):
    from numpy import log, exp
    tlow, thigh = temp_range
    temps = []
    for i in range(nreps):
        temps.append(tlow*exp((i)*log(thigh/tlow)/(nreps-1)))

    return np.array(temps)

def make_seq_pdb(pdb_path:str):

    import mdtraj as md

    aa_dict = {
        'A': ['ALA', 'Alanine'],
        'R': ['ARG', 'Arginine'],
        'N': ['ASN', 'Asparagine'],
        'D': ['ASP', 'Aspartic-acid'],
        'C': ['CYS', 'Cysteine'],
        'E': ['GLU', 'Glutamic-acid'],
        'Q': ['GLN', 'Glutamine'],
        'G': ['GLY', 'Glycine'],
        'H': ['HIS', 'Histidine'],
        'I': ['ILE', 'Isoleucine'],
        'L': ['LEU', 'Leucine'],
        'K': ['LYS', 'Lysine'],
        'M': ['MET', 'Methionine'],
        'F': ['PHE', 'Phenylalanine'],
        'P': ['PRO', 'Proline'],
        'S': ['SER', 'Serine'],
        'T': ['THR', 'Threonine'],
        'W': ['TRP', 'Tryptophan'],
        'Y': ['TYR', 'Tyrosine'],
        'V': ['VAL', 'Valine']
    }

    pdb = md.load(pdb_path)
    fasta = pdb.topology.to_fasta()

    seq=np.zeros(len(fasta), dtype='U3')

    for i in range(len(fasta)):
        aa = fasta[i][0]
        if aa in aa_dict:
            seq[i] = aa_dict[aa][0]
        else:
            print(f"Warning: Unknown amino acid {aa} at position {i+1}. Using 'UNK'.")

    return fasta, seq

def make_seq( inp_seq: list):

    import re

    aa_dict = {
        'ALA': ['A', 'Alanine'],
        'ARG': ['R', 'Arginine'],
        'ASN': ['N', 'Asparagine'],
        'ASP': ['D', 'Aspartic acid'],
        'CYS': ['C', 'Cysteine'],
        'GLU': ['E', 'Glutamic acid'],
        'GLN': ['Q', 'Glutamine'],
        'GLY': ['G', 'Glycine'],
        'HIS': ['H', 'Histidine'],
        'ILE': ['I', 'Isoleucine'],
        'LEU': ['L', 'Leucine'],
        'LYS': ['K', 'Lysine'],
        'MET': ['M', 'Methionine'],
        'PHE': ['F', 'Phenylalanine'],
        'PRO': ['P', 'Proline'],
        'SER': ['S', 'Serine'],
        'THR': ['T', 'Threonine'],
        'TRP': ['W', 'Tryptophan'],
        'TYR': ['Y', 'Tyrosine'],
        'VAL': ['V', 'Valine']
    }

    aa_map = {k: v[0] for k, v in aa_dict.items()}
    
    b, c = [], []
    pattern = re.compile(r"([a-zA-Z]+)([0-9]+)")

    for entry in inp_seq:
        match = pattern.match(entry)
        if not match:
            continue  # skip if it doesn't match the expected pattern

        res_name, res_num = match.groups()
        res_name = res_name.upper()
        one_letter = aa_map.get(res_name, '?')  # default to '?' if not found
        b.append(f"{one_letter}_{res_num}")
        c.append(one_letter)

    return b, c

def _plot_cm_all(input: dict, cbar_label: str, file_name: str = None, save_fig: bool = False, show_fig: bool = False,
                title_dict: dict = None, offset=None, **kwargs):

    import matplotlib.pyplot as plt
    import numpy as np

    # Default plot parameters
    plot_args = {
        'vmin': 0.0, 'vmax': 0.5, 'fig_size': (30, 14), 'cax_coor': None, 'cmap': 'jet', 'aspect': 'auto',
        'rotation': {'x': 90, 'y': 0},
        'tick_size': {'x': 18, 'y': 18, 'cax': 25},
        'label_size': {'x': 30, 'y': 30, 'cax': 30},
        'title_size': 20, 'dpi': 310,
        'labels': {'x': "Residues", 'y': "Residues"},
        'nrows': 2, 'ncols': 5,
        'xticks': None, 'yticks': None,
        'tick_interval': 2
    }

    # Override with user kwargs
    for key, value in kwargs.items():
        if key in plot_args:
            plot_args[key] = value

    # Setup subplot grid
    fig, axes = plt.subplots(
        plot_args['nrows'], plot_args['ncols'],
        figsize=plot_args['fig_size'],
        sharex=True, sharey=True,
        dpi=plot_args['dpi'],
        layout='constrained'
    )

    images = []

    for i in input.keys():
        p, q = np.unravel_index(i, (plot_args['nrows'], plot_args['ncols']))
        ax = axes[p, q]
        im = ax.imshow(
            np.array(input[i]),
            vmin=plot_args['vmin'], vmax=plot_args['vmax'],
            cmap=plot_args['cmap'], aspect=plot_args['aspect']
        )

        ax.invert_yaxis()
        ax.tick_params(axis='both', which='both', direction='out')

        xticks = plot_args.get('xticks')
        yticks = plot_args.get('yticks')
        interval = plot_args['tick_interval']

        if xticks is not None:
            ax.set_xticks(range(0, len(xticks) + (offset or 0) - 1, interval))
            ax.set_xticklabels(xticks[::interval], rotation=plot_args['rotation']['x'], size=plot_args['tick_size']['x'])

        if yticks is not None:
            ax.set_yticks(range(0, len(yticks) + (offset or 0) - 1, interval))
            ax.set_yticklabels(yticks[::interval], rotation=plot_args['rotation']['y'], size=plot_args['tick_size']['y'])

        title = title_dict.get(i, f"Plot {i}") if title_dict else f"Plot {i}"
        ax.set_title(title, size=plot_args['title_size'], pad=10)

        images.append(im)

    # Colorbar
    if plot_args['cax_coor']:
        cax = fig.add_axes(plot_args['cax_coor'])
    else:
        last_ax = axes[plot_args['nrows'] - 1, plot_args['ncols'] - 1]
        cax = fig.add_axes([
            last_ax.get_position().x1 + 0.12,
            last_ax.get_position().y0 - 0.034,
            0.02,
            axes[0, 0].get_position().y1 - last_ax.get_position().y0 + 0.134
        ])

    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label(cbar_label, size=plot_args['label_size']['cax'])
    cbar.ax.tick_params(labelsize=plot_args['tick_size']['cax'])

    # Axis labels
    fig.supxlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'])
    fig.supylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'])

    # Save/show
    if save_fig:
        assert file_name, "file_name must be provided if save_fig is True"
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, dpi=plot_args['dpi'], bbox_inches='tight')

    if show_fig:
        plt.show()

def _prepare_plot_args_cm(custom_args: dict) -> dict:
    """Internal helper to prepare default + custom plot arguments."""
    plot_args = {
        'vmin': 0.0, 'vmax': 0.5, 'fig_size': (30, 14), 'cax_coor': [0.93, 0.2, 0.02, 0.6],
        'cmap': 'jet', 'aspect': 'auto',
        'rotation': {'x': 90, 'y': 0},
        'tick_size': {'x': 18, 'y': 18, 'cax': 25},
        'label_size': {'x': 30, 'y': 30, 'cax': 30},
        'title_size': 20, 'dpi': 310,
        'labels': {'x': "Residues", 'y': "Residues"}
    }
    if custom_args:
        plot_args.update(custom_args)
    return plot_args

def blocking_2d_hist(Y, X, y0:float=None, ymax:float=None, x0:float=None,
                   xmax:float=None, weights=None, chunk_width=10, bin_count=35):
    """blocking_2d_hist used to produce individual 2D histograms of chunk_width
    blocks of data. The width needs to be identified from reblocking analysis of
    each 2D variable. Bin count is up to you, but should be the same for all."""
    
    import numpy as np
    from Block_analysis import chunkIt

    assert len(Y) == len(X), "Y and X must have the same length"
    chunks = chunkIt(len(Y), chunk_width)
    bounds=np.array([[y0, ymax], [x0, xmax]]) + np.array([[-0.01, 0.01], [-0.01, 0.01]])
    bins = [np.linspace(bounds[0][0], bounds[0][1], bin_count+1),
            np.linspace(bounds[1][0], bounds[1][1], bin_count+1)]
    flattened_histos = np.zeros((bin_count**2,len(chunks)))
    for i, chunk in enumerate(chunks):
        histo, _, _ = np.histogram2d(
                            Y[chunk[0]:chunk[1]], 
                            X[chunk[0]:chunk[1]], 
                            bins, 
                            bounds, 
                            density=False, 
                            weights=weights
                            )   
        flattened_histos[:, i] = histo.flatten()
    xcenters = (bins[0][1:]+bins[0][:-1])/2
    ycenters = (bins[1][1:]+bins[1][:-1])/2

    return flattened_histos/chunk_width, xcenters, ycenters

def blocking_bootstrap_2d_fe(X, Y, bin_count,y0, ymax, x0, xmax,):

    import numpy as np
    import pyblock as pb
    from scipy.stats import bootstrap

    assert len(Y) == len(X), "Y and X must have the same length"

    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)

    
    reblock_data_X = pb.blocking.reblock(X)
    reblock_data_Y = pb.blocking.reblock(Y)

    opt_X = pb.blocking.find_optimal_block(
        len(X), reblock_data_X)[0]
    opt_Y = pb.blocking.find_optimal_block(
        len(Y), reblock_data_Y)[0]

    # block_length = reblock_data_X[opt_X].ndata if \
    #             reblock_data_X[opt_X].ndata > reblock_data_X[opt_Y].ndata \
    #                     else reblock_data_Y[opt_Y].ndata


    # print('Optimal block size for X:', reblock_data_X[opt_X].ndata)
    # print('Optimal block size for Y:', reblock_data_Y[opt_Y].ndata)

    # block_length = input("Enter the desired block length: ")
    # block_length = int(block_length)
    # if block_length <= 0:
    #     raise ValueError("Block length must be a positive integer.")
    block_length = reblock_data_X[opt_X].ndata if \
                reblock_data_X[opt_X].ndata > reblock_data_Y[opt_Y].ndata \
                        else reblock_data_Y[opt_Y].ndata
    
    skip = len(X) % block_length
    
    # print('Block length:', block_length)
    # print('Skip:', skip)

    options = {'Y':Y[skip:], 'X':X[skip:], 'y0':y0, 'ymax':ymax, 'x0':x0, 'xmax':xmax, 
                'chunk_width':block_length, 'bin_count':bin_count}

    hist_2d_for_fe, xcenters, ycenters = blocking_2d_hist(**options)

    def fe2D(x): 
        p = np.sum(x, axis=-1)/len(x)
        fe = -(0.001987*300)*np.log(p+0.000000001)
        return fe

    bootstrap_results = bootstrap((hist_2d_for_fe.T,), fe2D, n_resamples=1000, method='percentile')

    return bootstrap_results, xcenters, ycenters

#######################################################################################################################################################################################################################

class plots():

    def __init__(self, data_dir:str, out_dir:str, p_seq:list, json_files_dict:dict, num_of_replicas:int, rows_cols:dict={}, out_files_dict:dict={}, rep_demux_switch:str="on",
                 temp_range:list=[300,450], out_file_type:str='png', apo:bool=False, offset:int=0, dpi:int=310, **kwargs ):


        #TODO : add a **kwargs praser and check
        assert 'title_names' in kwargs.keys() or rep_demux_switch == "on", "Please provide title_names in kwargs or set rep_demux_switch to 'on'."
        # if 's' in kwargs.keys() : self.s = kwargs['s']
        # if 'rows' in kwargs.keys() : self.rows = kwargs['rows']
        # if 'cols' in kwargs.keys() : self.cols = kwargs['cols']
        
        self.apo=apo
        self.offset=offset
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.out_file_type = out_file_type
        self.nreps = num_of_replicas
        self.replica_temps = compute_temperatures(temp_range)

        print(f"Your temperatures are : \n{np.array([f'{i:6.3f}' for i in self.replica_temps])}\n")

        self.dpi = dpi

        if rows_cols : self.rows, self.cols = rows_cols['rows'], rows_cols['cols']
        elif not rows_cols : self.cols = 4 ; self.rows = self.nreps//self.cols


        if rep_demux_switch == "on" :

            self.s, self.title_names =['rep', 'demux'] , ['Replica', 'Demuxed replica']


        else : self.s, self.title_names = None, kwargs['title_names']


        self.p_seq, self.p_seq_a = make_seq(p_seq)

        self.in_files_dict = {
            'p_cm': False,
            'l_cm': False,
            'helix': False,
            'sheet': False,
            'rg': False,
            'kd_time': False,
            'lig_modes': {'aro': False,
                          'hyphob': False,
                          'hbond': False,
                          'charge': False,},
            'sa': False,
            'ba': False,
            'bf': False,
            'lig_contacts': False,
            'cm' : False,
            # 'rg_2d': False,
            # 'pp': False,
            # 'phipsi': False,
        }
        
        self.out_files_dict = {
            'p_cm': False,
            'l_cm': False,
            'helix': False,
            'sheet': False,
            'rg': False,
            'rg_hist': False,
            'bf_time': False,
            'lig_modes': {'aro': False,
                          'hyphob': False,
                          'hbond': False,
                          'charge': False,},
            'lig_contacts': False,
            'rg_2d': False,
            'sa': False,
            'ba': False,
            # 'pp': False,
            # 'phipsi': False,
            # 'rg_hist_2': False
        }

        if json_files_dict:
            try:
                for key, value in json_files_dict.items():
                    if key in self.in_files_dict:
                        self.in_files_dict[key] = value
                    if key in self.out_files_dict and 'out_files_dict' in locals():
                        self.out_files_dict[key] = out_files_dict.get(key, False)
            except Exception as e:
                print(f"[ERROR] Failed to update file dictionaries: {e}")
                print(f"Expected keys: {list(self.in_files_dict.keys())}")




    def _plot_cm_wrapper(self, cm_type: str, save_fig=False, show_fig=False, demux=False, plot_args_in=None):
        """Generic wrapper to plot contact map for given type (`p_cm` or `l_cm`)."""
        assert cm_type in self.in_files_dict, f"Invalid contact map type '{cm_type}'"
        assert self.in_files_dict[cm_type], f"Load the {cm_type} contact data before plotting!"

        in_file = f"{self.data_dir}/{self.in_files_dict[cm_type]}"
        in_data = load_json(in_file)

        out_suffix = "demux" if demux else "rep"
        out_file = f"{self.out_files_dict[cm_type]}_{out_suffix}.{self.out_file_type}"

        plot_args = _prepare_plot_args_cm(plot_args_in or {})
        _plot_cm_all(in_data, 'Contact Probability', out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, **plot_args)


    def plot_p_cm(self, save_fig: bool = False, show_fig: bool = False, demux: bool = False, plot_args_in: dict = None):
        self._plot_cm_wrapper('p_cm', save_fig, show_fig, demux, plot_args_in)


    def plot_l_cm(self, save_fig: bool = False, show_fig: bool = False, demux: bool = False, plot_args_in: dict = None):
        self._plot_cm_wrapper('l_cm', save_fig, show_fig, demux, plot_args_in)        

    