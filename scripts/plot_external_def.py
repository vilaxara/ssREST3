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

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
        b.append(f"{one_letter}{res_num}")
        c.append(one_letter)

    return b, c

def markers_colors(cm=plt.get_cmap('tab20'), nreps:int=20):
        
    import matplotlib
    cm =cm if isinstance(cm, matplotlib.colors.Colormap) else plt.get_cmap(cm)
    colors = cm(np.linspace(0, 1, nreps))
    # np.array([i for i in matplotlib.markers.MarkerStyle.markers.keys()])[:-4]
    
    return np.array(colors), np.array([i for i in matplotlib.markers.MarkerStyle.markers.keys() if i not in ['None', 'none', ' ', '', ',']])

############################################################################################################################################################################################################
def plot_cm_all(input: dict, cbar_label: str, file_name: str = None, save_fig: bool = False, show_fig: bool = False,
                title: list = None, offset=None, demux=None, out_file_type=None, out_dir=None, **kwargs):

    import matplotlib.pyplot as plt
    import numpy as np

    # Default plot parameters
    plot_args = {
        'vmin': 0.0, 
        'vmax': 0.5, 
        'fig_size': (30, 14), 
        'cax_coor': None, 
        'cmap': 'jet', 
        'aspect': 'auto',
        'rotation': {'x': 90, 'y': 0},
        'tick_size': {'x': 25, 'y': 25, 'cax': 25},
        'label_size': {'x': 30, 'y': 30, 'cax': 30},
        'title_size': 20, 'dpi': 310,
        'labels': {'x': "Residues", 'y': "Residues"},
        'nrows': 2, 'ncols': 5,
        'xticks': None, 'yticks': None,
        # 'tick_interval': 2
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
        # layout='constrained'
    )

    images = []

    a_x = 0

    for i in input.keys():
        p, q = np.unravel_index(a_x, (plot_args['nrows'], plot_args['ncols']))
        ax = axes[p, q]
        im = ax.imshow(
            np.array(input[i]),
            vmin=plot_args['vmin'], vmax=plot_args['vmax'],
            cmap=plot_args['cmap'], aspect=plot_args['aspect']
        )

        ax.invert_yaxis()
        ax.tick_params(axis='both', which='both', direction=plot_args['minor_ticks']['direction'] if 'minor_ticks' in plot_args else 'out')

        xticks = plot_args.get('xticks')
        yticks = plot_args.get('yticks')
        # interval = plot_args['tick_interval']

        if xticks is not None:
            ax.set_xticks(range(0, len(xticks) + (offset or 0) ))
            ax.set_xticklabels(xticks[::], rotation=plot_args['rotation']['x'], size=plot_args['tick_size']['x'])

        if yticks is not None:
            ax.set_yticks(range(0, len(yticks) + (offset or 0)))
            ax.set_yticklabels(yticks[::], rotation=plot_args['rotation']['y'], size=plot_args['tick_size']['y'])

        # title = title_dict.get(i, f"Plot {i}") if title_dict else f"Plot {i}"
        ax.set_title(title[a_x], size=plot_args['title_size'], pad=10)

        images.append(im)
        a_x+=1

    # Colorbar
    if plot_args['cax_coor']:
        cax = fig.add_axes(plot_args['cax_coor'])
    else:
        last_ax = axes[plot_args['nrows'] - 1, plot_args['ncols'] - 1]
        cax = fig.add_axes([
            last_ax.get_position().x1 + 0.05,
            last_ax.get_position().y0 , #- 0.0,
            0.03,
            axes[0, 0].get_position().y1 - last_ax.get_position().y0, # + 0.0
        ])
           
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label(cbar_label, size=plot_args['label_size']['cax'])
    cbar.ax.tick_params(labelsize=plot_args['tick_size']['cax'])

    fig.text(0.5, 0.04, plot_args['labels']['x'], ha="center", fontsize=plot_args['label_size']['x'])
    fig.text(-0.05, 0.5, plot_args['labels']['y'], va="center", rotation="vertical", fontsize=plot_args['label_size']['y'])

    # Axis labels
    # fig.supxlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'])
    # fig.supylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'])

    # Save/show
    if save_fig:
        assert file_name, "file_name must be provided if save_fig is True"
        print(f"Saving figure to {file_name}")
        plt.savefig(f'{out_dir}/{file_name}{out_file_type}', dpi=plot_args['dpi'], bbox_inches='tight')

    if show_fig:
        plt.show()


def plot_rg_vs_temperature(rg: dict, temperature: list, nrep: int, show: bool = True, 
                           save: bool = False, out_dir: str = None, out_file_type: str = 'png', 
                           out_file: str = 'rg_vs_temp', **kwargs):
    
    import matplotlib.pyplot as plt
    import numpy as np

    # Default plot parameters
    plot_args = {
        'fig_size': (7, 5), 
        'rotation': {'x': 0, 'y': 0},
        'tick_size': {'x': 25, 'y': 25},
        'label_size': {'x': 25, 'y': 25},
        'title_size': 20, 'dpi': 310,
        'labels': {'x': "Temperature (K)", 'y': "<Rg> (nm)"},
        'xticks': None, 'yticks': None,
        'marker':'o', 'markersize':8, 
        'linestyle':'-', 'linewidth':2, 'color':None, 'alpha':0.3,
        'labelpad':10,
        'minor_ticks' : {'x_loc':50/5, 'y_loc':0.1/2, 
                        'length':3, 'width':1, 'color':'k', 'direction':'out'},
        # 'text_params': {'x_start':300, 'y_start':1.16, 'y_step':0.02, 
        #                 'y_substep':0.013, 'x_step':20, 'size':15, 'color': None}, 
        'bbox_inches' : 'tight',
        # 'tick_interval': 2
    }

    # Override with user kwargs
    for key, value in kwargs.items():
        if key in plot_args:
            plot_args[key] = value
            
    fig, ax = plt.subplots(figsize=plot_args['fig_size'])

    # Scatter plot of Rg means
    rg_means = [get_blockerror_pyblock_nanskip(np.array(rg[f'rep:{i}']))[0] for i in range(nrep)]
    rg_errors = [get_blockerror_pyblock_nanskip(np.array(rg[f'rep:{i}']))[1] for i in range(nrep)]
    # colors = [plt.cm.viridis(i / nrep) for i in range(nrep)]


    ax.plot(temperature, rg_means, marker=plot_args['marker'], markersize=plot_args['markersize'], 
            linestyle=plot_args['linestyle'], linewidth=plot_args['linewidth'], color=plot_args['color'])
    ax.fill_between(temperature, np.array(rg_means) - np.array(rg_errors), 
                    np.array(rg_means) + np.array(rg_errors), color=plot_args['color'], alpha=plot_args['alpha'])
    
    # Axis styling
    ax.tick_params(labelsize=plot_args['tick_size']['x'], axis='x')
    ax.tick_params(labelsize=plot_args['tick_size']['y'], axis='y')
    # ax.set_yticks(np.arange(1.0, 1.55, 0.1), [f"{i:2.1f}" for i in np.arange(1.0, 1.55, 0.1)])
    # ax.set_xticks(np.arange(300, 500, 50), np.arange(300, 500, 50))
    ax.set_yticks(plot_args['yticks'], [f"{i:2.1f}" for i in plot_args['yticks']])
    ax.set_xticks(plot_args['xticks'], plot_args['xticks'])

    ax.set_xlabel(plot_args['labels']['x'], labelpad=plot_args['labelpad'], size=plot_args['label_size']['x'])
    ax.set_ylabel(plot_args['labels']['x'], labelpad=plot_args['labelpad'], size=plot_args['label_size']['y'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(plot_args['linewidth'])
    ax.spines['bottom'].set_linewidth(plot_args['linewidth'])

    # Minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(plot_args['minor_ticks']['x_loc']))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(plot_args['minor_ticks']['y_loc']))
    ax.tick_params(axis='x', which='minor', length=plot_args['minor_ticks']['length'], 
                   width=plot_args['minor_ticks']['width'], color=plot_args['minor_ticks']['color'],
                   direction=plot_args['minor_ticks']['direction'])
    ax.tick_params(axis='y', which='minor', length=plot_args['minor_ticks']['length'], 
                   width=plot_args['minor_ticks']['width'], color=plot_args['minor_ticks']['color'],
                   direction=plot_args['minor_ticks']['direction'])

    plt.tight_layout()

    if show : plt.show()
    if save : 
        assert out_dir, "Please provide out_dir to save the figure."
        save_file = f"{out_dir}/{out_file}.{out_file_type}"
        print(f"Saving figure to {save_file}")
        plt.savefig(save_file, dpi=plot_args['dpi'], bbox_inches='tight')



def compute_time_array(input: list, timestep: float = 80) -> np.ndarray:
    import numpy as np

    n_frames = len(input)
    return np.array([(i * timestep) / 1e6 for i in range(n_frames)])

def compute_time_splits(
    input: np.ndarray,
    magic_num: float = 125,
    min_val: int = 10,
    max_val: int = 510,
    step: int = 10,
    timestep: float = 80,
    verbose: bool = True
) -> np.ndarray[int]:

    # Generate the time array internally
    time = compute_time_array(input, timestep)

    splits = []
    n_frames = time.size

    for i in range(step, max_val, step):
        idx = int(magic_num * i) + 1
        # cap at end of array to include trickle frames
        if idx >= n_frames:
            idx = n_frames

        count = idx  # number of frames in this segment
        if count in splits:
            # already recorded this split, skip but keep going
            if idx == n_frames:
                break
            else:
                continue

        last_time = time[count - 1]
        splits.append(count)

        if verbose:
            print(f"i={i:>3} → frames={count:>6}, time={last_time:.6f} μs")

        # once we've included the very last frame, stop
        if idx == n_frames:
            break

    return np.array(splits)

def plot_free_energy_1d(
    rg_series: np.ndarray,
    splits: list[int],
    magic_num: float,
    T: float = 300,
    bins: int = 50,
    blocks: int = 5,
    weights=None,
    cmap_name: str = 'tab20c',
    out_file: str = 'free_energy_plot',
    save_fig: bool = False,
    show_fig: bool = True,
    out_dir: str = None,
    **kwargs
):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plot_args = {
        'fig_size': (10,8), 
        'rotation': {'x': 0, 'y': 0},
        'tick_size': {'x': 25, 'y': 25},
        'label_size': {'x': 25, 'y': 25},
        'dpi': 310,
        'labels': {'x': 'Radius of Gyration (nm)', 'y': 'Free Energy (kcal/mol)'},
        'xticks': None, 'yticks': None,
        'linestyle':'-', 'linewidth':2, 'alpha':0.3,
        'labelpad':10,
        # 'minor_ticks' : {'x_loc':50/5, 'y_loc':0.1/2, 
        #                 'length':3, 'width':1, 'color':'k', 'direction':'out'},
        'bbox_inches' : 'tight',
        'legend_params' : {'loc': 'upper center', 'fontsize': 12, 'ncol': 5, 'bbox_to_anchor':(0.38, 1.0)}, 
        'grid': True,
    }

    # Override with user kwargs
    for key, value in kwargs.items():
        if key in plot_args:
            plot_args[key] = value
    
    fig, ax = plt.subplots(figsize=plot_args['fig_size'])
    cmap = plt.cm.get_cmap(cmap_name, len(splits))

    # helper to compute and plot one curve
    def _plot_segment(data, label, color):
        dG, centers, err = free_energy_1D_blockerror(
            data, T=T,
            x0=data.min(),
            xmax=data.max(),
            bins=bins,
            blocks=blocks,
            weights=weights
        )
        sns.lineplot(x=centers, y=dG, ax=ax, color=color, linewidth=plot_args['linewidth'], label=label)
        ax.fill_between(centers, dG - err, dG + err, color=color, alpha=plot_args['alpha'])

    # plot each truncated segment
    for idx, factor in enumerate(splits):
        end = int(magic_num * factor) + 1
        segment = rg_series[:end]
        _plot_segment(np.array(segment), label=f"{factor/100:.2f}", color=cmap(idx))

    # plot the full series
    _plot_segment(np.array(rg_series), label='full', color='k')

    # finalize axes
    ax.set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'])
    ax.set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'])
    ax.tick_params(axis='both', labelsize=plot_args['tick_size']['x'])
    ax.legend(loc=plot_args['legend_params']['loc'], fontsize=plot_args['legend_params']['fontsize'], 
              ncol=plot_args['legend_params']['ncol'] or min(len(splits)+1, 5), bbox_to_anchor=plot_args['legend_params']['bbox_to_anchor'])
    ax.grid(plot_args['grid'])
    plt.tight_layout()
    
    if show_fig : plt.show()
    if save_fig:
        assert out_dir, "Please provide out_dir to save the figure."
        save_file = f"{out_dir}/{out_file}.{kwargs.get('out_file_type', 'png')}"
        print(f"Saving figure to {save_file}")
        plt.savefig(save_file, dpi=plot_args['dpi'], bbox_inches=plot_args['bbox_inches'])

def plot_aggregate_rg(
    # time: np.ndarray,
    rg_series: np.ndarray,
    magic_num: float = 78.1,
    start_factor: int = 10,
    end_factor: int = 510,
    step: int = 10,
    timestep: float = 80.0,
    show_fig: bool = True,
    save_fig: bool = False,
    out_file: str = 'aggregate_rg_plot',
    out_file_type: str = 'png',
    out_dir: str = None,

    **kwargs
):

    import numpy as np
    import matplotlib.pyplot as plt

    plot_args = {
        'fig_size': (10,8), 
        'rotation': {'x': 0, 'y': 0},
        'tick_size': {'x': 25, 'y': 25},
        'label_size': {'x': 25, 'y': 25},
        'dpi': 310,
        'labels': {'x': r'Aggregate Time ($\mu$s)', 'y': r'<Rg> (nm)'},
        'xticks': None, 'yticks': None,
        'labelpad':10,
        'bbox_inches' : 'tight',
        'markersize': 10,
        'capsize': 5,
        'capthick': 2,
        'color': 'black',
        'fmt': 'o',
    }

    # Override with user kwargs
    for key, value in kwargs.items():
        if key in plot_args:
            plot_args[key] = value

    if not isinstance(rg_series, np.ndarray):
        rg_series=np.array(rg_series, dtype=float)

    time_agg = []
    means = []
    errors = []
    seen = set()

    time = compute_time_array(rg_series, timestep)

    n = time.size
    for factor in range(start_factor, end_factor, step):

        idx = int(magic_num * factor) + 1
        if idx > n:
            idx = n

        if idx in seen:
            if idx == n:
                break
            continue

        seen.add(idx)
        time_agg.append(time[idx - 1])
        mean, err = get_blockerror_pyblock_nanskip(rg_series[:idx])
        means.append(mean)
        errors.append(err)

        if idx == n:
            break

    time_agg = np.array(time_agg)
    means = np.array(means)
    errors = np.array(errors)

    fig, ax = plt.subplots(figsize=plot_args['fig_size'])
    ax.errorbar(
        time_agg, means, yerr=errors,
        fmt='o', color=plot_args['color'],
        markersize=plot_args['markersize'],
        capsize=plot_args['capsize'],
        capthick=plot_args['capthick'],
    )

    ax.tick_params(labelsize=plot_args['tick_size']['x'], axis='x')
    ax.tick_params(labelsize=plot_args['tick_size']['y'], axis='y')
    ax.set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=plot_args['labelpad'])
    ax.set_ylabel(plot_args['labels']['y'], size= plot_args['label_size']['y'], labelpad=plot_args['labelpad'])
    ax.grid(False)
    plt.tight_layout()

    if show_fig : plt.show()
    if save_fig:
        assert out_dir, "Please provide out_dir to save the figure."
        save_file = f"{out_dir}/{out_file}.{out_file_type}"
        print(f"Saving figure to {save_file}")
        plt.savefig(save_file, dpi=plot_args['dpi'], bbox_inches=plot_args['bbox_inches'])


############################################################################################################################################################################################################

def prepare_plot_args_cm(custom_args: dict) -> dict:
    """Internal helper to prepare default + custom plot arguments."""
    plot_args = {
        'vmin': 0.0, 
        'vmax': 0.5, 
        'fig_size': (10, 24), 
        # 'cax_coor': [0.93, 0.2, 0.02, 0.6],
        'cmap': 'jet', 
        'aspect': 'auto',
        'rotation': {'x': 90, 'y': 0},
        'tick_size': {'x': 28, 'y': 28, 'cax': 30},
        'label_size': {'x': 38, 'y': 38, 'cax': 36},
        'title_size': 28, 
        'dpi': 310,
        'labels': {'x': "Residues", 'y': "Residues"},
        # 'xticks': None, 
        # 'yticks': None,
        # 'tick_interval': 2
    }
    if custom_args:
        plot_args.update(custom_args)
    return plot_args


def prepare_plot_args_rg_vs_temp(custom_args: dict) -> dict:
    """Internal helper to prepare default + custom plot arguments."""
    plot_args = {
        'fig_size': (7, 5), 
        'rotation': {'x': 0, 'y': 0},
        'tick_size': {'x': 25, 'y': 25},
        'label_size': {'x': 25, 'y': 25},
        'title_size': 20, 'dpi': 310,
        'labels': {'x': "Temperature (K)", 'y': "<Rg> (nm)"},
        'yticks': np.arange(1.0, 1.5, 0.1), 'xticks': np.arange(300, 500, 50),
        'marker':'o', 'markersize':8, 
        'linestyle':'-', 'linewidth':2, 'color':'blue', 'alpha':0.3,
        'labelpad':10,
        'minor_ticks' : {'x_loc':50/5, 'y_loc':0.1/2, 
                        'length':3, 'width':1, 'color':'k', 'direction':'out'},
        # 'text_params': {'x_start':300, 'y_start':1.16, 'y_step':0.02, 
        #                 'y_substep':0.013, 'x_step':20, 'size':15, 'color': None}, 
        'bbox_inches' : 'tight',
        # 'tick_interval': 2
    }
    if custom_args:
        plot_args.update(custom_args)
    return plot_args

def prepare_plot_args_fe_1d(custom_args: dict) -> dict:
    """Internal helper to prepare default + custom plot arguments."""
    plot_args = {
        'fig_size': (10,8), 
        'rotation': {'x': 0, 'y': 0},
        'tick_size': {'x': 25, 'y': 25},
        'label_size': {'x': 25, 'y': 25},
        'dpi': 310,
        'labels': {'x': 'Radius of Gyration (nm)', 'y': 'Free Energy (kcal/mol)'},
        'xticks': None, 'yticks': None,
        'linestyle':'-', 'linewidth':2, 'alpha':0.3,
        'labelpad':10,
        # 'minor_ticks' : {'x_loc':50/5, 'y_loc':0.1/2, 
        #                 'length':3, 'width':1, 'color':'k', 'direction':'out'},
        'bbox_inches' : 'tight',
        'legend_params' : {'loc': 'upper center', 'fontsize': 12, 'ncol': 5, 'bbox_to_anchor':(0.38, 1.0)}, 
        'grid': True,
    }
    if custom_args:
        plot_args.update(custom_args)
    return plot_args

def prepare_plot_args_agg_rg(custom_args: dict) -> dict:
    """Internal helper to prepare default + custom plot arguments."""
    plot_args = {
        'fig_size': (10,8), 
        'rotation': {'x': 0, 'y': 0},
        'tick_size': {'x': 25, 'y': 25},
        'label_size': {'x': 25, 'y': 25},
        'dpi': 310,
        'labels': {'x': r'Aggregate Time ($\mu$s)', 'y': r'<Rg> (nm)'},
        'xticks': None, 'yticks': None,
        'labelpad':10,
        'bbox_inches' : 'tight',
        'markersize': 10,
        'capsize': 5,
        'capthick': 2,
        'color': 'black',
        'fmt': 'o',
    }
    if custom_args:
        plot_args.update(custom_args)
    return plot_args

def prepare_plot_args_ss(custom_args: dict) -> dict:
    """Internal helper to prepare default + custom plot arguments."""
    plot_args = {
        'fig_size': (7,5),
        'cols': 1,
        'rows': 2,
        'sharex': True, 'sharey': True,
        'nreps': 10, 
        'cmap': 'viridis',
        'linewidth': 2,
        'capsize': 5,
        'rotation': {'x': 90, 'y': 0},
        'tick_size': {'x': 20, 'y': 20},
        'label_size': {'x': 24, 'y': 24},
        'dpi': 310,
        'labels': {'x': 'Residues', 'y1': r'$\alpha$', 'y2': r'$\beta$'},
        'xticks': range(0,20,2), 'yticks': np.arange(0,0.14,0.04),
        'labelpad':15,
        'bbox_inches' : 'tight',
        'border_width': 2,
    }
    if custom_args:
        plot_args.update(custom_args)
    return plot_args

def prepare_plot_args_lig_modes(custom_args: dict) -> dict:
    """Internal helper to prepare default + custom plot arguments."""
    plot_args = {
        'fig_size': (10,8),
        'cols': 2,
        'rows': 2,
        'sharex': True, 'sharey': True,
        'nreps': 10, 
        'cmap': 'viridis',
        'linewidth': 2,
        'alpha': 0.3,
        'rotation': {'x': 90, 'y': 0},
        'tick_size': {'x': 20, 'y': 20},
        'label_size': {'x': 25, 'y': 25},
        'dpi': 310,
        'labels': {'x': 'Residues', 'y': 'Fraction'},
        'xticks': None, 'yticks': None,
        'xtick_interval': 2,
        'labelpad':15,
        'bbox_inches' : 'tight',
        'border_width': 2,
        'title' :  {'fontsize': 20, 'pad': 10},
    }
 
    if custom_args:
        plot_args.update(custom_args)
    return plot_args

############################################################################################################################################################################################################

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
