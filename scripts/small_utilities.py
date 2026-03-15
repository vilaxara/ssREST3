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

def markers_colors():
        
    import matplotlib.pyplot as plt
    import matplotlib

    cm = plt.get_cmap('tab20')
    colors = cm(np.linspace(0, 1, 20))
    # np.array([i for i in matplotlib.markers.MarkerStyle.markers.keys()])[:-4]

    return np.array(colors), np.array([i for i in matplotlib.markers.MarkerStyle.markers.keys() if i not in ['None', 'none', ' ', '', ',']])

def plot_cm_all(input:dict, cbar_label:str, file_name:str=None, save_fig:bool = False, show_fig:bool = False, title_dict:dict = None, offset=None,
                plot_args_in:dict = {'vmin': 0.0, 'vmax': 0.5, 'fig_size': (30,14), 'cax_coor': None, 'cmap': 'jet', 'aspect' : 'auto', 'rotation' : {'x' : 90, 'y' :0},
                                  'tick_size' : {'x' : 18, 'y' : 18, 'cax' :25}, 'label_size' : {'x' : 30, 'y' : 30, 'cax' :30}, 'title_size' :20, 'dpi' : 310, 'labels' : {'x' :"Residues", 'y':"Residues" , 'x_pos' : None, 'y_pos' : None},
                                  'nrows' : 2, 'ncols' : 5, 'xticks' : None, 'yticks' : None, 'tick_interval' : 2, 'sup title':{'title' : None, 'fontsize': None, 'x': None, 'y': None}}):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # [0.93, 0.2, 0.02, 0.6] defult cax

    plot_args:dict = {'vmin': 0.0, 'vmax': 0.5, 'fig_size': (30,14), 'cax_coor': None, 'cmap': 'jet', 'aspect' : 'auto', 'rotation' : {'x' : 90, 'y' :0},
                        'tick_size' : {'x' : 18, 'y' : 18, 'cax' :25}, 'label_size' : {'x' : 30, 'y' : 30, 'cax' :30}, 'title_size' :20, 'dpi' : 310, 'labels' : {'x' :"Residues", 'y':"Residues", 'x_pos' : None, 'y_pos' : None },
                        'nrows' : 2, 'ncols' : 5, 'xticks' : None, 'yticks' : None, 'tick_interval' : 2, 'sup title':{'title' : None, 'fontsize': None, 'x': None, 'y': None}}
    
    if plot_args_in :

        for k in plot_args.keys():

            if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]

    fig, axes = plt.subplots(plot_args['nrows'],plot_args['ncols'], figsize=plot_args['fig_size'], sharex=True, sharey=True, dpi=610, layout='constrained')
    images=[]
    for i in input.keys():
        # ax.set_axis_off()

        p,q = np.unravel_index(i,(plot_args['nrows'], plot_args['ncols']))

        im = axes[p,q].imshow(np.array(input[i]), vmin=plot_args['vmin'], vmax=plot_args['vmax'],cmap=plot_args['cmap'], aspect=plot_args['aspect'])
        im.axes.tick_params(axis='both',which='both',direction='out')
        axes[p,q].invert_yaxis()
        axes[p,q].set_xticks(range(0, len(plot_args['xticks'])+(offset-1),plot_args['tick_interval']), plot_args['xticks'][::plot_args['tick_interval']], rotation=plot_args['rotation']['x'], size=plot_args['tick_size']['x'])
        axes[p,q].set_yticks(range(0, len(plot_args['yticks'])+(offset-1),plot_args['tick_interval']), plot_args['xticks'][::plot_args['tick_interval']], rotation=plot_args['rotation']['y'], size=plot_args['tick_size']['y'])
        axes[p,q].grid(False)
        axes[p,q].set_title(f"{title_dict[i]}",size=plot_args['title_size'], pad=10)


        # if not q : axes[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
        # if p == plot_args['nrows']-1 : axes[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
        images.append(im)

    [fig.delaxes(ax) for ax in axes.flat if not ax.has_data()]
            
    if plot_args['cax_coor'] : cax= fig.add_axes(plot_args['cax_coor'])
    # else : cax = fig.add_axes([axes[plot_args['nrows']-1,plot_args['ncols']-1].get_position().x1+0.02,axes[plot_args['nrows']-1,plot_args['ncols']-1].get_position().y0,0.02,axes[0,0].get_position().y1-axes[plot_args['nrows']-1,plot_args['ncols']-1].get_position().y0])
    # else : cax = fig.add_axes([axes[plot_args['nrows']-1,plot_args['ncols']-1].get_position().x1+0.12, axes[plot_args['nrows']-1,plot_args['ncols']-1].get_position().y0-0.034, 0.02, axes[0,0].get_position().y1-axes[plot_args['nrows']-1,plot_args['ncols']-1].get_position().y0+0.134])
    else : cax = fig.add_axes([axes[plot_args['nrows']-1,plot_args['ncols']-1].get_position().x1+0.12, axes[plot_args['nrows']-1,plot_args['ncols']-1].get_position().y0+0.016, 0.02, axes[0,0].get_position().y1-axes[plot_args['nrows']-1,plot_args['ncols']-1].get_position().y0+0.05])
    
    # else : 
    #     divider = make_axes_locatable(axes[0,0])
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
        
    cbar = fig.colorbar(images[-1],cax=cax)
    cbar.set_label(cbar_label, size=plot_args['label_size']['cax'], labelpad=20)
    cbar.ax.tick_params(labelsize=plot_args['tick_size']['cax'])
    # cbar.set_ticks([])
    # cbar.set_ticklabels([])

    # fig.text(0.5, 0.04, plot_args['labels']['x'], ha="center", fontsize=plot_args['label_size']['x'])
    # fig.text(-0.05, 0.5, plot_args['labels']['y'], va="center", rotation="vertical", fontsize=plot_args['label_size']['y'])

    fig.supxlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], y=plot_args['labels']['x_pos'])
    fig.supylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], x=plot_args['labels']['y_pos'])
    fig.text(plot_args['sup title']['x'], plot_args['sup title']['y'], plot_args['sup title']['title'], ha="center", fontsize=plot_args['sup title']['fontsize'])    
    
    # plt.tight_layout()
    # plt.grid(alpha=0.1)
    if show_fig : plt.show()

    if save_fig : assert file_name ; out_f = f"{file_name}" ; print(f'saving figure {file_name}!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
    else : pass

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

def adjust_min(x):
    idx = (x == 0)
    x[idx] = x[~idx].min()
    return x


def compute_time(data, timestep=80):
    
    import numpy as np
    
    """
    Compute time values (in microseconds) for each frame in a trajectory.

    Parameters
    ----------
    data : list or np.ndarray
        Input data array (e.g., trajectory frames or any list of frames).
    timestep : float, optional
        Simulation time step in picoseconds (default = 80 ps).

    Returns
    -------
    np.ndarray
        Array of time values in microseconds.
    """
    n_frames = len(data)
    return np.arange(n_frames) * timestep / 1e6

import numpy as np

def compute_time_batches(timestep, n_frames, magic_num, check:bool=True):
    """
    Compute batch end-times based on magic_num scaling logic.
    
    Parameters
    ----------
    timestep : float
        Timestep in picoseconds (ps), same as used in compute_time().
    n_frames : int
        Total number of frames in the dataset.
    magic_num : float
        Scaling constant used to compute frame cutoffs.

    Returns
    -------
    time_agg_dict : dict
        Dictionary mapping batch index (i/10) -> time at end of batch (microseconds).
    time_agg : np.ndarray
        Array of batch times (microseconds).
    frame_counts : list
        Unique frame cutoffs used (same as original time_split logic).
    """

    # Precompute full time array (in microseconds)
    time = np.arange(n_frames) * timestep / 1e6

    i_vals = np.arange(10, 510, 10)  # 10,20,...,500
    end_frames = (magic_num * i_vals + 1).astype(int)
    end_frames = np.clip(end_frames, 1, n_frames)

    # Keep only strictly increasing frame indices (matches your break logic)
    keep = np.concatenate(([True], np.diff(end_frames) > 0))
    i_keep = i_vals[keep]
    end_keep = end_frames[keep]

    # Compute times at those frame cutoffs
    time_keep = time[end_keep - 1]

    # Build dictionary {i/10 : time}
    time_agg_dict = {i/10: t for i, t in zip(i_keep, time_keep)}

    if check :
            
        # Print output like your original code
        for i, ef, last in zip(i_keep, end_keep, time_keep):
            # print(f"Index: {i}; # of frames: {ef}, Time: {last:.4f} µs")
            print(f"Index: {i}; # of frames: {ef}, Time: {last} µs")


    return time_agg_dict, np.asarray(time_keep), list(end_keep)


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    import matplotlib
    
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), size=18, **kw)
            texts.append(text)

    return texts


def reposition_first(array):
    if len(array.shape) == 2:  # Check if it's 2D
        # Reposition first row to the last
        return np.vstack((array[1:], array[0]))
    elif len(array.shape) == 3:  # Check if it's 3D
        # Reposition first slice to the last
        return np.concatenate((array[1:], array[:1]), axis=0)

import matplotlib.pyplot as plt
import numpy as np

def plot_colvar(colvar, trj_frames, num_cvs, labels, weights=None, out_dir=None):
    """
    Plot collective variables over time with corresponding histograms and reweighted histograms.

    Parameters:
        colvar (np.ndarray): Array of collective variables with time as the first column.
        trj_frames (dict): Dictionary mapping frame indices to frame times.
        num_cvs (int): Number of collective variables.
        labels (dict): Dictionary of labels for the collective variables.
        weights (np.ndarray, optional): Normalized weights for reweighting histograms. Must match the number of frames.
        out_dir (str, optional): Directory to save the plot. If None, the plot is shown but not saved.
    """
    # Configure subplots
    height_ratios = [1] * (num_cvs - 1)
    fig, axes = plt.subplots(
        num_cvs - 1, 2, figsize=(32, 40),
        gridspec_kw={
            'width_ratios': [4, 1],
            'height_ratios': height_ratios,
            'left': 0.1, 'right': 0.9,
            'bottom': 0.1, 'top': 0.9,
            'wspace': 0.02, 'hspace': 0.2
        },
        sharey=False,
        # sharex=True
    )

    # Check if weights are provided
    if weights is not None:
        if len(weights) != colvar.shape[0]:
            raise ValueError("Weights must match the number of frames in colvar.")
        weights = weights / np.sum(weights)  # Normalize weights

    # Loop over collective variables
    for i in range(num_cvs - 1):
        time = colvar.T[0] / 10**5  # Time in appropriate units
        data = colvar.T[i + 1]  # Data for the i-th collective variable

        # Line plot of collective variable over time
        axes[i, 0].plot(time, data, color='r', label=labels.get(str(i + 1), f'CV {i + 1}'))
        axes[i, 0].set_ylabel(labels.get(str(i + 1), f'CV {i + 1}'), size=35)
        axes[i, 0].set_xlabel(r"Time ($\mu$s)", size=35)
        axes[i, 0].tick_params(labelsize=25)
        xticks=[frame_time[1] / 10**5 for frame_idx, frame_time in trj_frames.items()]

        axes[i, 0].set_xticks(xticks)

        # Add vertical lines for trajectory frames
        for frame_idx, frame_time in trj_frames.items():
            axes[i, 0].axvline(x=frame_time[1] / 10**5, color='black', linewidth=3, linestyle='--')

        # Plot histogram
        axes[i, 1].hist(data, bins=50, orientation='horizontal', density=True, color='black', histtype='step', linewidth=2)

        # Plot reweighted histogram (if weights are provided)
        if weights is not None:
            # # Compute weighted histogram
            # hist, bin_edges = np.histogram(data, bins=50, density=True, weights=weights)
            # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # axes[i, 1].plot(hist, bin_centers, color='green', linewidth=2, label='Reweighted')
            axes[i, 1].hist(data, bins=50, orientation='horizontal', density=True, color='green', histtype='step', linewidth=2, weights=weights)


        axes[i, 1].tick_params(labelsize=25)
        axes[i, 1].set_yticks([])

    # Save or display the plot
    if out_dir:
        plt.savefig(f"{out_dir}/cv_plot.png", dpi=310, bbox_inches='tight')
    else:
        plt.show()

def create_trj_frames(colvar_data, nreps=16):
    """
    Create trajectory frame intervals based on the given data.

    Parameters:
        colvar_data (list): A list or array-like structure containing the data.
        nreps (int): Number of intervals (default is 16).
        index (int): Index of the dataset to process (default is 100).

    Returns:
        dict: A dictionary where keys are strings of interval indices and values are lists of start and end points.
    """
    # Determine the total length and calculate interval size
    total_frames = colvar_data.shape[0] - 1
    interval_size = total_frames / nreps

    # Initialize variables
    a, b = 0, interval_size
    trj_frames = {}

    # Create intervals
    for i in range(nreps):
        trj_frames[str(i)] = [a, b]
        a = b
        b += interval_size

    return trj_frames
