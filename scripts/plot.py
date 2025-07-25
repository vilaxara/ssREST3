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

old_path='/home/jayakrishna/work/scripts/analysis_class'
if old_path in sys.path : sys.path.remove(old_path)

# sys.path.insert(0, '.')
from Block_analysis import *
from small_utilities import *

from importlib import reload
import plot_external_def
reload(plot_external_def)
from plot_external_def import *

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

def compute_agg_distance(
        data: np.ndarray,
        analysis_func: str,
        analysis_kwargs: Optional[dict] = None,
        magic_num: float = 78.1,
        start: int = 10,
        stop: int = 510,
        step: int = 10,
        timestep: float=80.0,
        # series_offset: int = 1,
        # series_count: Optional[int] = None

) ->  Dict[int, np.ndarray] :

    import numpy as np

    if analysis_kwargs is None:
        analysis_kwargs = {}

    n_rows, n_frames = data.shape
    n_dist = n_rows - 1
    time = np.zeros(n_frames)
    for i in range(n_frames):
        time[i]=(i*timestep)/(10**6)
    
    if analysis_func == 'agg_free_energy' :

        result={}

        for dist_no in range(n_dist):

            splits = []
            d=[]

            for i in range(start, stop, step):
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

                d.append(free_energy_1D_blockerror(data[:idx], **analysis_kwargs))

            result[dist_no]=np.ndarray(d)

    elif analysis_func == 'agg_mean' :

        result={}

        for dist_no in range(n_dist):

            splits = []
            d=[]

            for i in range(start, stop, step):
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

                d.append(get_blockerror_pyblock_nanskip(data[:idx]))

            result[dist_no]=np.ndarray(d)

    else : print("ERROR : Select either agg_free_energy or agg_mean flags!") ; exit

    return result
    
#######################################################################################################################################################################################################################

class plots():

    def __init__(self, data_dir:str, out_dir:str, p_seq:list, json_files_dict:dict, num_of_replicas:int, out_files_dict:dict={}, rep_demux_switch:str="on",
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
        self.replica_temps = compute_temperatures(temp_range, nreps=self.nreps)

        print(f"Your temperatures are : \n{np.array([f'{i:6.3f}K' for i in self.replica_temps])}\n")

        self.dpi = dpi

        # if rows_cols : self.rows, self.cols = rows_cols['rows'], rows_cols['cols']
        # elif not rows_cols : self.cols = 4 ; self.rows = self.nreps//self.cols


        if rep_demux_switch == "on" :

            self.s, self.title_names =['rep', 'demux'] , ['Replica', 'Demuxed Replicas']


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
            'colvar' : False
            # 'rg_2d': False,
            # 'pp': False,
            # 'phipsi': False,
        }
        
        self.out_files_dict = {
            'p_cm': False,
            'l_cm': False,
            'rg': False,
            'rg_fe': False,
            'agg_rg': False,
            'ss': False,
            'lig_modes': False,
            'lig_contacts': False,
            'dist': False,
            'dist_fe': False,
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

        in_data={i:in_data[i] for i in in_data.keys() if 'rep' in i} if not demux else \
                {i:in_data[i] for i in in_data.keys() if 'demux' in i}

        out_suffix = "demux" if demux else "rep"
        out_file = f"{self.out_files_dict[cm_type]}_{out_suffix}.{self.out_file_type}"

        title=[f'{round(i)}K' for i in self.replica_temps] if not demux else [f'Demux : {i}' for i in range(self.nreps)]

        plot_args = prepare_plot_args_cm(plot_args_in or {})

        if 'xticks' not in plot_args:
            plot_args['xticks'] = [" "  if i%3 else self.p_seq[i] for i in range(len(self.p_seq)) ]
            # print(f"Using xticks: {self.p_seq}")
        if 'yticks' not in plot_args:
            plot_args['yticks'] = [" "  if i%3 else self.p_seq[i] for i in range(len(self.p_seq)) ]


        plot_cm_all(in_data, 'Contact Probability', out_file, save_fig=save_fig, show_fig=show_fig, 
                     demux=demux, title=title, out_dir=self.out_dir, **plot_args)


    def plot_p_cm(self, save_fig: bool = False, show_fig: bool = False, demux: bool = False, plot_args_in: dict = None):
        self._plot_cm_wrapper('p_cm', save_fig, show_fig, demux, plot_args_in)


    def plot_l_cm(self, save_fig: bool = False, show_fig: bool = False, demux: bool = False, plot_args_in: dict = None):
        self._plot_cm_wrapper('l_cm', save_fig, show_fig, demux, plot_args_in)        




    def plot_rg_temperature(self, show_fig: bool = True, save_fig: bool = False, dpi: int = 310, plot_args_in: dict = {}):
                
        plot_args = prepare_plot_args_rg_vs_temp(plot_args_in)

        if not self.in_files_dict['rg']:
            print("Rg data not loaded. Please load the data first.")
            return

        in_file = f"{self.data_dir}/{self.in_files_dict['rg']}"
        rg_data = load_json(in_file)

        plot_rg_vs_temperature(rg_data, self.replica_temps, self.nreps, show=show_fig, 
                               save=save_fig, out_dir=self.out_dir, out_file=self.out_files_dict['rg'], out_file_type=self.out_file_type,
                               **plot_args)
        
    def print_time_splits(self, timestep: float = 80.0, magic_num: float = 125.0, max_val: int = 510, step: int = 10):
        if not self.in_files_dict['rg']:
            print("Rg data not loaded. Please load the data first.")
            return

        in_file = f"{self.data_dir}/{self.in_files_dict['rg']}"
        rg_data = load_json(in_file)

        # Use the first replica's data to compute time splits
        first_rep_key = next((key for key in rg_data.keys() if self.s[0] in key), None)
        if not first_rep_key:
            print("No replica data found in Rg data.")
            return

        time_splits = compute_time_splits(rg_data[first_rep_key], magic_num=magic_num, max_val=max_val, step=step, timestep=timestep)
        # print(f"Computed time splits (in frames): {time_splits}")


    def free_energy_1d(self, splits: list[int] = None, magic_num: float = 125.0,
                     T: float = 300.0, bins: int = 50, blocks: int = 5, weights=None,
                     cmap_name: str = 'tab20c', save_fig: bool = False, show_fig: bool = True,
                     plot_args_in: dict = {}):

        plot_args = prepare_plot_args_fe_1d(plot_args_in)


        if not self.in_files_dict['rg']:
            print("Rg data not loaded. Please load the data first.")

        in_file = f"{self.data_dir}/{self.in_files_dict['rg']}"
        rg_data = load_json(in_file)

        # Use the first replica's data to compute time splits
        first_rep_key = next((key for key in rg_data.keys() if self.s[0] in key), None)
        if not first_rep_key:
            print("No replica data found in Rg data.")
    
        rg_series = rg_data[first_rep_key]
            
        if not isinstance(rg_series, np.ndarray):
            rg_series=np.array(rg_series, dtype=float)
        
        plot_free_energy_1d(
            rg_series=rg_series,
            splits=splits,
            magic_num=magic_num,
            T=T,
            bins=bins,
            blocks=blocks,
            weights=weights,
            cmap_name=cmap_name,
            out_file=self.out_files_dict['rg_fe'],
            save_fig=save_fig,
            show_fig=show_fig,
            out_dir=self.out_dir,
            out_file_type=self.out_file_type
            **plot_args
        )

    def aggregate_rg(self, save_fig: bool = False, show_fig: bool = True, plot_args_in: dict = {}):

        plot_args = prepare_plot_args_agg_rg(plot_args_in)

        if not self.in_files_dict['rg']:
            print("Rg data not loaded. Please load the data first.")
            return

        in_file = f"{self.data_dir}/{self.in_files_dict['rg']}"
        rg_data = load_json(in_file)

        # Use the first replica's data to compute time splits
        first_rep_key = next((key for key in rg_data.keys() if self.s[0] in key), None)
        if not first_rep_key:
            print("No replica data found in Rg data.")
    
        rg_series = rg_data[first_rep_key]

        plot_aggregate_rg(rg_series, show=show_fig,
                            save=save_fig, out_dir=self.out_dir, out_file=self.out_files_dict['agg_rg'],
                            out_file_type=self.out_file_type, **plot_args)

    def helix_sheet(self, save_fig: bool = False, show_fig: bool = True, plot_args_in: dict = {},
                            demux: bool = False):

        plot_args = prepare_plot_args_ss(plot_args_in)

        if not self.in_files_dict['helix'] or not self.in_files_dict['sheet']:
            print("Helix or Sheet data not loaded. Please load the data first.")
            return

        helix_data = load_json(f"{self.data_dir}/{self.in_files_dict['helix']}")
        sheet_data = load_json(f"{self.data_dir}/{self.in_files_dict['sheet']}")

        if not demux:
            in_dict_h={int(i.split(':')[1]):helix_data[i] for i in helix_data.keys() if self.s[0] in i}
            in_dict_s={int(i.split(':')[1]):sheet_data[i] for i in sheet_data.keys() if self.s[0] in i}
        else:
            in_dict_h={int(i.split(':')[1]):helix_data[i] for i in helix_data.keys() if self.s[1] in i}
            in_dict_s={int(i.split(':')[1]):sheet_data[i] for i in sheet_data.keys() if self.s[1] in i}

        # print(self.p_seq)

        fig, axes = plt.subplots(plot_args['rows'], plot_args['cols'], figsize=plot_args['fig_size'],
                                 sharex=plot_args['sharex'], sharey=plot_args['sharey'])


        for i in in_dict_h.keys():

            axes[0].errorbar(self.p_seq, np.array(in_dict_h[i]).T[0], yerr=np.array(in_dict_h[i]).T[1],
                             color=plt.cm.viridis(i/plot_args['nreps']), lw=plot_args['linewidth'], capsize=plot_args['capsize'],)
            axes[1].errorbar(self.p_seq, np.array(in_dict_s[i]).T[0], yerr=np.array(in_dict_s[i]).T[1],
                             color=plt.cm.viridis(i/plot_args['nreps']), lw=plot_args['linewidth'], capsize=plot_args['capsize'],)
            
        axes[0].tick_params(axis='x', labelsize=plot_args['tick_size']['x']) 
        axes[0].tick_params(axis='y', labelsize=plot_args['tick_size']['y'])

        axes[0].set_ylabel(plot_args['labels']['y1'], fontsize=plot_args['label_size']['y'])

        axes[1].tick_params(axis='x', labelsize=plot_args['tick_size']['x'])
        axes[1].tick_params(axis='y', labelsize=plot_args['tick_size']['y'])
        axes[1].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'])
        axes[1].set_ylabel(plot_args['labels']['y2'], fontsize=plot_args['label_size']['y'])
        axes[1].set_xticks(plot_args['xticks'], self.p_seq[::2], rotation=plot_args['rotation']['x'])
        axes[1].set_yticks(plot_args['yticks'])

        for ax in axes.flat:
            for spines in ax.spines.values():
                spines.set_linewidth(plot_args['border_width'])


        plt.tight_layout()

        if save_fig:
            out_dir = self.out_dir
            out_file_path = os.path.join(out_dir, f"{self.out_files_dict['ss']}.{self.out_file_type}")
            plt.savefig(out_file_path, dpi=plot_args['dpi'], bbox_inches=plot_args['bbox_inches'])
            print(f"Saved figure to {out_file_path}")

        if show_fig:
            plt.show()


# from typing import Dict, Sequence, Tuple, Any

    def ligand_binding_profile( self,
        demux: bool = False,
        plot_args_in: dict = {},
        annotation: bool = True,
        annot_x_start: int = 123,
        annot_y_start: int = 0.35,
        annot_step_x: int = 3.4,
        annot_step_y_main: float = 0.04,
        annot_step_y_sub: float = 0.025,
        annot_size: int = 15,
        show_fig: bool = True,
        save_fig: bool = False,
        ):
            
        import matplotlib.pyplot as plt
        import numpy as np

        temperature = self.replica_temps
        nrep =  self.nreps   
        
        plot_args = prepare_plot_args_lig_modes(plot_args_in)

        assert self.in_files_dict['lig_modes'], "Load the lig_modes data before plotting!"
        
        metrics = {
        'Aromatic': 'aro',
        'Hydrophobic': 'hyphob',
        'Hbond': 'hbond',
        'Charge': 'charge'
        }

        for key, value in metrics.items():
            if value in self.in_files_dict['lig_modes']:
                metrics[key] = self.in_files_dict['lig_modes'][value]

        for mode in  metrics.keys():
            if not metrics[mode]:
                print(f"Data for {mode} not found in input files. Skipping this mode.")
                continue
            in_file = f"{self.data_dir}/{metrics[mode]}"
            data = load_json(in_file)

            if not demux:
                metrics[mode] = {f'{self.s[0]}:{i}': data[f'{self.s[0]}:{i}'] for i in range(nrep)}
            else:
                metrics[mode] = {f'{self.s[1]}:{i}': data[f'{self.s[1]}:{i}'] for i in range(nrep)}

        titles = list(metrics.keys())
        n_metrics = len(titles)
        if not plot_args['cols'] : ncols = 2
        else : ncols = plot_args['cols']
        if not plot_args['rows'] : nrows = (n_metrics + 1) // 2
        else :
            nrows = plot_args['rows']
            if n_metrics < ncols * nrows:
                n_metrics = ncols * nrows

        fig, axes = plt.subplots(nrows, ncols, figsize=plot_args['fig_size'], sharex=plot_args['sharex'], sharey=plot_args['sharey'], layout='constrained')

        axes = axes.flatten()
        cmap = plt.cm.get_cmap(plot_args['cmap'])

        if demux : tag= self.s[1]
        else : tag = self.s[0]
        
        # Plot each metric
        for ax, title in zip(axes, titles):
            metric_data = metrics[title]
            ax.tick_params(labelsize=plot_args['tick_size']['x'], axis='both')
            for i in range(nrep):
                arr = np.array(metric_data[f'{tag}:{i}'])
                x_vals = arr.T[0]
                y_vals = arr.T[1]
                y_err = arr.T[2]
                color = cmap(i / nrep)
                ax.plot(x_vals, y_vals, color=color, lw=2)
                ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, color=color, alpha=0.3)
            ax.set_title(title, size=plot_args['title']['fontsize'], pad=plot_args['title']['pad'])

            # X-ticks and Y-ticks
            x0 = np.array(metric_data[f'{tag}:0']).T[0]
            ax.set_xticks(x0[::plot_args['xtick_interval']], self.p_seq[::plot_args['xtick_interval']], rotation=plot_args['rotation']['x'])
            y_positions = plot_args['yticks'] if plot_args['yticks'] is not None else np.arange(0, 0.5, 0.1)
            ax.set_yticks(y_positions, [f"{v:2.1f}" for v in y_positions])

        # Supplied axis labels
        fig.supxlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'])
        fig.supylabel(plot_args['labels']['y'], size= plot_args['label_size']['y'])

        # Annotate temperatures in bottom-left plot
        if annotation:
            ax_ann = axes[ncols]  # bottom-left
            y = annot_y_start 
            x = annot_x_start 
            for idx, temp in enumerate(temperature):
                ax_ann.text(x, y, f"{round(temp)}K", size=annot_size, color=cmap(idx / nrep))
                y -= annot_step_y_main 
                if idx % 2:
                    x += annot_step_x 
                    y = annot_y_start
                else:
                    y -= annot_step_y_sub 

        # Thicken spines
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_linewidth(plot_args['border_width'])

        plt.tight_layout()

        if show_fig : plt.show()
        
        if save_fig:
            out_file_path = os.path.join(self.out_dir, f"{self.out_files_dict['lig_modes']}.{self.out_file_type}")
            plt.savefig(out_file_path, dpi=plot_args['dpi'], bbox_inches=plot_args['bbox_inches'])
            print(f"Saved figure to {out_file_path}")


    def ligand_contacts(
        self,
        demux: bool = False,
        save_fig: bool = False,
        show_fig: bool = False,
        plot_args_in: dict = None
    ):
        """
        Plot per-replica or averaged ligand–protein contact probabilities.

        Parameters:
        - demux: if True, plot demultiplexed data; otherwise plot full-replica data
        - save_fig: whether to save the figure to disk
        - show_fig: whether to display the figure interactively
        - plot_args_in: dict of plot customizations to override defaults
        """
        import matplotlib.pyplot as plt
        import numpy as np
        # 1. Prepare defaults and override
        plot_args = {
            'fig_size':       (12, 8),
            'dpi':            310,
            'rotation':       {'x': 0,  'y': 0},
            'tick_size':      {'x': 20, 'y': 20},
            'label_size':     {'x': 25, 'y': 25},
            'labels':         {'x': 'Residue', 'y': 'Contact probability'},
            'title':          {'label': '', 'size': 28},
            'legend_args':    {'fontsize': 14, 'loc': 'upper right', 'ncol': 2},
            'cmap':           'viridis',
            'offset':         getattr(self, 'offset', 0),
            'seq_length':     len(self.p_seq),
            'top_labels':     self.p_seq_a,
            'linewidth':      2.5,
            'bbox_inches':    'tight',
            'alpha':          0.2,
            'labelpad':       15,
        }

        if plot_args_in:
            plot_args.update(plot_args_in)

        # 2. Select key prefix and title
        prefix = self.s[1] if demux else self.s[0]
        title  = self.title_names[1] if demux else self.title_names[0]
        plot_args['title']['label'] = title

        # 3. Load data
        assert self.in_files_dict['lig_contacts'], "Load ligand–protein contacts data first!"
        in_file = f"{self.data_dir}/{self.in_files_dict['lig_contacts']}"
        data = load_json(in_file)

        # 4. Prepare figure
        fig, ax = plt.subplots(figsize=plot_args['fig_size'])

        # 5. Plot each replica
        cmap = plt.cm.get_cmap(plot_args['cmap'])

        for i in range(self.nreps):
            arr = np.array(data[f"{prefix}:{i}"])
            x, y, err = arr.T
            color = cmap(i / max(self.nreps - 1, 1))

            label = f"{round(self.replica_temps[i])}K" if not demux else f"Demux : {i}"
            ax.plot(x, y, linewidth=plot_args['linewidth'], color=color, label=label)
            ax.fill_between(x, y - err, y + err, color=color, alpha=plot_args['alpha'])

        # 6. Axis styling
        ax.tick_params(labelsize=plot_args['tick_size']['x'])
        plt.setp(ax.get_xticklabels(), rotation=plot_args['rotation']['x'])
        ax.set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=plot_args['labelpad'])
        ax.set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=plot_args['labelpad'])
        ax.set_title(plot_args['title']['label'], size=plot_args['title']['size'], pad=plot_args['labelpad'])
        ax.set_ylim(0, 1.0)
        ax.set_xlim(
            0 + plot_args['offset'] - 1,
            plot_args['seq_length'] + plot_args['offset']
        )
        # X-ticks every 2 residues
        xticks = range(0 + plot_args['offset'], plot_args['seq_length'] + plot_args['offset'], 2)
        ax.set_xticks(xticks, xticks)

        # 7. Legend, grid, minor ticks
        ax.legend(**plot_args['legend_args'])
        ax.grid(alpha=0.4)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.tick_params(axis='x', which='minor', length=3, width=1, direction='out')

        # 8. Top twin axis with amino-acid labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.tick_params(labelsize=plot_args['tick_size']['x'])
        ax2.set_xticks(range(0 + plot_args['offset'], plot_args['seq_length'] + plot_args['offset']))
        ax2.set_xticklabels(plot_args['top_labels'], rotation=plot_args['rotation']['x'], fontsize=plot_args['tick_size']['x'])
        ax2.spines['top'].set_position(('outward', 0))
        ax2.spines['bottom'].set_visible(False)
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax2.tick_params(axis='x', which='minor', length=3, width=1, direction='out')

        plt.tight_layout()

        # 9. Show or save
        if show_fig:
            plt.show()
        if save_fig:
            assert self.out_files_dict['lig_contacts'], "Provide file_name to save the figure"
            out_path = f"{self.out_dir}/{self.out_files_dict['lig_contacts']}.{self.out_file_type}"
            print(f"Saving figure to {out_path}")
            plt.savefig(out_path, dpi=plot_args['dpi'], bbox_inches=plot_args['bbox_inches'])

    def bf_vs_replica(
        self,
        demux: bool = False,
        save_fig: bool = False,
        show_fig: bool = False,
        plot_args_in: dict = None
    ):
        """
        Parameters:
        - demux: if True, plot demultiplexed data; otherwise plot full-replica data
        - save_fig: whether to save the figure to disk
        - show_fig: whether to display the figure interactively
        - plot_args_in: dict of plot customizations to override defaults
        """
        import matplotlib.pyplot as plt
        import numpy as np
        # 1. Prepare defaults and override
        plot_args = {
            'fig_size':       (12, 8),
            'dpi':            310,
            'rotation':       {'x': 0,  'y': 0},
            'tick_size':      {'x': 20, 'y': 20},
            'label_size':     {'x': 25, 'y': 25},
            'labels':         {'x': 'Replica', 'y': 'Bound Fraction'},
            'title':          {'label': '', 'size': 28},
            'legend_args':    {'fontsize': 14, 'loc': 'upper right', 'ncol': 2},
            'color':          'k',
            'xticks':         None,
            'yticks':         None,
            'linewidth':      2.5,
            'bbox_inches':    'tight',
            'alpha':          0.2,
            'labelpad':       15,
            'marker':         'o',
            'markersize':      12,
        }

        if plot_args_in:
            plot_args.update(plot_args_in)

        # 2. Select key prefix and title
        prefix = self.s[1] if demux else self.s[0]

        # 3. Load data
        assert self.in_files_dict['bf'], "Load bound fraction data first!"
        in_file = f"{self.data_dir}/{self.in_files_dict['bf']}"
        data = load_json(in_file)

        # 4. Prepare figure
        fig, ax = plt.subplots(figsize=plot_args['fig_size'])

        ax.plot(range(self.nreps), np.array([data[f'{prefix}:{i}'] for i in range(self.nreps)]).T[0], lw=plot_args['linewidth'], color='k',
                marker=plot_args['marker'], markersize=plot_args['markersize'])
        ax.fill_between(range(self.nreps), np.array([data[f'{prefix}:{i}'] for i in range(self.nreps)]).T[0]-np.array([data[f'{prefix}:{i}'] for i in range(self.nreps)]).T[1], 
                        np.array([data[f'{prefix}:{i}'] for i in range(self.nreps)]).T[0]+np.array([data[f'{prefix}:{i}'] for i in range(self.nreps)]).T[1],
                        alpha=plot_args['alpha'], color=plot_args['color'])

        ax.tick_params(labelsize=plot_args['tick_size']['x'], axis='x')
        ax.tick_params(labelsize=plot_args['tick_size']['y'], axis='y')

        xticks= range(self.nreps) if not plot_args['xticks'] else plot_args['xticks']
        yticks = np.arange(0.1, 0.5, 0.05) if not plot_args['yticks'] else plot_args['yticks']

        ax.set_xticks(xticks, xticks, rotation=plot_args['rotation']['x'])
        ax.set_yticks(yticks, [f"{i:2.2f}" for i in yticks])

        ax.set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=plot_args['labelpad'])
        ax.set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['x'], labelpad=plot_args['labelpad'])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        if show_fig:
            plt.show()

        if save_fig:
            assert self.out_files_dict['bf_temp'], "Provide file_name to save the figure"
            out_path = f"{self.out_dir}/{self.out_files_dict['bf_temp']}.{self.out_file_type}"
            print(f"Saving figure to {out_path}")
            plt.savefig(out_path, dpi=plot_args['dpi'], bbox_inches=plot_args['bbox_inches'])


    def aggregate_distance(
            self,
            analysis_func: str,
            save_fig: bool = False,
            show_fig: bool = False,
            analysis_kwargs: Optional[dict] = None,
            magic_num: float = 78.1,
            start: int = 10,
            stop: int = 510,
            step: int = 10,
            timestep: float=80.0,
            plot_args_in: dict = None,
    ):
        
        import matplotlib.pyplot as plt
        import numpy as np

        plot_args = {
            'fig_size':       (20,12),
            'dpi':            310,
            'nrows':          None,
            'ncols':          None,
            'rotation':       {'x': 0,  'y': 0},
            'tick_size':      {'x': 20, 'y': 20},
            'label_size':     {'x': 25, 'y': 25},
            'labels':         {'x': 'Distance', 'y': 'Free energy (kcal/mol)'},
            'title':          {'label': '', 'size': 28},
            'legend_args':    {'fontsize': 14, 'loc': 'upper right', 'ncol': 2},
            'cmap':           'tab20',
            'xticks':         None,
            'yticks':         None,
            'linewidth':      3,
            'bbox_inches':    'tight',
            'alpha':          0.2,
            'labelpad':       15,
            'tick_width':     1.75,
        }

        if plot_args_in:
            plot_args.update(plot_args_in)

        assert self.in_files_dict['colvar'], "Load colvar file containing distance data!"
        in_file = f"{self.in_files_dict['colvar']}"
        data = np.loadtxt(in_file, comments=['#', '@']).T

        data_rows, n_frames = data.shape

        if analysis_func is 'agg_free_energy' :
    
            dist_agg_fe = compute_agg_distance(data=data,
                                            analysis_func=analysis_func,
                                            analysis_kwargs=analysis_kwargs,
                                            magic_num=magic_num,
                                            start=start,
                                            stop=stop,
                                            step=step,
                                            timestep=timestep)
            
            fig, ax = plt.subplots(nrows=plot_args['nrows'] or (data_rows-1)/3, 
                                   ncols=plot_args['ncols'] or 3,figsize=plot_args['fig_size'], sharex=True, sharey=True)

            ax = ax.flatten()
            cmap=plt.cm.get_cmap(plot_args['cmap'])

            for d, axes in enumerate(ax):

                for i in range(100, 1100, 100):

                    dG1, bin_centers, ferr = dist_agg_fe[d][int(i/100)-1][0], dist_agg_fe[d][int(i/100)-1][1], dist_agg_fe[d][int(i/100)-1][2]

                    sns.lineplot(x=bin_centers, y=dG1, color=cmap(int(i/100)), linewidth=plot_args['linewidth'], ax=axes)
                    axes.fill_between(bin_centers, dG1-ferr, dG1+ferr, color=cmap(int(i/100)), alpha=plot_args['alpha'])

                    axes.tick_params(labelsize=plot_args['label_size']['x'], width=plot_args['tick_width'])
                    plt.rcParams["axes.linewidth"] = plot_args['linewidth']

                    axes.set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=plot_args['labelpad'])
                    axes.set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=plot_args['labelpad'])
                    
                    axes.set_title(plot_args['title']['label'], size=plot_args['title']['size'], pad=plot_args['labelpad'])

            plt.tight_layout

            if show_fig : plt.show()

            if save_fig : 
                assert self.out_files_dict['dist_fe'], "Provide file_name to save the figure"
                out_path = f"{self.out_dir}/{self.out_files_dict['dist_fe']}.{self.out_file_type}"
                print(f"Saving figure to {out_path}")
                plt.savefig(out_path, dpi=plot_args['dpi'], bbox_inches=plot_args['bbox_inches'])

        elif analysis_func is 'agg_mean' :

            marker=['o','v','<', 's','^','>']

            fig, ax = plt.subplots(1,1,figsize=plot_args['fig_size'])

            for d in range