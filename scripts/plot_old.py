import os
import sys
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib import colors

import math
from numpy import log2, zeros, mean, var, sum, loadtxt, arange, array, cumsum, dot, transpose, diagonal, floor
from numpy.linalg import inv, lstsq

import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def subplot_imshow(data, xedges, yedges, ax=None):
    
    from numpy import ndarray
    
    if type(data) is ndarray:

        if ax is None:
            ax = plt.gca()
            
        im = ax.imshow(data, interpolation='gaussian', extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]],
                        cmap='jet', aspect='auto')
        #cb = plt.colorbar(ticks=cbar_ticks, format=('% .1f'),aspect=10)
        
        return im
    
    else : pass

def load_json(file:str, flag:str='r'):

    return json.load(open(file,flag))

class plots():

    def __init__(self, data_dir:str, out_dir:str, p_seq:list, json_files_dict:dict, num_of_replicas:int, rows_cols:dict={}, out_files_dict:dict={}, rep_demux_switch:str="on",
                 temp_range:list=[300,450], out_file_type:str='png', apo:bool=False, offset:int=0 ):

        self.aa_dict = {'ALA' : ["A", "Alanine"], 'ARG' : ["R", "Arginine"], 'ASN' : ["N", "Asparagine"], 'ASP' : ["D", "Aspartic-acid"], 'CYS' : ["C", "Cysteine"],
                        'GLU' : ["E", "Glutamic-acid"], 'GLN' : ["Q", "Glutamine"], 'GLY' : ["G", "Glycine"], 'HIS' : ["H", "Histidine"], 'ILE' : ["I", "Isoleucine"],
                        'LEU' : ["L", "Leucine"], 'LYS' : ["K", "Lysine"], 'MET' : ["M", "Methionine"], 'PHE' : ["F", "Phenylalanine"], 'PRO' : ["P", "Proline"], 
                        'SER' : ["S", "Serine"], 'THR' : ["T", "Threonine"], 'TRP' : ["W", "Truptophan"], 'TYR' : ["Y", "Tyrosine"], 'VAL' : ["V", "Valine"]}
        
        self.apo=apo
        self.offset=offset
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.out_file_type = out_file_type
        self.nreps = num_of_replicas
        self.replica_temps = self.compute_temperatures(temp_range)

        print(f"Your temperatures are : \n{np.array([f'{i:6.3f}' for i in self.replica_temps])}\n")

        if rows_cols : self.rows, self.cols = rows_cols['rows'], rows_cols['cols']
        elif not rows_cols : self.cols = 4 ; self.rows = self.nreps//self.cols

        if rep_demux_switch == "on" :

            self.s, self.title_names =['rep', 'demux'] , ['Replica', 'Demuxed replica']

        else : pass

        self.p_seq, self.p_seq_a = self.make_seq(p_seq)
        self.in_files_dict = {'p_cm' : False, 'l_cm' : False, 'kd' : False, 'ss' : False, 'rg' : False, 'kd_time' : False,
                              'lig_contacts' :False, 'lig_modes' : False, 'rg_2d' : False, 'sa' : False, 'pp' : False, 'phipsi' : False, 
                              'ba': False}
        self.out_files_dict = {'p_cm' : False, 'l_cm' : False, 'kd' : False, 'ss' : False, 'rg' : False, 'rg_hist' : False, 'kd_time' : False,
                               'lig_contacts' :False, 'lig_modes' : False, 'rg_2d' : False, 'sa' : False, 'pp' : False, 'phipsi' : False,
                               'ba' : False, 'rg_hist_2' : False}

        if json_files_dict :
            try : 
                        
                for i in json_files_dict.keys() :

                    if i in self.in_files_dict.keys() : self.in_files_dict[i] = json_files_dict[i]
                    if i in self.out_files_dict.keys() : self.out_files_dict[i] = out_files_dict[i]

            except : print(f"Should pass a dictnory of json files with filename with keys {self.in_files_dict.keys()} !\n")

        # if out_files_dict :
        #     try : 
                        
        #         for i in out_files_dict.keys() :

        #             if i in self.out_files_dict.keys() : self.out_files_dict[i] = out_files_dict[i]
            
        #     except : print(f"Should pass a dictnory of json files with filename with keys {self.in_files_dict} !\n")
            
        
        # self.main_plots(save_fig=True, show_fig=True)
        
        pass


    def compute_temperatures(self, temp_range:list):
        from numpy import log, exp
        tlow, thigh = temp_range
        temps = []
        for i in range(self.nreps):
            temps.append(tlow*exp((i)*log(thigh/tlow)/(self.nreps-1)))

        return np.array(temps)
    
    def make_seq(self, inp_seq:list) :

        def split_temp(inp:list):
            import re
            import numpy as np

            exp=r"([a-z]+)([0-9]+)"
            out=[]

            for i in range(len(inp)):
                match = re.match(exp, inp[i], re.I)

                if match:
                    items = match.groups()

                out.append(list(items))

            return np.array(out)
        
        a=split_temp(inp_seq)

        b=[]
        c=[]

        for i in range(0,len(a)):

            for k in self.aa_dict.keys():

                if k in a[i]:
                    
                    a[i][0] = self.aa_dict[k][0]

            b.append('_'.join(a[i]))
            c.append(a[i][0])

        return b,c

            
    def _plot_cm_all(self, input:dict, cbar_label:str, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False,
                    plot_args:dict = {'vmin': 0.0, 'vmax': 0.5, 'fig_size': (30,14), 'cax_coor': [0.93, 0.2, 0.02, 0.6], 'cmap': 'jet', 'aspect' : 'auto', 'rotation' : {'x' : 90, 'y' :0},
                                      'tick_size' : {'x' : 18, 'y' : 18, 'cax' :25}, 'label_size' : {'x' : 30, 'y' : 30, 'cax' :30}, 'title_size' :20, 'dpi' : 310, 'labels' : {'x' :"Residues", 'y':"Residues" }}):
        import matplotlib.pyplot as plt
        # [0.93, 0.2, 0.02, 0.6] defult cax

        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1]

        fig, axes = plt.subplots(self.rows,self.cols, figsize=plot_args['fig_size'], sharex=True, sharey=True)
        if plot_args['cax_coor'] : cax= fig.add_axes(plot_args['cax_coor'])
        else : cax = fig.add_axes([axes[self.rows-1,self.cols-1].get_position().x1+0.02,axes[1,2].get_position().y0,0.02,axes[0,0].get_position().y1-axes[self.rows-1,self.cols-1].get_position().y0])

        images=[]
        for i in range(self.nreps):
            # ax.set_axis_off()

            contact_map = np.array(input[f"{s}:{i}"])
            p,q = np.unravel_index(i,(self.rows, self.cols))

            im = axes[p,q].imshow(contact_map, vmin=plot_args['vmin'], vmax=plot_args['vmax'],cmap=plot_args['cmap'], aspect=plot_args['aspect'])
            axes[p,q].invert_yaxis()
            axes[p,q].set_xticks(range(0, len(self.p_seq),2), self.p_seq[::2], rotation=plot_args['rotation']['x'], size=plot_args['tick_size']['x'])
            axes[p,q].set_yticks(range(0, len(self.p_seq),2), self.p_seq[::2], rotation=plot_args['rotation']['y'], size=plot_args['tick_size']['y'])
            axes[p,q].grid(False)
            if not demux : axes[p,q].set_title(f"{title} : {np.round(self.replica_temps[i],3)}K",size=plot_args['title_size'])
            elif demux : axes[p,q].set_title(f"{title} : {i}",size=plot_args['title_size'])

            if not q : axes[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
            if p == self.rows-1 : axes[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
            images.append(im)

        cbar = fig.colorbar(images[-1],cax=cax)
        cbar.set_label(cbar_label, size=plot_args['label_size']['cax'])
        cbar.ax.tick_params(labelsize=plot_args['tick_size']['cax'])
        # cbar.set_ticks([])
        # cbar.set_ticklabels([])

        # fig.text(0.5, 0.005, "Residues", ha="center", fontsize=30)
        # fig.text(0.008, 0.5, "Residues", va="center", rotation="vertical", fontsize=30)

        # plt.tight_layout()
        # plt.grid(alpha=0.1)
        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_p_cm(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, 
                  plot_args_in:dict = {}) : 

        plot_args = {'vmin': 0.0, 'vmax': 0.5, 'fig_size': (30,14), 'cax_coor': [0.93, 0.2, 0.02, 0.6], 'cmap': 'jet', 'aspect' : 'auto', 'rotation' : {'x' : 90, 'y' :0},'tick_size' : {'x' : 18, 'y' : 18, 'cax' :25}, 
                     'label_size' : {'x' : 30, 'y' : 30, 'cax' :30}, 'title_size' :20, 'dpi' : 310, 'labels' : {'x' :"Residues", 'y':"Residues" }}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]

        assert self.in_files_dict['p_cm'] , f"Load the protein-protein contact data for the plot!"

        if demux : out_file = f"{self.out_files_dict['p_cm']}_demux.{self.out_file_type}"
        elif not demux : out_file = f"{self.out_files_dict['p_cm']}_rep.{self.out_file_type}"

        in_data = load_json(f"{self.data_dir}/{self.in_files_dict['p_cm']}")
        self._plot_cm_all(in_data, 'Contact Probability', out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args)

    def plot_l_cm(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, 
                  plot_args_in:dict = {}) : 

        plot_args = {'vmin': 0.0, 'vmax': 0.5, 'fig_size': (30,14), 'cax_coor': [0.93, 0.2, 0.02, 0.6], 'cmap': 'jet', 'aspect' : 'auto', 'rotation' : {'x' : 90, 'y' :0},'tick_size' : {'x' : 18, 'y' : 18, 'cax' :25}, 
                     'label_size' : {'x' : 30, 'y' : 30, 'cax' :30}, 'title_size' :20, 'dpi' : 310, 'labels' : {'x' :"Residues", 'y':"Residues" }}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]


        assert self.in_files_dict['l_cm'] , f"Load the ligand-protein contact data for the plot!"

        if demux : out_file = f"{self.out_files_dict['p_cm']}_demux.{self.out_file_type}"
        elif not demux : out_file = f"{self.out_files_dict['p_cm']}_rep.{self.out_file_type}"

        in_data = load_json(f"{self.data_dir}/{self.in_files_dict['l_cm']}")
        self._plot_cm_all(in_data, 'Contact Probability', out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args)

    def _plot_kd_all(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False,
                     plot_args:dict = {'fig_size': (10,8), 'rotation' : {'x' : 0, 'y' :0},'tick_size' : {'x' : 20, 'y' : 20},
                                       'label_size' : {'x' : 24, 'y' : 24}, 'title_size' :20, 'dpi' : 310, 'labels' : {'x' :"Replica Number (Ligand 47)", 'y':r'$K_{D}$(mM)' }}) :

        kd=[]
        a=0

        if not demux :
                
            for keys, value in input.items():
                if 'rep' in keys :
                    kd.append(value)
                    a=a+1

            title=self.title_names[0]
            
        elif demux :

            for keys, value in input.items():
                if 'demux' in keys :
                    kd.append(value)
                    a=a+1
            
            title=self.title_names[1]

        kd=np.array(kd).T
        # print(kd)

        fig=plt.figure(figsize=plot_args['fig_size'])

        #default_x_ticks=range(nrep)
        plt.tick_params(labelsize=plot_args['tick_size']['x'])
        plt.xticks(range(0,self.nreps), rotation=plot_args['rotation']['x'])
        #ax1.set_xticklabels(kd41[:,0])
        plt.xlabel(plot_args['labels']['x'], fontsize=plot_args['label_size']['x'])
        plt.plot(range(self.nreps),kd[0],linestyle='--', marker='o', color='r', lw=2, ms=7)
        plt.fill_between(range(self.nreps), kd[0]-kd[1],
                        kd[0]+kd[1], color='r', alpha=0.2)

        # plt.yticks(np.arange(5.0,15.0,0.75),size=20)
        plt.grid(alpha=0.4)
        plt.ylabel(plot_args['labels']['y'],size=plot_args['label_size']['y'])
        plt.title(title,size=plot_args['title_size'])
        plt.tight_layout()

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_kd(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, 
                  plot_args_in:dict = {}) : 

        plot_args = {'fig_size': (10,8), 'rotation' : {'x' : 0, 'y' :0},'tick_size' : {'x' : 20, 'y' : 20},
                     'label_size' : {'x' : 24, 'y' : 24}, 'title_size' :20, 'dpi' : 310, 'labels' : {'x' :"Replica Number (Ligand 47)", 'y':r'$K_{D}$(mM)'}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]


        assert self.in_files_dict['kd'] , f"Load the kd data for the plot!"

        if demux : out_file = f"{self.out_files_dict['kd']}_demux.{self.out_file_type}"
        elif not demux : out_file = f"{self.out_files_dict['kd']}_rep.{self.out_file_type}"

        in_data = load_json(f"{self.data_dir}/{self.in_files_dict['kd']}")
        self._plot_kd_all(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args)


    def _plot_ss_all(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False, kd_text:bool =True, kd_in:dict={},
                     kd_text_args:dict = {'size' :28, 'loc' : {'x' : 120.3, 'y': 0.63}},
                plot_args:dict = {'fig_size': (32,10), 'rotation' : {'x' :90, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                  'label_size' : {'x' : 28, 'y' : 28}, 'title_size' :25, 'dpi' : 310, 'labels' : {'x' :'Protein Residues', 'y':"SS Fraction" },
                                  'legend_args' : {'font_size' : 25, 'loc' : 1, 'ncol':1}}) :
        
        kd=[]
        a=0
        if demux :
            s=self.s[1] ; title=self.title_names[1]
            for keys, value in kd_in.items():
                if 'demux' in keys :
                    kd.append(value)
                    a=a+1
        elif not demux :
            s = self.s[0]; title=self.title_names[0]
            for keys, value in kd_in.items():
                if 'rep' in keys :
                    kd.append(value)
                    a=a+1

        kd=np.array(kd).T
        # print(kd)

        fig, ax = plt.subplots(self.rows,self.cols, figsize=plot_args['fig_size'], sharex=True, sharey=True)

        for val in range(self.nreps):
            p, q = np.unravel_index(val,(self.rows,self.cols))
            
            ax[p,q].errorbar(range(0+self.offset, len(self.p_seq)+self.offset),np.array(input['helix'][f"{s}:{val}"]).T[0][:len(self.p_seq)],
                            yerr=np.array(input['helix'][f"{s}:{val}"]).T[1][:len(self.p_seq)], capsize=5,label='Helix',linewidth=2.5)
            ax[p,q].errorbar(range(0+self.offset, len(self.p_seq)+self.offset),np.array(input['sheet'][f"{s}:{val}"]).T[0][:len(self.p_seq)],
                            yerr=np.array(input['sheet'][f"{s}:{val}"]).T[1][:len(self.p_seq)], capsize=5,label='Sheet',linewidth=2.5)
            
            ax[p,q].tick_params(labelsize=plot_args['tick_size']['x'])
            ax[p,q].grid(alpha=0.4)
            plt.setp(ax[p,q].get_xticklabels(), rotation=plot_args['rotation']['x'])
            ax[p,q].set_ylim(0,1.0)
            ax[p,q].set_xlim(0+self.offset, len(self.p_seq)+self.offset)

            ax[p,q].set_xticks(range(0+self.offset, len(self.p_seq)+self.offset,2), self.p_seq[::2])
            if not demux : ax[p,q].set_title(f"{title} : {np.round(self.replica_temps[val],3)}K",size=plot_args['title_size'])
            elif demux : ax[p,q].set_title(f"{title} : {val}",size=plot_args['title_size'])
            
            ax2 = ax[p,q].twiny()

            # Set custom labels for the top x-axis
            # top_labels = c[::2]
            # top_tick_positions = range(0, 20, 2)

            # Use invisible spines and set labels for the top x-axis
            ax2.spines['top'].set_position(('outward', 0))
            ax2.spines['top'].set_visible(True)
            ax2.spines['bottom'].set_position(('outward', 0))
            ax2.spines['bottom'].set_visible(False)
            ax2.set_xlim(0+self.offset, len(self.p_seq)+self.offset)
            ax2.set_xticks(range(0+self.offset, len(self.p_seq)+self.offset,2), self.p_seq_a[::2])
            # ax2.set_xticklabels(top_labels)
            ax2.tick_params(axis='x', labelsize=20)
            plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
            ax2.tick_params(axis='x', which='minor', length=3, width=1, color='k', direction='out')
            
            if val == 0:
                ax[p,q].legend(loc=plot_args['legend_args']['loc'],prop={'size': plot_args['legend_args']['font_size']})
            
            if not q : ax[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
            if p == self.rows-1 : ax[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
            
            if kd_text :
                    
                text_=r'$K_{D}$'+' : '+str(round(kd[0][val],2))+r'$\pm$'+str(round(kd[1][val],2))
                ax[p,q].text(kd_text_args['loc']['x'], kd_text_args['loc']['y'], text_, fontsize = kd_text_args['size'])

            else : pass
            
        plt.tight_layout()

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_ss_all(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, kd_text:bool =False,  
                  kd_text_args:dict = {'size' :28, 'loc' : {'x' : 120.3, 'y': 0.63}}, plot_args_in:dict = {}) : 

        plot_args:dict = {'fig_size': (32,10), 'rotation' : {'x' :90, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                          'label_size' : {'x' : 28, 'y' : 28}, 'title_size' :25, 'dpi' : 310, 'labels' : {'x' :'Protein Residues', 'y':"SS Fraction" },
                          'legend_args' : {'font_size' : 25, 'loc' : 1, 'ncol':1}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]


        assert self.in_files_dict['ss'] , f"Load the secondary structure data for the plot!"

        if demux :
            out_file = f"{self.out_files_dict['ss']['both']}_demux.{self.out_file_type}"

        elif not demux : 
            out_file = f"{self.out_files_dict['ss']['both']}_rep.{self.out_file_type}"

        in_data={}
        in_data['helix'] = load_json(f"{self.data_dir}/{self.in_files_dict['ss']['helix']}")
        in_data['sheet'] = load_json(f"{self.data_dir}/{self.in_files_dict['ss']['sheet']}")

        if kd_text :

            assert self.in_files_dict['kd'] , f"Load the kd data for the plot!"

            kd_data = load_json(f"{self.data_dir}/{self.in_files_dict['kd']}")

        else : kd_data = {}

        self._plot_ss_all(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args, kd_in=kd_data, kd_text=kd_text, kd_text_args=kd_text_args)


    def _plot_ss_seperate(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False, plot_type:str='H',
                   plot_args:dict = {'fig_size': (10,8), 'rotation' : {'x' :90, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                     'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Protein Residues', 'y':"Secondary Structure Fraction" },
                                     'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':1}, 'title' :{'label' : '', 'size' : 28}}) :
        
        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1] 

        if plot_type == 'H' : t='helix' ; plot_args['title']['label'] = "Helical Propensities"
        elif plot_type == 'S' : t='sheet' ;  plot_args['title']['label'] = "Sheet Propensities"
        else : print(f"Mention the ss type you want to plot!\n")
                     
        fig, axes = plt.subplots(1,1,figsize=plot_args['fig_size'])

        for val in range(self.nreps):

            if not demux : label=f'{title} :{np.round(self.replica_temps[val],3)}K'
            elif demux : label=f'{title} :{val}'
            
            axes.errorbar(range(0+self.offset, len(self.p_seq)+self.offset),np.array(input[t][f"{s}:{val}"]).T[0][:len(self.p_seq)],yerr=np.array(input[t][f"{s}:{val}"]).T[1][:len(self.p_seq)],
                        capsize=5,label=label,linewidth=2.5,c=plt.cm.tab20c(val))
            

        axes.tick_params(labelsize=plot_args['label_size']['x'])
        axes.grid(alpha=0.4)
        plt.setp(axes.get_xticklabels(), rotation=plot_args['rotation']['x'])
        axes.set_ylim(0,1.0)
        axes.set_xlim(0+self.offset, len(self.p_seq)+self.offset)
        # ax[val].set_xticks(range(0+self.offset, len(self.p_seq)+self.offset,2), b[::2])
        axes.set_xticks(range(0+self.offset, len(self.p_seq)+self.offset,2), range(0+self.offset, len(self.p_seq)+self.offset,2))
        axes.legend( fontsize=plot_args['legend_args']['font_size'],ncol=plot_args['legend_args']['ncol']) 
        axes.set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
        axes.set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        axes.tick_params(axis='x', which='minor', length=3, width=1, color='k', direction='out')    
        axes.set_title(plot_args['title']['label'], size=plot_args['title']['size'])
        ax2 = axes.twiny()

        # Set custom labels for the top x-axis
        # top_labels = c[::2]
        # top_tick_positions = range(0, 20, 2)

        # Use invisible spines and set labels for the top x-axis
        ax2.spines['top'].set_position(('outward', 0))
        ax2.spines['top'].set_visible(True)
        ax2.spines['bottom'].set_position(('outward', 0))
        ax2.spines['bottom'].set_visible(False)
        ax2.set_xlim(0+self.offset, len(self.p_seq)+self.offset)
        ax2.set_xticks(range(0+self.offset, len(self.p_seq)+self.offset,2), self.p_seq_a[::2])
        # ax2.set_xticklabels(top_labels)
        ax2.tick_params(axis='x', labelsize=20)
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax2.tick_params(axis='x', which='minor', length=3, width=1, color='k', direction='out')
            
        #     text_=r'$K_{D}$'+' : '+str(round(np.array(bf_kd['rep'])[2][val],2))+r'$\pm$'+str(round(np.array(bf_kd['rep'])[3][val],2))
        #     ax[p,q].text(120.3, 0.63, text_, fontsize = 28)
        plt.tight_layout()
        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_ss_seperate(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, plot_args_in:dict = {},ss_type:str='H') : 

        plot_args:dict = {'fig_size': (10,8), 'rotation' : {'x' :90, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                          'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Protein Residues', 'y':"Secondary Structure Fraction" },
                          'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':2}, 'title' :{'label' : 'Helical Propensities', 'size' : 28}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]


        assert self.in_files_dict['ss'] , f"Load the secondary structure data for the plot!"

        if demux :
            if ss_type=='H':
                out_file = f"{self.out_files_dict['ss']['helix']}_demux.{self.out_file_type}"
            elif ss_type=='S':
                out_file = f"{self.out_files_dict['ss']['sheet']}_demux.{self.out_file_type}"

        elif not demux :
            if ss_type=='H':
                out_file = f"{self.out_files_dict['ss']['helix']}_rep.{self.out_file_type}"
            elif ss_type=='S':
                out_file = f"{self.out_files_dict['ss']['sheet']}_rep.{self.out_file_type}"

        in_data={}
        in_data['helix'] = load_json(f"{self.data_dir}/{self.in_files_dict['ss']['helix']}")
        in_data['sheet'] = load_json(f"{self.data_dir}/{self.in_files_dict['ss']['sheet']}")



        self._plot_ss_seperate(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args, plot_type=ss_type)


    def _plot_rg(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False, time_step_in_ps:float=80.0, convolve_step_size:int=200,
                   plot_args:dict = {'fig_size': (30,10), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                     'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Time ($\mu$s)', 'y':"Rg (nm)" },
                                     'title' :{'label' : '', 'size' : 28}}) :
        
        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1]
 

        fig, ax = plt.subplots(self.rows, self.cols, figsize=plot_args['fig_size'], sharex=True, sharey=True)

        N=convolve_step_size

        time={}
        rg_min=[]
        rg_max=[]

        for val in range(self.nreps):
            t=[]

            rg_min.append(np.array(input[f"{s}:{val}"]).min())
            rg_max.append(np.array(input[f"{s}:{val}"]).max())

            for j in range(len(input[f"{s}:{val}"])):
                t.append((j*time_step_in_ps)/10**6)

            time[f'{s}:{val}']=np.array(t)

        # print(time)
        p_min=min(rg_min)
        p_max=max(rg_max)
        for val in range(self.nreps):
            p, q = np.unravel_index(val,(self.rows, self.cols))

            rg_min = np.array(input[f"{s}:{val}"]).min()
            rg_max = np.array(input[f"{s}:{val}"]).max()

            ax[p,q].plot(time[f"{s}:{val}"],np.array(input[f"{s}:{val}"]))
            ax[p,q].plot(np.convolve(time[f"{s}:{val}"], np.ones(N)/N, mode='valid'),
                        np.convolve(np.array(input[f"{s}:{val}"]), np.ones(N)/N, mode='valid'),linewidth=2.5)
            
            ax[p,q].tick_params(labelsize=plot_args['tick_size']['x'])
            plt.setp(ax[p,q].get_xticklabels(), rotation=plot_args['rotation']['x'])
            ax[p,q].set_ylim(p_min-0.3,p_max+0.3)
            #ax[p,q].set_xticks(range(0,142,2))
            ax[p,q].set_yticks(np.arange(p_min-0.3,p_max+0.3,0.4))
            
            if not demux : ax[p,q].set_title(f"{title} : {np.round(self.replica_temps[val],3)}K",size=plot_args['title']['size'])
            elif demux : ax[p,q].set_title(f"{title} : {val}",size=plot_args['title']['size'])

            if not q : ax[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
            if p == self.rows-1 : ax[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
                
        plt.tight_layout()

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_rg(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, plot_args_in:dict = {}, time_step_in_ps:float=80.0, convolve_step_size:int=200) : 

        plot_args:dict = {'fig_size': (30,10), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                          'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Time ($\mu$s)', 'y':"Rg (nm)" },
                          'title' :{'label' : '', 'size' : 28}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]


        assert self.in_files_dict['rg'] , f"Load the Rg data for the plot!"

        if demux :
            out_file = f"{self.out_files_dict['rg']}_demux.{self.out_file_type}"

        elif not demux : 
            out_file = f"{self.out_files_dict['rg']}_rep.{self.out_file_type}"

        in_data= load_json(f"{self.data_dir}/{self.in_files_dict['rg']}")

        self._plot_rg(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args, time_step_in_ps=time_step_in_ps, convolve_step_size=convolve_step_size)


    def _plot_rg_hist(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False, time_step_in_ps:float=80.0, convolve_step_size:int=200,
                   plot_args:dict = {'fig_size': (12,8), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                     'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Rg (nm)', 'y':"Distribution" },
                                     'title' :{'label' : '', 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':2}}) :
        
        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1]

        plt.figure(figsize=plot_args['fig_size'])

        for val in range(self.nreps):

            if not demux : label=f'{title} :{np.round(self.replica_temps[val],3)}K'
            elif demux : label=f'{title} :{val}'
            
            plt.hist(np.array(input[f"{s}:{val}"]),bins=40,density=True,histtype='step',color=plt.cm.tab20c(val),label=label)
            
            plt.tick_params(labelsize=plot_args['tick_size']['x'])
            #ax[p,q].set_ylim(1,4)
            #ax[p,q].set_xticks(range(0,142,2))
            #ax[p,q].set_yticks(np.arange(1.0,4.0,0.4))
            plt.vlines(np.mean(np.array(input[f"{s}:{val}"])),0,2, color=plt.cm.tab20c(val), lw=2)    

            
        plt.title(title,size=plot_args['title']['size'])
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        plt.legend(loc=plot_args['legend_args']['loc'], fontsize=plot_args['legend_args']['font_size'],ncol=plot_args['legend_args']['ncol'])
        plt.ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
        plt.xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
                
        plt.tight_layout()

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_rg_hist(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, plot_args_in:dict = {}) : 

        plot_args:dict = {'fig_size': (12,8), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                          'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Rg (nm)', 'y':"Distribution" },
                          'title' :{'label' : '', 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':2}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]


        assert self.in_files_dict['rg'] , f"Load the Rg data for the plot!"

        if demux :
            out_file = f"{self.out_files_dict['rg_hist']}_demux.{self.out_file_type}"

        elif not demux : 
            out_file = f"{self.out_files_dict['rg_hist']}_rep.{self.out_file_type}"

        in_data= load_json(f"{self.data_dir}/{self.in_files_dict['rg']}")

        self._plot_rg_hist(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args)


    def _plot_kd_time(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False, kd_text:bool =True, kd_in:dict={},
                      kd_text_args:dict = {'size' :28, 'loc' : {'x' : 0.45, 'y': 33}},
                   plot_args:dict = {'fig_size': (32,10), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                     'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Time ($\mu$s)', 'y':'$K_D$ (mM)' },
                                     'title' :{'label' : '', 'size' : 28}}) :
        
        kd=[]
        a=0
        if demux :
            s=self.s[1] ; title=self.title_names[1]
            for keys, value in kd_in.items():
                if 'demux' in keys :
                    kd.append(value)
                    a=a+1
        elif not demux :
            s = self.s[0]; title=self.title_names[0]
            for keys, value in kd_in.items():
                if 'rep' in keys :
                    kd.append(value)
                    a=a+1

        kd=np.array(kd).T

        fig, ax = plt.subplots(self.rows,self.cols, figsize=plot_args['fig_size'], sharex=True, sharey=True)

        for val in range(self.nreps):
            p, q = np.unravel_index(val,(self.rows, self.cols))
            
            inf_start=len(np.where(np.array(input[f"{s}:{val}"]).T[4]==0)[0])

            ax[p,q].plot(np.array(input[f"{s}:{val}"]).T[0][inf_start::], np.array(input[f"{s}:{val}"]).T[1][inf_start::], 
                        color='blue',linewidth=2)
            ax[p,q].fill_between(np.array(input[f"{s}:{val}"]).T[0][inf_start::],np.array(input[f"{s}:{val}"]).T[2][inf_start::],
                                 np.array(input[f"{s}:{val}"]).T[3][inf_start::],color='blue', alpha=0.2)

            

            ax[p,q].grid()
            ax[p, q].set_ylim(0, 40.0)
            ax[p,q].tick_params(labelsize=plot_args['tick_size']['x'])
            #plt.setp(ax[p,q].get_xticklabels(), rotation=45)
            
            ax[p,q].set_title(title,size=plot_args['title']['size'])
            
            if kd_text :
                    
                text_=r'$K_{D}$'+' : '+str(round(kd[0][val],2))+r'$\pm$'+str(round(kd[1][val],2))
                ax[p,q].text(kd_text_args['loc']['x'], kd_text_args['loc']['y'], text_, fontsize = kd_text_args['size'])

            else : pass
            
            if not q : ax[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
            if p == self.rows-1 : ax[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
                
            
        plt.tight_layout()

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_kd_time(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False,kd_text:bool =False,
                      kd_text_args:dict = {'size' :28, 'loc' : {'x' : 0.45, 'y': 33}}, plot_args_in:dict = {}) : 

        plot_args:dict = {'fig_size': (32,10), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                          'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 210, 'labels' : {'x' :'Time ($\mu$s)', 'y':'$K_D$ (mM)' },
                          'title' :{'label' : '', 'size' : 28}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]


        assert self.in_files_dict['kd_time'] , f"Load the Kd data for the plot!"

        if demux :
            out_file = f"{self.out_files_dict['kd_time']}_demux.{self.out_file_type}"

        elif not demux : 
            out_file = f"{self.out_files_dict['kd_time']}_rep.{self.out_file_type}"

        in_data= load_json(f"{self.data_dir}/{self.in_files_dict['kd_time']}")

        if kd_text :

            kd_data = load_json(f"{self.data_dir}/{self.in_files_dict['kd']}")

        else : kd_data = {}

        self._plot_kd_time(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args, kd_in=kd_data, kd_text=kd_text, kd_text_args=kd_text_args)

    def _plot_lig_contacts(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False,
                           plot_args:dict = {'fig_size': (12,8), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                             'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Residue', 'y':'Contact probability' },
                                             'title' :{'label' : '', 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':1}}):
        
        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1]

        fig, axes = plt.subplots(1,1,figsize=plot_args['fig_size'])

        for i in range(self.nreps):

            if not demux : label=f'{title} :{np.round(self.replica_temps[i],3)}K'
            elif demux : label=f'{title} :{i}'
        
            axes.plot(np.array(input[f"{s}:{i}"]).T[0], np.array(input[f"{s}:{i}"]).T[1], linewidth=2.5,
                    c=plt.cm.tab20c(i),label=label)

            axes.fill_between(np.array(input[f"{s}:{i}"]).T[0], np.array(input[f"{s}:{i}"]).T[1]-np.array(input[f"{s}:{i}"]).T[2],
                            np.array(input[f"{s}:{i}"]).T[1]+np.array(input[f"{s}:{i}"]).T[2], alpha=0.2,color=plt.cm.tab20c(i))


        axes.tick_params(labelsize=plot_args['tick_size']['x'])
        plt.setp(axes.get_xticklabels(), rotation=plot_args['rotation']['x'])

        axes.set_title(title,size=plot_args['title']['size'])
        axes.set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
        axes.set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
        axes.set_ylim(0,1.0)
        # axes.set_xlim(120,141)
        axes.set_xlim(0+self.offset-1,len(self.p_seq)+self.offset)
        axes.set_xticks(range(0+self.offset, len(self.p_seq)+self.offset,2), range(0+self.offset, len(self.p_seq)+self.offset,2))
        axes.legend(loc=plot_args['legend_args']['loc'], fontsize=plot_args['legend_args']['font_size'])
        axes.grid(alpha=0.4)
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        axes.tick_params(axis='x', which='minor', length=3, width=1, color='k', direction='out')    
        axes.set_title(plot_args['title']['label'], size=plot_args['title']['size'])
        ax2 = axes.twiny()

        # Set custom labels for the top x-axis
        # top_labels = c[::2]
        # top_tick_positions = range(0, 20, 2)

        # Use invisible spines and set labels for the top x-axis
        ax2.spines['top'].set_position(('outward', 0))
        ax2.spines['top'].set_visible(True)
        ax2.spines['bottom'].set_position(('outward', 0))
        ax2.spines['bottom'].set_visible(False)
        ax2.set_xlim(0+self.offset-1, len(self.p_seq)+self.offset)
        ax2.set_xticks(range(0+self.offset, len(self.p_seq)+self.offset), self.p_seq_a[::])
        # ax2.set_xticklabels(top_labels)
        ax2.tick_params(axis='x', labelsize=20)
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax2.tick_params(axis='x', which='minor', length=3, width=1, color='k', direction='out')
            
        #     text_=r'$K_{D}$'+' : '+str(round(np.array(bf_kd['rep'])[2][val],2))+r'$\pm$'+str(round(np.array(bf_kd['rep'])[3][val],2))
        #     ax[p,q].text(120.3, 0.63, text_, fontsize = 28)

        plt.tight_layout()

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_avg_lig_contacts(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, plot_args_in:dict = {}):

        plot_args:dict = {'fig_size': (12,8), 'rotation' : {'x' :90, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                            'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Residue', 'y':'Contact probability' },
                            'title' :{'label' : '', 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':1}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]


        assert self.in_files_dict['lig_contacts'] , f"Load the average ligand:protein contacts data for the plot!"

        if demux :
            out_file = f"{self.out_files_dict['lig_contacts']}_demux.{self.out_file_type}"

        elif not demux : 
            out_file = f"{self.out_files_dict['lig_contacts']}_rep.{self.out_file_type}"

        in_data= load_json(f"{self.data_dir}/{self.in_files_dict['lig_contacts']}")

        self._plot_lig_contacts(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args)

    def _plot_lig_modes_grid(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False, modes_keys:list=[],
                           plot_args:dict = {'fig_size': (12,8), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                             'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Residue', 'y': '' },
                                             'title' :{'label' : {}, 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':1}}):
        
        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1]

        fig, axes = plt.subplots(2,2,figsize=plot_args['fig_size'], sharex=True, sharey=True)

        bx=0
        for ax in modes_keys:
            p, q = np.unravel_index(bx,(2,2))

            for i in range(self.nreps):

                if not demux : label=f'Replica : {self.replica_temps[i]:.3f} K'
                elif demux : label=f'Replica :{i}'
            
                axes[p,q].plot(np.array(input[ax][f"{s}:{i}"]).T[0], np.array(input[ax][f"{s}:{i}"]).T[1], linewidth=2.5,
                        c=plt.cm.tab20c(i),label=label)

                axes[p,q].fill_between(np.array(input[ax][f"{s}:{i}"]).T[0], np.array(input[ax][f"{s}:{i}"]).T[1]-np.array(input[ax][f"{s}:{i}"]).T[2],
                                np.array(input[ax][f"{s}:{i}"]).T[1]+np.array(input[ax][f"{s}:{i}"]).T[2], alpha=0.2,color=plt.cm.tab20c(i))


            axes[p,q].tick_params(labelsize=plot_args['tick_size']['x'])
            plt.setp(axes[p,q].get_xticklabels(), rotation=plot_args['rotation']['x'])

            axes[p,q].set_title(plot_args['title']['label'][ax],size=plot_args['title']['size'])
            # axes[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
            # axes[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
            axes[p,q].set_ylim(0,1.0)
            axes[p,q].set_xlim(0+self.offset-1,len(self.p_seq)+self.offset)
            axes[p,q].set_xticks(range(0+self.offset, len(self.p_seq)+self.offset,2), range(0+self.offset, len(self.p_seq)+self.offset,2))
            # axes[p,q].legend(loc=plot_args['legend_args']['loc'], fontsize=plot_args['legend_args']['font_size'])
            axes[p,q].grid(alpha=0.4)
            axes[p,q].xaxis.set_minor_locator(plt.MultipleLocator(1))
            axes[p,q].tick_params(axis='x', which='minor', length=3, width=1, color='k', direction='out')    
            ax2 = axes[p,q].twiny()

            # Set custom labels for the top x-axis
            # top_labels = c[::2]
            # top_tick_positions = range(0, 20, 2)

            # Use invisible spines and set labels for the top x-axis
            ax2.spines['top'].set_position(('outward', 0))
            ax2.spines['top'].set_visible(True)
            ax2.spines['bottom'].set_position(('outward', 0))
            ax2.spines['bottom'].set_visible(False)
            ax2.set_xlim(0+self.offset-1, len(self.p_seq)+self.offset)
            ax2.set_xticks(range(0+self.offset, len(self.p_seq)+self.offset), self.p_seq_a[::])
            # ax2.set_xticklabels(top_labels)
            ax2.tick_params(axis='x', labelsize=20)
            # plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
            ax2.tick_params(axis='x', which='minor', length=3, width=1, color='k', direction='out')
                
            #     text_=r'$K_{D}$'+' : '+str(round(np.array(bf_kd['rep'])[2][val],2))+r'$\pm$'+str(round(np.array(bf_kd['rep'])[3][val],2))
            #     ax[p,q].text(120.3, 0.63, text_, fontsize = 28)

            if p == 0 and q == 0:
                axes[p,q].legend(ncol=2,loc=plot_args['legend_args']['loc'],prop={'size': plot_args['legend_args']['font_size']})

            if not q : axes[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
            if p == 2-1 : axes[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
            
            bx+=1

        fig.suptitle(title, size=plot_args['title']['size'])
        plt.tight_layout()

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_lig_modes_grid(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, plot_args_in:dict = {}):

        plot_args = {'fig_size': (18,15), 'rotation' : {'x' :45, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                            'label_size' : {'x' :24, 'y' : 24}, 'dpi' : 310,'labels' : {'x' :'Residue', 'y':"Probability"},
                            'title' :{'label' : {'aro' : 'Aromatic contacts',
                                                 'hyphob' : 'Hydrophobic contacts',
                                                 'hbond' : 'H-bond contacts',
                                                 'charge' :'Charge contacts'}, 'size' : 28},
                             'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':1}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]

        if self.in_files_dict['lig_modes'] :
                
            assert self.in_files_dict['lig_modes']['aro'] , f"Load the ligand:protein aromatic contacts data for the plot!"
            assert self.in_files_dict['lig_modes']['hyphob'] , f"Load the ligand:protein hydrophobic contacts data for the plot!"
            assert self.in_files_dict['lig_modes']['hbond'] , f"Load the ligand:protein H-bond contacts data for the plot!"
            assert self.in_files_dict['lig_modes']['charge'] , f"Load the ligand:protein charge contacts data for the plot!"

        elif not self.in_files_dict['lig_modes'] : raise Exception("Give a dictonory of ligand:protein interaction files\
                                                                    (.json) with keywords ['aro','hyphob','hbond','charge']")


        assert self.out_files_dict['lig_modes']['togeather'] , "Give a file name for ligand mode grid plots with key 'togeather'!!"

        if demux :
            out_file = f"{self.out_files_dict['lig_modes']['togeather']}_demux.{self.out_file_type}"

        elif not demux : 
            out_file = f"{self.out_files_dict['lig_modes']['togeather']}_rep.{self.out_file_type}"

        in_data={}
        modes_keys=[]
        for k in self.in_files_dict['lig_modes'].keys():
                
            in_data[k] = load_json(f"{self.data_dir}/{self.in_files_dict['lig_modes'][k]}")
            modes_keys.append(k)
            if plot_args_in : plot_args['title']['label'][k]=plot_args_in['title']['label'][k]

        modes_keys=np.asarray(modes_keys)
        self._plot_lig_modes_grid(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args, modes_keys=modes_keys)

    def _plot_lig_modes(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False, modes_keys:list=[], kd_text:bool =True, kd_in:dict={}, 
                        charge:bool =True, kd_text_args:dict = {'size' :20, 'loc' : {'x' : 121, 'y': 0.75}},
                           plot_args:dict = {'fig_size': (32,10), 'rotation' : {'x' :45, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                             'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Residue', 'y': 'Probability' },
                                             'title' :{'label' : '', 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':1}}):
        
        if kd_text : kd=[]
        a=0
        if demux :
            s=self.s[1] ; title=self.title_names[1]

            if kd_text :
                for keys, value in kd_in.items():
                    if 'demux' in keys :
                        kd.append(value)
                        a=a+1
            else : pass

        elif not demux :
            s = self.s[0]; title=self.title_names[0]

            if kd_text :    
                for keys, value in kd_in.items():
                    if 'rep' in keys :
                        kd.append(value)
                        a=a+1

        if kd_text : kd=np.array(kd).T

        for k in modes_keys : 

            assert input[k] , f"Keyword {k} is missing!!"

        fig, ax = plt.subplots(self.rows,self.cols, figsize=plot_args['fig_size'], sharex=True, sharey=True)

        for i in range(self.nreps):
            p, q = np.unravel_index(i,(self.rows, self.cols))
            
            ax[p, q].plot(np.array(input['hbond'][f"{s}:{i}"]).T[0], np.array(input['hbond'][f"{s}:{i}"]).T[1], linewidth=2.5,
                                label='Hydrogen Bond', color='red')
            ax[p, q].fill_between(np.array(input['hbond'][f"{s}:{i}"]).T[0], np.array(input['hbond'][f"{s}:{i}"]).T[1]-np.array(input['hbond'][f"{s}:{i}"]).T[2],
                                    np.array(input['hbond'][f"{s}:{i}"]).T[1]+np.array(input['hbond'][f"{s}:{i}"]).T[2], alpha=0.2,color='r')
            
            
            ax[p, q].plot(np.array(input['aro'][f"{s}:{i}"]).T[0], np.array(input['aro'][f"{s}:{i}"]).T[1], linewidth=2.5,
                                color='black', label='Aromatic Stacking')
            ax[p, q].fill_between(np.array(input['aro'][f"{s}:{i}"]).T[0], np.array(input['aro'][f"{s}:{i}"]).T[1]-np.array(input['aro'][f"{s}:{i}"]).T[2],
                                    np.array(input['aro'][f"{s}:{i}"]).T[1]+np.array(input['aro'][f"{s}:{i}"]).T[2], alpha=0.2,color='black')
            
            if charge :
                    
                ax[p, q].plot(np.array(input['charge'][f"{s}:{i}"]).T[0], np.array(input['charge'][f"{s}:{i}"]).T[1], linewidth=2.5,
                                    label='Charge Contacts',c='blue')
                ax[p, q].fill_between(np.array(input['charge'][f"{s}:{i}"]).T[0], np.array(input['charge'][f"{s}:{i}"]).T[1]-np.array(input['charge'][f"{s}:{i}"]).T[2],
                                        np.array(input['charge'][f"{s}:{i}"]).T[1]+np.array(input['charge'][f"{s}:{i}"]).T[2], alpha=0.2,color='b')
            else : pass     
            ax[p, q].plot(np.array(input['hyphob'][f"{s}:{i}"]).T[0], np.array(input['hyphob'][f"{s}:{i}"]).T[1], linewidth=2.5,
                                label='Hydrophobic Contacts', color='green')
            ax[p, q].fill_between(np.array(input['hyphob'][f"{s}:{i}"]).T[0], np.array(input['hyphob'][f"{s}:{i}"]).T[1]-np.array(input['hyphob'][f"{s}:{i}"]).T[2],
                                    np.array(input['hyphob'][f"{s}:{i}"]).T[1]+np.array(input['hyphob'][f"{s}:{i}"]).T[2], alpha=0.2,color='g')
            
            ax[p,q].tick_params(labelsize=plot_args['tick_size']['x'])
            plt.setp(ax[p,q].get_xticklabels(), rotation=plot_args['rotation']['x'])

            if not demux : ax[p,q].set_title(f"{title} : {self.replica_temps[i]:.3f} K",size=plot_args['title']['size'])
            elif demux : ax[p,q].set_title(f"{title} : {i}",size=plot_args['title']['size'])

            ax[p, q].set_ylim(0,1.0)
            ax[p,q].set_xlim(120,141)
            ax[p,q].set_xticks(range(0+self.offset, len(self.p_seq)+self.offset,2), range(0+self.offset, len(self.p_seq)+self.offset,2))
            ax[p,q].grid(alpha=0.4)
            ax[p,q].xaxis.set_minor_locator(plt.MultipleLocator(1))
            ax[p,q].tick_params(axis='x', which='minor', length=3, width=1, color='k', direction='out')    
            
            ax2 = ax[p,q].twiny()

            # Set custom labels for the top x-axis
            # top_labels = c[::2]
            # top_tick_positions = range(0, 20, 2)

            # Use invisible spines and set labels for the top x-axis
            ax2.spines['top'].set_position(('outward', 0))
            ax2.spines['top'].set_visible(True)
            ax2.spines['bottom'].set_position(('outward', 0))
            ax2.spines['bottom'].set_visible(False)
            ax2.set_xlim(0+self.offset, len(self.p_seq)+self.offset)
            ax2.set_xticks(range(0+self.offset, len(self.p_seq)+self.offset), self.p_seq_a[::])
            # ax2.set_xticklabels(top_labels)
            ax2.tick_params(axis='x', labelsize=20)
            # plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
            ax2.tick_params(axis='x', which='minor', length=3, width=1, color='k', direction='out')
                
            if kd_text :
                    
                text_=r'$K_{D}$'+' : '+str(round(kd[0][i],2))+r'$\pm$'+str(round(kd[1][i],2))
                ax[p,q].text(kd_text_args['loc']['x'], kd_text_args['loc']['y'], text_, fontsize = kd_text_args['size'])

            else : pass

            if p == 0 and q == 0:
                ax[p,q].legend(ncol=2,loc=plot_args['legend_args']['loc'],prop={'size': plot_args['legend_args']['font_size']})

            if not q : ax[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
            if p == self.rows-1 : ax[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)

        plt.tight_layout()

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_lig_modes(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, kd_text:bool =False, charge:bool = True,
                        kd_text_args:dict = {'size' :20, 'loc' : {'x' : 121, 'y': 0.75}}, plot_args_in:dict = {}):

        plot_args:dict = {'fig_size': (32,10), 'rotation' : {'x' :45, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                          'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Residue', 'y': 'Probability' },
                          'title' :{'label' : '', 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':1}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]

        if self.in_files_dict['lig_modes'] :
        
            assert self.in_files_dict['lig_modes']['aro'] , f"Load the ligand:protein aromatic contacts data for the plot!"
            assert self.in_files_dict['lig_modes']['hyphob'] , f"Load the ligand:protein hydrophobic contacts data for the plot!"
            assert self.in_files_dict['lig_modes']['hbond'] , f"Load the ligand:protein H-bond contacts data for the plot!"
            assert self.in_files_dict['lig_modes']['charge'] , f"Load the ligand:protein charge contacts data for the plot!"

        elif not self.in_files_dict['lig_modes'] : raise Exception("Give a dictonory of ligand:protein interaction files\
                                                                    (.json) with keywords ['aro','hyphob','hbond','charge']")


        assert self.out_files_dict['lig_modes']['seperate'] , "Give a file name for ligand mode seperate plots with key 'seperate'!!"

        if demux :
            out_file = f"{self.out_files_dict['lig_modes']['seperate']}_demux.{self.out_file_type}"

        elif not demux : 
            out_file = f"{self.out_files_dict['lig_modes']['seperate']}_rep.{self.out_file_type}"

        if kd_text :

            assert self.in_files_dict['kd'] , f"Load the kd data for the plot!"

            kd_data = load_json(f"{self.data_dir}/{self.in_files_dict['kd']}")

        else : kd_data = {}

        in_data={}
        modes_keys=[]
        for k in self.in_files_dict['lig_modes'].keys():
        
            in_data[k] = load_json(f"{self.data_dir}/{self.in_files_dict['lig_modes'][k]}")
            modes_keys.append(k)
            if 'title' in plot_args_in.keys() : plot_args['title']['label'][k]=plot_args_in['title']['label'][k]

        modes_keys=np.asarray(modes_keys)
        self._plot_lig_modes(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args, modes_keys=modes_keys,
                             kd_text=kd_text, kd_in=kd_data, kd_text_args=kd_text_args)

    def _plot_2d_rg(self, input_rg:dict, input_ss:dict, cbar_label:str, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False,
                    plot_args:dict = {'vmin': 0.0, 'vmax': 1.0, 'fig_size': (30,14), 'cax_coor': [0.93, 0.2, 0.02, 0.6], 'cmap': 'jet', 'aspect' : 'auto', 'rotation' : {'x' : 0, 'y' :0},
                                      'interpolation' : 'gaussian', 'extent' : [],
                                      'tick_size' : {'x' : 18, 'y' : 18, 'cax' :25}, 'label_size' : {'x' : 30, 'y' : 30, 'cax' :30}, 'title_size' :20, 'dpi' : 310, 'labels' : {'x' :"Residues", 'y':"Residues" }},
                    hist_args:dict ={'nbins' :30, 'density':True, 'weights':None, 'y_min_max': [0.6, 2.0], 'x_min_max': [0, 5.0]}, temperature:float = 300.0, grid:bool = True):
        
        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1]

        fig, axes = plt.subplots(self.rows,self.cols, figsize=plot_args['fig_size'], sharex=True, sharey=True)
        if plot_args['cax_coor'] : cax= fig.add_axes(plot_args['cax_coor'])
        else : cax = fig.add_axes([axes[self.rows-1,self.cols-1].get_position().x1+0.02,axes[1,2].get_position().y0,0.02,axes[0,0].get_position().y1-axes[self.rows-1,self.cols-1].get_position().y0])

        images=[]
        for i in range(self.nreps):
            p,q = np.unravel_index(i,(self.rows, self.cols))
 
            X=np.sum(np.array(input_ss[f"{s}:{i}"]),axis=0)
            a, xedges, yedges = np.histogram2d(np.array(input_rg[f"{s}:{i}"]), X, hist_args['nbins'], [hist_args['y_min_max'],hist_args['x_min_max']],
                                               density=True, weights=None)

            plot_args['extent'] = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
            a = np.log(np.flipud(a)+.000001)
            T = temperature
            a = -(0.001987*T)*a

            # im = axes[p,q].imshow(a, vmin=plot_args['vmin'], vmax=plot_args['vmax'],cmap=plot_args['cmap'], aspect=plot_args['aspect'],
            #                       interpolation='gaussian', extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]])
            
            im = axes[p,q].imshow(a, cmap=plot_args['cmap'], aspect=plot_args['aspect'],
                                  interpolation=plot_args['interpolation'], extent=plot_args['extent'])

            imaxes = plt.gca()

            axes[p,q].tick_params(labelsize=plot_args['tick_size']['x'])
            # axes[p,q].set_xlim(hist_args['x_min_max'][0], hist_args['x_min_max'][1])
            xl=np.arange(hist_args['x_min_max'][0], hist_args['x_min_max'][1])
            yl=np.arange(hist_args['y_min_max'][0], hist_args['y_min_max'][1])
            # axes[p,q].set_xticks(xl, xl, rotation=plot_args['rotation']['x'], size=plot_args['tick_size']['x'])
            # axes[p,q].set_yticks(yl, yl, rotation=plot_args['rotation']['y'], size=plot_args['tick_size']['y'])
            if grid : axes[p,q].grid()
            else : axes[p,q].grid(False)
            if not demux : axes[p,q].set_title(f"{title} : {np.round(self.replica_temps[i],3)}K",size=plot_args['title_size'])
            elif demux : axes[p,q].set_title(f"{title} : {i}",size=plot_args['title_size'])

            if not q : axes[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
            if p == self.rows-1 : axes[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
            images.append(im)

        cbar = fig.colorbar(images[-1],cax=cax)
        cbar.set_label(cbar_label, size=plot_args['label_size']['cax'])
        cbar.ax.tick_params(labelsize=plot_args['tick_size']['cax'])

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass


    def plot_2d_rg_sa(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, temperature:float = 300.0, 
                      plot_args_in:dict = {}, hist_args_in:dict={},grid:bool = True) :

        plot_args:dict = {'vmin': 0.0, 'vmax': 0.5, 'fig_size': (30,14), 'cax_coor': [0.93, 0.2, 0.02, 0.6], 'cmap': 'jet', 'aspect' : 'auto', 'rotation' : {'x' : 0, 'y' :0},
                           'interpolation' : 'gaussian', 'extent' : [],
                           'tick_size' : {'x' : 18, 'y' : 18, 'cax' :25}, 'label_size' : {'x' : 30, 'y' : 30, 'cax' :30}, 'title_size' :20, 'dpi' : 310, 'labels' : {'y' :"Rg (nm)", 'x':"Sα" }}

        hist_args:dict ={'nbins' :30, 'density':True, 'weights':None, 'y_min_max': [0.6, 2.0], 'x_min_max': [0, 4.0]}
    
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]

        if hist_args_in :

            for k in hist_args_in.keys():

                hist_args[k] = hist_args_in[k]

        assert self.in_files_dict['rg_2d']['rg'] , f"Load the Rg data for the plot with keyword rg!"
        assert self.in_files_dict['rg_2d']['sa'] , f"Load the SAlpha RMSD data for the plot with keyword sa!"
        
        if demux : out_file = f"{self.out_files_dict['rg_2d']}_demux.{self.out_file_type}"
        elif not demux : out_file = f"{self.out_files_dict['rg_2d']}_rep.{self.out_file_type}"

        in_data_rg = load_json(f"{self.data_dir}/{self.in_files_dict['rg_2d']['rg']}")
        in_data_sa = load_json(f"{self.data_dir}/{self.in_files_dict['rg_2d']['sa']}")

        self._plot_2d_rg(input_rg=in_data_rg, input_ss=in_data_sa, cbar_label='FE (kcal/mol)', file_name=out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args, hist_args=hist_args, temperature=temperature, grid=grid)


    def _plot_sa_pp(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False, time_step_in_ps:float=80.0, convolve_step_size:int=200,
                   plot_args:dict = {'fig_size': (30,10), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                     'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Time ($\mu$s)', 'y':"" },
                                     'title' :{'label' : '', 'size' : 28}}) :
        
        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1]
 

        fig, ax = plt.subplots(self.rows, self.cols, figsize=plot_args['fig_size'], sharex=True, sharey=True)

        N=convolve_step_size


        time={}
        X={}
        rg_min=[]
        rg_max=[]
        for val in range(self.nreps):

            X[val]=np.sum(np.array(input[f"{s}:{val}"]),axis=0)
            rg_min.append(X[val].min())
            rg_max.append(X[val].max())

            t=[]
            for j in range(len(X[val])):
                t.append((j*time_step_in_ps)/10**6)

            time[f'{s}:{val}']=np.array(t)

        p_min=min(rg_min)
        p_max=max(rg_max)


        for val in range(self.nreps):
            p, q = np.unravel_index(val,(self.rows, self.cols))


            ax[p,q].plot(time[f"{s}:{val}"],X[val])
            ax[p,q].plot(np.convolve(time[f"{s}:{val}"], np.ones(N)/N, mode='valid'),
                        np.convolve(X[val], np.ones(N)/N, mode='valid'),linewidth=2.5)
            
            ax[p,q].tick_params(labelsize=plot_args['tick_size']['x'])
            plt.setp(ax[p,q].get_xticklabels(), rotation=plot_args['rotation']['x'])
            ax[p,q].set_ylim(0,p_max)
            #ax[p,q].set_xticks(range(0,142,2))
            ax[p,q].set_yticks(np.arange(0,p_max,1))
            
            if not demux : ax[p,q].set_title(f"{title} : {np.round(self.replica_temps[val],3)}K",size=plot_args['title']['size'])
            elif demux : ax[p,q].set_title(f"{title} : {val}",size=plot_args['title']['size'])

            if not q : ax[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
            if p == self.rows-1 : ax[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
                
        plt.tight_layout()

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_sa_pp(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, plot_args_in:dict = {}, time_step_in_ps:float=80.0, convolve_step_size:int=200, sa_or_pp:str='sa') : 

        plot_args:dict = {'fig_size': (30,10), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                          'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Time ($\mu$s)', 'y': '' },
                          'title' :{'label' : '', 'size' : 28}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]

        if sa_or_pp=='sa' :

            plot_args['labels']['y'] = "Sα"    
            assert self.in_files_dict['sa'] , f"Load the SAlpha RMSD data for the plot with keyword sa!"

            if demux :
                out_file = f"{self.out_files_dict['sa']}_demux.{self.out_file_type}"

            elif not demux : 
                out_file = f"{self.out_files_dict['sa']}_rep.{self.out_file_type}"

            in_data= load_json(f"{self.data_dir}/{self.in_files_dict['sa']}")

        elif sa_or_pp=='pp' :

            plot_args['labels']['y'] = "PP-II"    
            assert self.in_files_dict['pp'] , f"Load the Poly-Prolein RMSD data for the plot with keyword pp!"

            if demux :
                out_file = f"{self.out_files_dict['pp']}_demux.{self.out_file_type}"

            elif not demux : 
                out_file = f"{self.out_files_dict['pp']}_rep.{self.out_file_type}"

            in_data= load_json(f"{self.data_dir}/{self.in_files_dict['pp']}")


        self._plot_sa_pp(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args, time_step_in_ps=time_step_in_ps, convolve_step_size=convolve_step_size)


    def _plot_phipsi(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False,
                   plot_args:dict = {'fig_size': (30,10), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                     'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Alphabeta RMSD', 'y':"Probability" },
                                     'title' :{'label' : '', 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol' : 2}}) :
        
        from sklearn.neighbors import KernelDensity
        from scipy.stats import norm
        import seaborn as sns

        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1]
 

        fig, ax = plt.subplots(self.rows, self.cols, figsize=plot_args['fig_size'], sharex=True, sharey=True)
    
        for val in range(self.nreps):
            p, q = np.unravel_index(val,(self.rows, self.cols,))


            
            sns.kdeplot(data=np.array(input['helix'][f'{s}:{val}']), label='α-Helix',ax=ax[p,q], linewidth=2,c='b')
            sns.kdeplot(data=np.array(input['sheet'][f'{s}:{val}']), label='Beta Sheet',ax=ax[p,q], linewidth=2,c='darkorange')
            sns.kdeplot(data=np.array(input['pp'][f'{s}:{val}']), label='ppII',ax=ax[p,q], linewidth=2,c='g')
            
            
            ax[p, q].set_ylim(0,0.5)
            #ax[p, q].set_xticks(range(121,142,2))
            ax[p,q].tick_params(labelsize=plot_args['tick_size']['x'])
            ax[p,q].grid(alpha=0.4)
            plt.setp(ax[p,q].get_xticklabels(), rotation=plot_args['rotation']['x'])
            
            if val == 0:
                ax[p,q].legend(loc=plot_args['legend_args']['loc'],prop={'size': plot_args['legend_args']['font_size']},ncol=plot_args['legend_args']['ncol'])

            if not demux : ax[p,q].set_title(f"{title} : {np.round(self.replica_temps[val],3)}K",size=plot_args['title']['size'])
            elif demux : ax[p,q].set_title(f"{title} : {val}",size=plot_args['title']['size'])
            
        #     text_=r'$K_{D}$'+' : '+str(round(np.array(bf_kd['rep'])[2][val],2))+r'$\pm$'+str(round(np.array(bf_kd['rep'])[3][val],2))
        #     ax[p,q].text(120.3, 0.43, text_, fontsize = 26)
            
            if not q : ax[p,q].set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
            if p == self.rows-1 : ax[p,q].set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)

        plt.tight_layout()

        if show_fig : plt.show()

        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_phipsi(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, plot_args_in:dict = {}) :

        plot_args:dict = {'fig_size': (30,10), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                           'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Alphabeta RMSD', 'y':"Probability" },
                           'title' :{'label' : '', 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol' : 2}}
         
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]

        assert self.in_files_dict['phipsi']['helix'] , f"Load the helix RMSD data computed from phipsi for the plot with keyword helix!"
        assert self.in_files_dict['phipsi']['sheet'] , f"Load the helix RMSD data computed from phipsi for the plot with keyword sheet!"
        assert self.in_files_dict['phipsi']['pp'] , f"Load the helix RMSD data computed from phipsi for the plot with keyword pp!"

        if demux : out_file = f"{self.out_files_dict['phipsi']}_demux.{self.out_file_type}"
        elif not demux : out_file = f"{self.out_files_dict['phipsi']}_rep.{self.out_file_type}"
        
        in_data={}
        for k in self.in_files_dict['phipsi'].keys():
                
            in_data[k] = load_json(f"{self.data_dir}/{self.in_files_dict['phipsi'][k]}")

        self._plot_phipsi(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args)

    def _plot_rg_2(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False, time_step_in_ps:float=80.0, convolve_step_size:int=200,
                   plot_args:dict = {'fig_size': (12,8), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                                     'label_size' : {'x' :18, 'y' : 18}, 'dpi' : 310, 'labels' : {'x' :'Replica number', 'y':"Rg (nm)" },
                                     'title' :{'label' : '', 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':2}}) :
        

        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1] 

        fig, axes = plt.subplots(1,1,figsize=plot_args['fig_size'])

        for i in range(self.nreps):

            axes.scatter(i,np.mean(np.array(input[f"{s}:{i}"])),color='black')
            axes.arrow(i, (np.mean(np.array(input[f"{s}:{i}"]))-(np.std(np.array(input[f"{s}:{i}"]))/2)),0,np.std(np.array(input[f"{s}:{i}"])))

        axes.tick_params(labelsize=plot_args['tick_size']['x'])
        
        plt.setp(axes.get_xticklabels(), rotation=plot_args['rotation']['x'])
        axes.set_xticks(range(0,self.nreps,2))
        axes.set_yticks(np.arange(0.8,1.7,0.1))

        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(2/2))
        axes.tick_params(axis='x', which='minor', length=3, width=1, color='k', direction='out')

        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1/2))
        axes.tick_params(axis='y', which='minor', length=3, width=1, color='k', direction='out')

        axes.set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
        axes.set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
        # axes.set_title(title, size=plot_args['title']['size'])
        axes.grid(alpha=0.4)

        plt.tight_layout()
        if show_fig : plt.show()
        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass

    def plot_rg_2(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, plot_args_in:dict = {}) : 

        plot_args:dict = {'fig_size': (10,8), 'rotation' : {'x' :0, 'y' :0},'tick_size' : {'x' : 25, 'y' : 25},
                          'label_size' : {'x' :25, 'y' : 25}, 'dpi' : 310, 'labels' : {'x' :'Replica number', 'y':"Rg (nm)" },
                          'title' :{'label' : '', 'size' : 28}, 'legend_args' : {'font_size' : 14, 'loc' : 1, 'ncol':2}}
        
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]


        assert self.in_files_dict['rg'] , f"Load the Rg data for the plot!"

        if demux :
            out_file = f"{self.out_files_dict['rg_hist_2']}_demux.{self.out_file_type}"

        elif not demux : 
            out_file = f"{self.out_files_dict['rg_hist_2']}_rep.{self.out_file_type}"

        in_data= load_json(f"{self.data_dir}/{self.in_files_dict['rg']}")

        self._plot_rg_2(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args)
        
    def _plot_bend_angle(self, input:dict, file_name:str=None, demux:bool=False, save_fig:bool = False, show_fig:bool = False,
                     plot_args:dict = {'fig_size': (10,8), 'rotation' : {'x' : 0, 'y' :0},'tick_size' : {'x' : 20, 'y' : 20},
                                       'label_size' : {'x' : 24, 'y' : 24}, 'title_size' :20, 'dpi' : 310, 'labels' : {'x' :"Replica Number", 'y':"Average bend angle ($^{o}$)" }}) :
        

        if not demux : s = self.s[0]; title=self.title_names[0]
        elif demux : s=self.s[1]; title=self.title_names[1] 

        fig, axes = plt.subplots(1,1,figsize=plot_args['fig_size'])

        for i in range(self.nreps):

            axes.scatter(i,np.array(input[f"{s}:{i}"])[0])
            axes.arrow(i, (np.array(input[f"{s}:{i}"])[0]-(np.array(input[f"{s}:{i}"])[1]/2)),0,np.array(input[f"{s}:{i}"])[1])

        axes.tick_params(labelsize=plot_args['tick_size']['x'])
        plt.setp(axes.get_xticklabels(), rotation=plot_args['rotation']['x'])
        axes.set_xticks(range(self.nreps))
        axes.set_yticks(range(30,160,10))
        axes.set_ylabel(plot_args['labels']['y'], size=plot_args['label_size']['y'], labelpad=15)
        axes.set_xlabel(plot_args['labels']['x'], size=plot_args['label_size']['x'], labelpad=15)
        axes.set_title(title, size=plot_args['title']['size'])
        axes.grid(alpha=0.4)

        plt.tight_layout()
        if show_fig : plt.show()
        if save_fig : assert file_name ; out_f = f"{self.out_dir}/{file_name}" ; print('saving figure!');plt.savefig(out_f, dpi=plot_args['dpi'],bbox_inches='tight')
        else : pass


    def plot_average_angle(self, save_fig:bool=False, show_fig:bool=False, demux:bool=False, plot_args_in:dict = {}) : 

        plot_args:dict = {'fig_size': (10,8), 'rotation' : {'x' : 0, 'y' :0},'tick_size' : {'x' : 20, 'y' : 20},
                           'label_size' : {'x' : 24, 'y' : 24}, 'title' :{'label' : '', 'size' : 28}, 'dpi' : 310,
                             'labels' : {'x' :"Replica Number", 'y':"Average bend angle ($^{o}$)" }}
         
        if plot_args_in :

            for k in plot_args.keys():

                if k in plot_args_in.keys() : plot_args[k] = plot_args_in[k]


        assert self.in_files_dict['ba'] , f"Load the average bend angle data for the plot!"

        if demux : out_file = f"{self.out_files_dict['ba']}_demux.{self.out_file_type}"
        elif not demux : out_file = f"{self.out_files_dict['ba']}_rep.{self.out_file_type}"

        in_data = load_json(f"{self.data_dir}/{self.in_files_dict['ba']}")

        self._plot_bend_angle(in_data, out_file, save_fig=save_fig, show_fig=show_fig, demux=demux, plot_args=plot_args)
