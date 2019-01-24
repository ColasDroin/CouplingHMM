# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sn

### Import internal modules
sys.path.insert(0, os.path.realpath('../Classes'))
from LoadData import LoadData

#nice plotting style
sn.set_style("whitegrid", { 'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

""""""""""""""""""""" FUNCTION """""""""""""""""""""
def plot_hist_periods(cell, temperature, division):
    """
    Given a cell condition, compute and plot a histgram of periods.

    Parameters
    ----------
    cell : string
        Cell condition.
    """
    ##################### LOAD DATA ##################
    if cell == 'NIH3T3':
        path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    else:
        path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
    dataClass=LoadData(path, 10000000, temperature = temperature,
                       division = division)
    (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
     ll_idx_cell_cycle_start, T_theta, T_phi) \
                                        = dataClass.load(load_annotation=True)

    if division:
        ##################### COMPUTE CELL CYCLE DISTRIBUTION ##################
        l_T_cell_cycle=[]
        for l_div_index in ll_idx_cell_cycle_start:
            for t1, t2 in zip(l_div_index[:-1], l_div_index[1:]):
                l_T_cell_cycle.append( (t2-t1)/2  )
    ##################### COMPUTE CIRCADIAN CLOCK DISTRIBUTION #################
    l_T_clock=[]
    for l_peak in ll_peak:
        l_idx_peak = [idx for idx, i in enumerate(l_peak) if i==1]
        for t_peak_1, t_peak_2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
            l_T_clock.append( (t_peak_2-t_peak_1)/2 )
    ##################### PLOT BOTH DISTRIBUTIONS ##################
    bins = np.linspace(8,38,40)
    if division:
        plt.hist(l_T_cell_cycle, bins, alpha=0.5, label='cell-cycle')
    plt.hist(l_T_clock, bins, alpha=0.5, label='clock')
    plt.legend(loc='upper right')
    if division:
        plt.savefig('../Results/RawData/Distributions_div_'+str(temperature)\
                    +"_"+cell+'.pdf', bbox_inches='tight')
    else:
        plt.savefig('../Results/RawData/Distributions_nodiv_'+str(temperature)\
                    +"_"+cell+'.pdf', bbox_inches='tight')
    plt.close()


""""""""""""" TEST """""""""""""
if __name__ == '__main__':
    plot_hist_periods(cell = "NIH3T3", temperature = 37, division = True)
    plot_hist_periods(cell = "NIH3T3", temperature = 34, division = True)
    plot_hist_periods(cell = "NIH3T3", temperature = 40, division = True)
    plot_hist_periods(cell = "NIH3T3", temperature = 37, division = False)
    plot_hist_periods(cell = "NIH3T3", temperature = 34, division = False)
    plot_hist_periods(cell = "NIH3T3", temperature = 40, division = False)
    plot_hist_periods(cell = "U2OS", temperature = 37, division = True)
