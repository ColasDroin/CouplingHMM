# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import pickle
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

def compute_periods(cell = 'NIH3T3'):
    """
    Given a cell condition, compute and plot a boxplot with all periods
    distributions for all temperature conditions, in dividing and non-dividing
    conditions.

    Parameters
    ----------
    cell : string
        Cell condition.
    """

    if cell == 'NIH3T3':
        path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    else:
        path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"

    nb_traces = 10000000

    t_T=(34, 37, 40)
    l_T_full = []
    for T in t_T:

        ### COMPUTE LIST CIRCADIAN PERIOD ###
        division = True
        dataClass=LoadData(path, nb_traces, temperature = T, division = division)
        (ll_area, ll_signal, ll_peak, ll_cell_cycle_start,
                                    ll_mitosis_start) = dataClass.filter_data()

        l_T = []
        for l_peak in ll_peak:
            l_idx_peak = [idx for idx, i in enumerate(l_peak) if i==1]
            for t_peak_1, t_peak_2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
                #print(t_peak_1, t_peak_2)
                l_T.append( (t_peak_2-t_peak_1)/2)
        l_T_full.append(l_T)

        division = False
        dataClass=LoadData(path, nb_traces,temperature = T, division = division)
        (ll_area, ll_signal, ll_peak, ll_cell_cycle_start,
                                    ll_mitosis_start) = dataClass.filter_data()

        l_T = []
        for l_peak in ll_peak:
            l_idx_peak = [idx for idx, i in enumerate(l_peak) if i==1]
            for t_peak_1, t_peak_2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
                #print(t_peak_1, t_peak_2)
                l_T.append( (t_peak_2-t_peak_1)/2)
        l_T_full.append(l_T)

        ### COMPUTE LIST CELL-CYCLE PERIOD ###
        division = True
        ll_idx_cell_cycle_start =[]
        dataClass=LoadData(path, nb_traces, temperature = T, division = division)
        (ll_area, ll_signal, ll_peak, ll_cell_cycle_start,
                                    ll_mitosis_start) = dataClass.filter_data()
        zp = zip(enumerate(ll_mitosis_start), ll_cell_cycle_start)
        for (idx, l_mitosis_start), l_cell_cycle_start in zp :
            l_idx_mitosis_start = [idx for idx, i in \
                                            enumerate(l_mitosis_start) if i==1]
            l_idx_cell_cycle_start = [idx for idx, i in \
                                         enumerate(l_cell_cycle_start) if i==1]
            ll_idx_cell_cycle_start.append(l_idx_cell_cycle_start)
        l_T2 = []
        for l_idx_cell_cycle_start in ll_idx_cell_cycle_start:
            for t_start_1, t_start_2 in zip(l_idx_cell_cycle_start[:-1],
                                            l_idx_cell_cycle_start[1:]):
                #print(t_peak_1, t_peak_2)
                l_T2.append( (t_start_2-t_start_1)/2)
        l_T_full.append(l_T2)

    #plt.style.use('default')
    l_label_full = [r'$T_\theta^D$,34°C', r'$T_\theta^{ND}$,34°C',
                    r'$T_\phi^D$,34°C', r'$T_\theta^D$,37°C',
                    r'$T_\theta^{ND}$,37°C', r'$T_\phi^D$,37°C',
                    r'$T_\theta^D$,40°C', r'$T_\theta^{ND}$,40°C',
                    r'$T_\phi^D$,40°C']
    # Create a figure instance
    fig = plt.figure(1, figsize=(10, 5))
    # Create an axes instance
    ax = fig.add_subplot(111)

    ## add patch_artist=True option to ax.boxplot()
    ## to get fill color
    bp = ax.boxplot(l_T_full, showmeans=False, patch_artist=True)
    #ax.axhline(22, ls = '--',color = 'yellow', label = '22h')
    #ax.axhline(24, ls = '--', color = 'orange', label = '24h')
    ax.legend()
    ## change outline color, fill color and linewidth of the boxes
    for i, box in enumerate(bp['boxes']):
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        if (i+1)%3==0:
            # change fill color
            box.set( facecolor = '#1b9e77' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)


    ax.set_xticklabels(l_label_full)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylabel('Hours')

    if not os.path.exists("../Results/RawData"):
        os.makedirs("../Results/RawData")

    fig.savefig('../Results/RawData/Periods.pdf', bbox_inches='tight')
    fig.show()
    plt.close()
def compute_periods_simple(cell = 'NIH3T3'):
    """
    Given a cell condition, compute and plot a boxplot with all periods
    distributions for all temperature conditions, in dividing conditions.

    Parameters
    ----------
    cell : string
        Cell condition.
    """
    if cell == 'NIH3T3':
        path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    else:
        path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"

    nb_traces = 10000000

    t_T=(34, 37, 40)
    l_T_full = []
    for T in t_T:

        ### COMPUTE LIST CELL-CYCLE PERIOD ###
        division = True
        ll_idx_cell_cycle_start =[]
        dataClass=LoadData(path, nb_traces, temperature = T, division = division)
        (ll_area, ll_signal, ll_peak, ll_cell_cycle_start,
                                    ll_mitosis_start) = dataClass.filter_data()
        zp = zip(enumerate(ll_mitosis_start), ll_cell_cycle_start)
        for (idx, l_mitosis_start), l_cell_cycle_start in zp:
            l_idx_mitosis_start = [idx for idx, i in \
                                             enumerate(l_mitosis_start) if i==1]
            l_idx_cell_cycle_start = [idx for idx, i in \
                                          enumerate(l_cell_cycle_start) if i==1]
            ll_idx_cell_cycle_start.append(l_idx_cell_cycle_start)
        l_T2 = []
        for l_idx_cell_cycle_start in ll_idx_cell_cycle_start:
            for t_start_1, t_start_2 in zip(l_idx_cell_cycle_start[:-1],
                                            l_idx_cell_cycle_start[1:]):
                #print(t_peak_1, t_peak_2)
                l_T2.append( (t_start_2-t_start_1)/2)
        l_T_full.append(l_T2)

    #plt.style.use('default')
    l_label_full = [ r'34°C',r'37°C', r'$40°C']
    # Create a figure instance
    fig = plt.figure(1, figsize=(5, 5))
    # Create an axes instance
    ax = fig.add_subplot(111)

    ## add patch_artist=True option to ax.boxplot()
    ## to get fill color
    bp = ax.boxplot(l_T_full, showmeans=False, patch_artist=True)
    #ax.axhline(22, ls = '--',color = 'yellow', label = '22h')
    ax.axhline(24, ls = '--', color = 'orange', label = '24h')
    ax.legend()


    ## change outline color, fill color and linewidth of the boxes
    for i, box in enumerate(bp['boxes']):
        # change outline color
        box.set( color='#7570b3', linewidth=2)


    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)


    ax.set_xticklabels(l_label_full)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylabel('Cell-cycle period (h)')

    if not os.path.exists("../Results/RawData"):
        os.makedirs("../Results/RawData")

    fig.savefig('../Results/RawData/Periods_simple.pdf', bbox_inches='tight')
    fig.show()
    plt.close()
""""""""""""""""""""" TEST """""""""""""""""""""


if __name__ == '__main__':
    compute_periods()
    compute_periods_simple()
