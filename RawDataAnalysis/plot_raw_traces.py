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
def plot_raw_traces(cell, temperature, division, nb_traces = 10):
    ##################### LOAD DATA ##################
    if cell == 'NIH3T3':
        path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    else:
        path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
    dataClass=LoadData(path, nb_traces, temperature = temperature,
                        division = division, several_cell_cycles = False)
    (ll_area, ll_signal, ll_nan_circadian_factor,
    ll_obs_phi, ll_peak, ll_idx_cell_cycle_start,
    T_theta, T_phi) = dataClass.load(load_annotation=True)

    if division:

        #keep only long traces and crop them
        to_keep = [i for i, l_signal in enumerate(ll_signal)\
                                                           if len(l_signal)>100]
        ll_signal = [ll_signal[i][:100] for i in to_keep]
        ll_area = [ll_area[i][:100] for i in to_keep]
        ll_nan_circadian_factor = \
                            [ll_nan_circadian_factor[i][:100] for i in to_keep]
        ll_obs_phi = [ll_obs_phi[i][:100] for i in to_keep]
        ll_peak = [ll_peak[i][:100] for i in to_keep]
        ll_idx_cell_cycle_start =  [ [j for j in ll_idx_cell_cycle_start[i] \
                                                    if j<100] for i in to_keep]

        zp = zip(enumerate(ll_signal), ll_nan_circadian_factor, ll_peak,
                ll_idx_cell_cycle_start)
        for (idx, l_signal), l_nan_circadian_factor, l_peak, l_idx_cc in zp:
            tspan = np.linspace(0,len(l_signal)/2, len(l_signal))
            l_idx_peak = [idx for idx, i in enumerate(l_peak) if i>0]
            l_t_nan_circadian = []
            i_temp = None


            for idxx, (i1, i2) in enumerate(zip(l_nan_circadian_factor[:-1],
                                                l_nan_circadian_factor[1:] )):
                if not i1 and i2:
                    i_temp = (idxx+1)/2
                if i1 and i2:
                    pass
                if i1 and not i2:
                    if i_temp == None:
                        print('BUG')
                    else:
                        l_t_nan_circadian.append((i_temp, idxx/2))
                        i_temp = None


            plt.subplot(210+ idx%2+1 )
            plt.plot(tspan, l_signal, '.', color = 'red')
            for p in l_idx_peak:
                #plt.axvline(p/2)
                pass
            for d in l_idx_cc:
                plt.axvline(d/2, color = 'black')
            for a,b in l_t_nan_circadian:
                #plt.axvspan(a, b, color='lightblue', alpha=0.5, lw=0)
                pass
            if idx%2==1:
                plt.tight_layout()
                plt.savefig('../Results/RawData/RawTrace_div_'+str(idx)+'_'\
                            +str(temperature)+"_"+cell+'.pdf')
                plt.show()
                plt.close()

    else:
        #keep only long traces and crop them so they all have the same size
        #(CAUTION, CODE FOR PEAK NOT CORRECT)
        ll_signal = [l_signal[:100] for l_signal in ll_signal \
                                                          if len(l_signal)>100]
        #ll_peak= [l_peak[:100] for l_signal, l_peak in zip(ll_signal, ll_peak)\
                                                        #if len(l_signal)>100]
        for (idx, l_signal), l_peak, in zip(enumerate(ll_signal), ll_peak):
            tspan = np.linspace(0,len(l_signal)/2, len(l_signal))
            l_idx_peak = [idx for idx, i in enumerate(l_peak) if i>0]

            plt.subplot(210+ idx%2+1 )
            plt.plot(tspan, l_signal, '.', color = 'orange')
            for p in l_idx_peak:
                #plt.axvline(p/2)
                pass

            if idx%2==1:
                plt.tight_layout()
                plt.savefig('../Results/RawData/RawTrace_nodiv_'+str(idx)+'_'\
                            +str(temperature)+"_"+cell+'.pdf')
                plt.show()
                plt.close()


""""""""""""" TEST """""""""""""
if __name__ == '__main__':
    plot_raw_traces(cell = "NIH3T3", temperature = 37, division = False,
                    nb_traces = 100)
