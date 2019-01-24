# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import pandas as pd
import sys
import os
import pickle
import scipy
import matplotlib.colors as mcolors
import seaborn as sn

#Import internal modules
sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

from Classes.LoadData import LoadData
from Classes.PlotResults import PlotResults
from Classes.HMM_SemiCoupled import HMM_SemiCoupled
import Classes.EM as EM
from Classes.DetSim import DetSim

from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.signal_model import signal_model
from Functions.make_colormap import make_colormap
from Functions.plot_phase_space_density import plot_phase_space_density

#nice plotting style
sn.set_style("whitegrid", {'grid.color': 'white',
            'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

np.set_printoptions(threshold=np.nan)
""""""""""""""""""""" FUNCTION """""""""""""""""""""
def compute_phase_for_phase_delay(path = None, temperature = None):
    """
    Compute phase of cells treated with various cell-cycle inhibitors and
    write results in a text file to be analyzed with R.

    Parameters
    ----------
    path : string
        Data path (if no temperature is given)
    temperature : integer
        Temperature condition (if no path is given)
    """
    if path is None and temperature is None:
        print('A PATH OR A TEMPERATURE MUST BE SPECIFIED')
    ##################### LOAD OPTIMIZED PARAMETERS ##################
    path_par = '../Parameters/Real/opt_parameters_div_'+str(temperature)\
                                                                    +'_NIH3T3.p'
    with open(path_par, 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)

    ##################### LOAD COLORMAP ##################
    c = mcolors.ColorConverter().to_rgb
    bwr = make_colormap(  [c('blue'), c('white'), 0.48, c('white'),
                         0.52,c('white'),  c('red')])

    ##################### DISPLAY PARAMETERS ##################
    display_parameters_from_file(path_par, show = True)

    ##################### LOAD DATA ##################
    if temperature is not None:
        path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
        dataClass=LoadData(path, 100000, temperature = temperature,
                          division = True, several_cell_cycles = False,
                          remove_odd_traces = True)
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, T_phi)\
                    = dataClass.load(period_phi = None, load_annotation = True)
        ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v>0] \
                                                        for l_peak in ll_peak]
        print(len(ll_signal), " traces kept")
        l_idx_trace = list(range(len(ll_area)))

    else:
        from os import listdir
        from os.path import isfile, join
        l_files = [f for f in listdir(path) if isfile(join(path, f))]

        ll_t = []
        ll_signal = []
        ll_area = []
        ll_peak = []
        ll_neb = []
        ll_cc_start = []
        ll_nan_circadian_factor = []
        ll_idx_cell_cycle_start =[]
        l_idx_trace = []

        for file in l_files:
            l_t = []
            l_signal = []
            l_area = []
            l_peak = []
            l_neb = []
            l_cc_start = []
            with open(path+'/'+file, 'r') as f:
                for idx, line in enumerate(f):
                    if idx==0:  continue
                    t, signal, area, peak, cc_start, neb = line.split()
                    l_signal.append(float(signal))
                    l_area.append(float(area))
                    l_t.append(float(t))
                    l_peak.append(int(peak))
                    l_cc_start.append(int(cc_start))
                    l_neb.append(int(neb))

                l_signal = np.array(l_signal) - np.percentile(l_signal, 5)
                l_signal = l_signal/np.percentile(l_signal, 95)

            garbage, idx_trace, garbage = file.split('.')
            if np.sum(l_cc_start)>=2:
                l_idx_trace.append(int(idx_trace))
                ll_t.append(l_t)
                ll_signal.append(l_signal)
                ll_area.append(l_area)
                ll_peak.append(l_peak)
                ll_neb.append(l_neb)
                ll_cc_start.append(l_cc_start)
                ll_peak.append(l_peak)


        ### NaN FOR CIRCADIAN SIGNAL ###
        for l_mitosis_start, l_cell_start, l_peak in zip(ll_neb, ll_cc_start,
                                                                    ll_peak):
            l_temp = [False]*len(l_mitosis_start)
            NaN = False
            for ind, (m, c) in enumerate(zip(l_mitosis_start, l_cell_start)):
                if m==1:
                    NaN = True
                if c==1:
                    NaN = False
                if NaN:
                    try:
                        l_temp[ind-1] = True #TO BE REMOVED POTENTIALLY
                    except:
                        pass
                    try:
                        l_temp[ind+1] = True
                    except:
                        pass
                    l_temp[ind] = True
            ll_nan_circadian_factor.append(  l_temp   )


        ### COMPUTE IDX PHI ###
        zp = zip(enumerate(ll_neb), ll_cc_start)
        for (idx, l_mitosis_start), l_cell_cycle_start in zp :
            l_idx_cell_cycle_start = \
                       [idx for idx, i in enumerate(l_cell_cycle_start) if i==1]
            ll_idx_cell_cycle_start.append(l_idx_cell_cycle_start)

        ### GET PHI OBS ###
        ll_obs_phi = []
        for l_signal, l_idx_cell_cycle_start in zip(ll_signal,
                                                    ll_idx_cell_cycle_start):
            l_obs_phi = [-1]*l_idx_cell_cycle_start[0]
            first = True
            for idx_div_1, idx_div_2 in zip(l_idx_cell_cycle_start[:-1],
                                                    l_idx_cell_cycle_start[1:]):
                if not first:
                    del l_obs_phi[-1]
                l_obs_phi.extend([i%(2*np.pi) for i \
                            in np.linspace(0,2*np.pi,idx_div_2-idx_div_1+1)])
                first = False

            l_obs_phi.extend( [-1] * \
                                  (len(l_signal)-l_idx_cell_cycle_start[-1]-1))
            ll_obs_phi.append(l_obs_phi)

        ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v>0] \
                                                          for l_peak in ll_peak]




    ##################### CREATE HIDDEN VARIABLES ##################
    theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
    l_var = [theta_var_coupled, amplitude_var, background_var]

    ##################### CREATE AND RUN HMM ##################
    hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian, ll_obs_phi,
                        waveform = W, ll_nan_factor = ll_nan_circadian_factor,
                        pi = pi, crop = True )
    l_gamma_div, l_logP_div = hmm.run(project = False)


    ##################### CROP SIGNALS FOR PLOTTING ##################
    l_first = [[it for it, obj in enumerate(l_obs_phi) \
                                    if obj!=-1][0] for l_obs_phi in ll_obs_phi]
    l_last = [[len(l_obs_phi)-it-1 for it, obj \
        in enumerate(l_obs_phi[::-1]) if obj!=-1][0] for l_obs_phi in ll_obs_phi]
    ll_signal = [l_signal[first:last+1] for l_signal, first, last \
                                            in zip(ll_signal, l_first, l_last)]
    ll_area = [l_area[first:last+1] for l_area, first, last \
                                            in zip(ll_area, l_first, l_last)]
    ll_obs_phi = [l_obs_phi[first:last+1] for l_obs_phi, first, last \
                                            in zip(ll_obs_phi, l_first, l_last)]
    ll_idx_cell_cycle_start = [ [v for v in l_idx_cell_cycle_start \
        if v>=first and v<=last  ] for l_idx_cell_cycle_start, first, last \
        in zip(ll_idx_cell_cycle_start, l_first, l_last)]
    ll_idx_peak = [ [v for v in l_idx_peak if v>=first and v<=last  ] \
            for l_idx_peak, first, last in zip(ll_idx_peak, l_first, l_last)]


    ##################### CREATE ll_idx_obs_phi and ll_val_phi##################
    ll_idx_obs_phi = []
    for l_obs in ll_obs_phi:
        l_idx_obs_phi = []
        for obs in l_obs:
            l_idx_obs_phi.append( int(round(obs/(2*np.pi) * \
                                len(theta_var_coupled.codomain )))\
                                %len(theta_var_coupled.codomain )   )
        ll_idx_obs_phi.append(l_idx_obs_phi)


    ##################### PLOT FITS ##################
    ll_esp_theta = []
    for ((idx, signal), gamma, logP, area, l_obs_phi, l_idx_cell_cycle_start,
            l_idx_peak) in zip(enumerate(ll_signal),l_gamma_div, l_logP_div,
            ll_area, ll_obs_phi, ll_idx_cell_cycle_start, ll_idx_peak):
        plt_result = PlotResults(gamma, l_var, signal_model, signal,
                                waveform = W, logP = logP,
                                temperature = temperature, cell = 'NIH3T3')
        try:
            E_model, E_theta, E_A, E_B \
                                = plt_result.plotEverythingEsperance(False, idx)
            ll_esp_theta.append(E_theta)
        except:
            ll_esp_theta.append([-1]*len(signal))
            ('Trace ' + str(idx) +' skipped !')


    ##################### RECORD INFERRED PHASES IN A TXT FILE ##################
    longest_trace = np.max([len(trace) for trace in ll_signal])
    print([len(trace) for trace in ll_signal])
    print(longest_trace)
    if temperature is None:
        appendix = path.split('/')[-1]
    else:
        appendix = str(temperature)
    with open('../Results/RawPhases/traces_theta_inferred_'\
                                            +appendix+'.txt', 'w') as f_theta:
        with open('../Results/RawPhases/traces_phi_inferred_'\
                                                +appendix+'.txt', 'w') as f_phi:
            zp = zip(l_idx_trace, ll_signal, ll_esp_theta, ll_obs_phi)
            for idx, signal, l_esp_theta, l_obs_phi in zp:
                f_theta.write(str(idx))
                f_phi.write(str(idx))
                for s in np.concatenate(
                                (l_esp_theta,[-1]*(longest_trace-len(signal))) ):
                    if s!=-1:
                        f_theta.write('\t'+str(s*2*np.pi))
                    else:
                        f_theta.write('\t'+str(s))
                for s in np.concatenate(
                                (l_obs_phi,[-1]*(longest_trace-len(signal))) ):
                    f_phi.write('\t'+str(s))
                f_theta.write('\n')
                f_phi.write('\n')



def simulated_phase_for_phase_delay():
    """
    Compute phase of simulated traces with various T_phi/T_theta ratios, to
    study how phase delay evolves with period ratio.
    Write results in a text file to be analyzed with R.
    """
    path_par = '../Parameters/Real/opt_parameters_div_None_NIH3T3.p'
    with open(path_par, 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)

    ll_theta = []
    ll_phi = []
    space_period = list(range(8,70))
    for T_phi in space_period:
        detSim = DetSim(l_parameters, cell = 'NIH3T3', temperature = None)
        ll_phase_theta, ll_phase_phi \
            = detSim.plot_trajectory( ti = 1000, tf = 1100, rand = True,
                                    save = False, K = 1, T_phi  = T_phi)
        l_theta = [x for array in ll_phase_theta for x in array]
        l_phi = [x for array in ll_phase_phi for x in array]
        ll_theta.append(l_theta)
        ll_phi.append(l_phi)

    #print(ll_theta)
    ##################### RECORD SIMULATED PHASES IN A TXT FILE ###############"
    longest_trace = np.max([len(trace) for trace in ll_theta])
    with open('../Results/RawPhases/'\
                      +'traces_theta_simulated_per_period.txt', 'w') as f_theta:
        with open('../Results/RawPhases/'\
                        +'traces_phi_simulated_per_period.txt', 'w') as f_phi:
            for T_phi, l_theta, l_phi in zip(space_period, ll_theta, ll_phi):
                f_theta.write(str(T_phi))
                f_phi.write(str(T_phi))

                for s in np.concatenate(
                                (l_theta,[-1]*(longest_trace-len(l_theta))) ):
                    f_theta.write('\t'+str(s) )
                for s in np.concatenate(
                                    (l_phi,[-1]*(longest_trace-len(l_phi))) ):
                    f_phi.write('\t'+str(s))
                f_theta.write('\n')
                f_phi.write('\n')

def make_figure_from_text_files():
    """
    Plot results of the analysis made with R.
    """
    with open('../Results/RawPhases/forColas.perturb.x.y.se.txt', 'r') as f:
        l_x, l_y, l_sex, l_sey, l_IDPerturb, l_DivFrac = [],[],[],[],[],[]
        for idx, line in enumerate(f):
            if idx==0:
                pass
            else:
                x,y,sex,sey,IDPerturb, DivFrac = line.split('\t')
                l_x.append(float(x))
                l_y.append(float(y))
                l_sex.append(float(sex))
                l_sey.append(float(sey))
                l_IDPerturb.append(IDPerturb)
                l_DivFrac.append(float(DivFrac))
    with open('../Results/RawPhases/'\
                        +'forColas.perturb.simulations.lines.txt', 'r') as f:
        l_x1, l_y1, l_x2, l_y2, l_x3, l_y3 = [],[],[],[],[],[]
        for idx, line in enumerate(f):
            if idx==0:
                pass
            else:
                x1,y1,x2,y2,x3, y3 = line.split('\t')
                l_x1.append(float(x1))
                l_y1.append(float(y1))
                l_x2.append(float(x2))
                l_y2.append(float(y2))
                l_x3.append(float(x3))
                l_y3.append(float(y3))
    #make the plot
    l_c = sn.color_palette("husl", 12)
    plt.figure(figsize=(6,5))
    plt.axhspan(0, 0.34, alpha=0.1, color='red')
    plt.axhspan(0.34, 0.65, alpha=0.1, color='green')
    plt.axhspan(0.65, 1, alpha=0.1, color='blue')
    plt.text(1.2,0.7,r'$\phi = 0$')
    plt.text(1.2,0.4,r'$\phi = 4\pi/3$')
    plt.text(1.2,0.1,r'$\phi = 2\pi/3$')
    idx = 0
    for x,y, sex, sey,IDPerturb in zip(l_x, l_y, l_sex, l_sey, l_IDPerturb):
        if idx<12:
            plt.errorbar(x, y, xerr=sex, yerr = sey, color = l_c[idx%12],
                        fmt='o', label = IDPerturb)
        else:
            plt.errorbar(x, y, xerr=sex, yerr = sey, color = l_c[idx%12],
                        fmt='o')
        idx+=1
        #plt.text(x*1.01,y*1.01, IDPerturb)
    plt.plot(l_x1, l_y1, color = 'grey', ls='--', label = 'Simulation')
    plt.plot(l_x2, l_y2, color = 'grey', ls='--' )
    plt.plot(l_x3, l_y3, color = 'grey', ls='--' )



    #plt.xlim([0.7,1.4])
    plt.ylim([0.,1.])
    plt.legend(loc=(1.04,0.3))
    plt.xlabel(r'Period ratio $T_{\phi}/T_{\theta}$')
    plt.ylabel(r'Circadian phase $\theta$')
    plt.tight_layout()
    plt.savefig('../Results/RawPhases/fig.pdf')
    plt.show()
    plt.close()



""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    #compute_phase_for_phase_delay(path = "../Data/NIH3T3-CDKin-2018-05-02",
    #                                temperature = None)
    #compute_phase_for_phase_delay(path = "../Data/NIH3T3-CDKin2-2018-05-02",
    #                                temperature = None)
    #compute_phase_for_phase_delay(path = "../Data/NIH3T3-shCRY-2018-05-02",
    #                                temperature = None)
    #compute_phase_for_phase_delay(path = None, temperature = 34)
    #compute_phase_for_phase_delay(path = None, temperature = 37)
    #compute_phase_for_phase_delay(path = None, temperature = 40)
    #simulated_phase_for_phase_delay()
    #make_figure_from_text_files()
    pass
