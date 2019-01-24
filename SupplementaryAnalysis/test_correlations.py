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
import seaborn as sn

#Import internal modules
sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

from Classes.LoadData import LoadData
from Classes.PlotResults import PlotResults
from Classes.HMM_SemiCoupled import HMM_SemiCoupled

from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.signal_model import signal_model

#nice plotting style
sn.set_style("whitegrid", {
            'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

np.set_printoptions(threshold=np.nan)
""""""""""""""""""""" FUNCTION """""""""""""""""""""
def test_correlations(cell = 'NIH3T3', temperature = 37, nb_traces = 500):
    """
    Compute and plot the correlation between the model variabels : phase, noise,
    amplitude and background.

    Parameters
    ----------
    cell : string
        Cell condition.
    temperature : integer
        Temperature condition.
    nb_traces : integer
        How many traces to run the experiment on.
    """
    ##################### LOAD OPTIMIZED PARAMETERS ##################
    path = '../Parameters/Real/opt_parameters_div_'+str(temperature)\
                                                                +"_"+cell+'.p'
    with open(path, 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)

        '''
        gamma = 0.005
        std_amplitude_theta = std_amplitude_theta \
                                            * (gamma/gamma_amplitude_theta)**0.5
        std_background_theta = std_background_theta \
                                           * (gamma/gamma_background_theta)**0.5
        gamma_amplitude_theta = gamma
        gamma_background_theta = gamma


        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F]
        '''
    ##################### DISPLAY PARAMETERS ##################
    display_parameters_from_file(path, show = True)

    ##################### LOAD DATA ##################
    if cell == 'NIH3T3':
        path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    else:
        path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
    dataClass=LoadData(path, nb_traces, temperature = temperature,
                        division = True, several_cell_cycles = False,
                        remove_odd_traces = True)
    (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
    ll_idx_cell_cycle_start, T_theta, T_phi) \
                                    = dataClass.load(load_annotation = True)
    ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v>0] \
                                                        for l_peak in ll_peak]
    print(len(ll_signal), " traces kept")

    ##################### CREATE HIDDEN VARIABLES ##################
    theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )
    l_var = [theta_var_coupled, amplitude_var, background_var]

    ##################### CREATE AND RUN HMM ##################
    hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian, ll_obs_phi,
                        waveform = W, ll_nan_factor = ll_nan_circadian_factor,
                        pi = pi, crop = True )
    l_gamma_div, l_logP_div = hmm.run(project = False)

    ##################### REMOVE BAD TRACES #####################
    Plim = np.percentile(l_logP_div, 10)
    idx_to_keep = [i for i, logP in enumerate(l_logP_div) if logP>Plim ]
    l_gamma_div = [l_gamma_div[i] for i in idx_to_keep]
    ll_signal = [ll_signal[i] for i in idx_to_keep]
    ll_area = [ll_area[i] for i in idx_to_keep]
    l_logP_div = [l_logP_div[i] for i in idx_to_keep]
    ll_obs_phi = [ll_obs_phi[i] for i in idx_to_keep]
    ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] for i in idx_to_keep]
    ll_idx_peak = [ll_idx_peak[i] for i in idx_to_keep]
    print("Kept traces with div: ", len(idx_to_keep))


    ##################### CROP SIGNALS FOR PLOTTING ##################

    l_first = [[it for it, obj in enumerate(l_obs_phi) \
                                    if obj!=-1][0] for l_obs_phi in ll_obs_phi]
    l_last = [[len(l_obs_phi)-it-1 for it, obj in enumerate(l_obs_phi[::-1]) \
                                    if obj!=-1][0] for l_obs_phi in ll_obs_phi]
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
            l_idx_obs_phi.append( int(round(obs/(2*np.pi)\
                                *len(theta_var_coupled.codomain )))\
                                %len(theta_var_coupled.codomain))
        ll_idx_obs_phi.append(l_idx_obs_phi)


    ##################### COMPUTE EXPECTED VALUE OF THE SIGNAL #################
    ll_model = []
    ll_phase = []
    ll_amplitude = []
    ll_background = []
    zp = zip(enumerate(ll_signal),l_gamma_div, l_logP_div, ll_area, ll_obs_phi,
            ll_idx_cell_cycle_start, ll_idx_peak)
    for ((idx, signal), gamma, logP, area, l_obs_phi,
        l_idx_cell_cycle_start, l_idx_peak) in zp:
        plt_result = PlotResults(gamma, l_var, signal_model, signal,
                                waveform = W, logP = logP,
                                temperature = temperature, cell = cell)
        E_model, E_theta, E_A, E_B \
                            = plt_result.plotEverythingEsperance(False, idx)
        ll_model.append(E_model)
        ll_phase.append(E_theta)
        ll_amplitude.append(E_A)
        ll_background.append(E_B)


    ##################### COMPUTE AND PLOT RESIDUALS ##################
    domain = list(range(N_theta))
    dic_phase_res = {}
    for phase in domain:
        dic_phase_res[phase] = []
    for (idx, l_signal), l_E_model, l_E_phase, in zip(enumerate(ll_signal),
                                                        ll_model, ll_phase):
        for signal, E_model, E_phase in zip(l_signal, l_E_model, l_E_phase):
            phase = int(round(E_phase*N_theta))
            if phase == N_theta:
                phase = 0
            dic_phase_res[phase].append(E_model-signal)


    l_mean = []
    l_std = []
    for phase in domain:
        l_mean.append(np.mean( dic_phase_res[phase]))
        l_std.append( np.std( dic_phase_res[phase]) )

    plt.close()
    plt.errorbar(np.array(domain)/len(domain), l_mean, yerr = l_std, fmt='o')
    plt.xlabel(r"Circadian phase $\theta$")
    plt.ylabel(r"Residuals  $S_t-O_t$")
    plt.tight_layout()
    plt.savefig('../Results/Correlation/Residuals_'+cell+'_'\
                                                    +str(temperature)+'.pdf')
    plt.show()
    plt.close()


    ##################### COMPUTE AND PLOT PHASE/AMP CORRELATION ###############
    dic_phase_amp = {}
    for phase in domain:
        dic_phase_amp[phase] = []
    for (idx, l_signal), l_E_amp, l_E_phase, in zip(enumerate(ll_signal),
                                                    ll_amplitude, ll_phase):
        for signal, E_amp, E_phase in zip(l_signal, l_E_amp, l_E_phase):
            phase = int(round(E_phase*N_theta))
            if phase == N_theta:
                phase = 0
            dic_phase_amp[phase].append(E_amp)


    l_mean = []
    l_std = []
    for phase in domain:
        l_mean.append(np.mean( dic_phase_amp[phase]))
        l_std.append( np.std( dic_phase_amp[phase]) )

    plt.errorbar(np.array(domain)/len(domain), l_mean, yerr = l_std, fmt='o')
    plt.xlabel(r"Circadian phase $\theta$")
    plt.ylabel(r"Amplitude Process $A_t$")
    plt.tight_layout()
    plt.savefig('../Results/Correlation/Amplitude_'+cell+'_'\
                                                       +str(temperature)+'.pdf')
    plt.show()
    plt.close()


    ##################### COMPUTE AND PLOT PHASE/BACK CORRELATION ##############
    dic_phase_bac = {}
    for phase in domain:
        dic_phase_bac[phase] = []
    for (idx, l_signal), l_E_bac, l_E_phase, in zip(enumerate(ll_signal),
                                                    ll_background, ll_phase):
        for signal, E_bac, E_phase in zip(l_signal, l_E_bac, l_E_phase):
            phase = int(round(E_phase*N_theta))
            if phase == N_theta:
                phase = 0
            dic_phase_bac[phase].append(E_bac)


    l_mean = []
    l_std = []
    for phase in domain:
        l_mean.append(np.mean( dic_phase_bac[phase]))
        l_std.append( np.std( dic_phase_bac[phase]) )

    plt.errorbar(np.array(domain)/len(domain), l_mean, yerr = l_std, fmt='o')
    plt.xlabel(r"Circadian phase $\theta$")
    plt.ylabel(r"Background Process $B_t$")
    plt.tight_layout()
    plt.savefig('../Results/Correlation/Background_'+cell+'_'\
                                                      +str(temperature)+'.pdf')
    plt.show()
    plt.close()


""""""""""""""""""""" TEST """""""""""""""""""""
if __name__ == '__main__':
    test_correlations(cell = 'NIH3T3', temperature = None, nb_traces = 200)
