# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.stats import norm
import matplotlib.mlab as mlab
import pandas as pd
import sys
import os
import pickle
import scipy

#Import internal modules
sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

from Classes.LoadData import LoadData
from Classes.PlotResults import PlotResults
from Classes.HMM_SemiCoupled import HMM_SemiCoupled
import Classes.EM as EM

from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.signal_model import signal_model

np.set_printoptions(threshold=np.nan)
""""""""""""""""""""" FUNCTION """""""""""""""""""""
def phase_delay(cell = 'NIH3T3', temperature = 37, nb_traces = 500,
                size_block = 100):
    """
    Compute and plot how phase-delay evolves with the difference of intrinsic
    periods between the two oscillators.

    Parameters
    ----------
    cell : string
        Cell condition.
    temperature : integer
        Temperature condition.
    nb_traces : integer
        How many traces to run the experiment on.
    size_block : integer
        Size of the traces chunks (to save memory).
    """

    l_delay = []
    l_std_delay = []
    l_T = [14,16,18,20,22,24,26,28,30,32,34]
    l_real_T = []
    l_real_std_T = []
    #l_T = [20,22,24]
    for period in l_T:
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
        ll_idx_cell_cycle_start, T_theta, std_T_theta, T_phi, std_T_phi) = \
                    dataClass.load(period_phi = period, load_annotation = True)
        l_real_T.append(24-T_phi)
        l_real_std_T.append(std_T_phi)
        ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v>0] \
                                                        for l_peak in ll_peak]
        print(len(ll_signal), " traces kept")

        ##################### CREATE HIDDEN VARIABLES ##################
        theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
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
        ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] \
                                                           for i in idx_to_keep]
        ll_idx_peak = [ll_idx_peak[i] for i in idx_to_keep]
        print("Kept traces with div: ", len(idx_to_keep))


        ##################### CROP SIGNALS FOR PLOTTING ##################
        l_first = [[it for it, obj in enumerate(l_obs_phi) if obj!=-1][0] \
                                                    for l_obs_phi in ll_obs_phi]
        l_last = [[len(l_obs_phi)-it-1 for it, obj in enumerate(l_obs_phi[::-1])\
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


        ##################### COMPUTE PHASE DELAY ##################
        l_phase_delay = []
        zp = zip(enumerate(ll_signal),l_gamma_div, l_logP_div, ll_area,
                ll_obs_phi, ll_idx_cell_cycle_start, ll_idx_peak)
        for ((idx, signal), gamma, logP, area, l_obs_phi,
            l_idx_cell_cycle_start, l_idx_peak) in zp:
            plt_result = PlotResults(gamma, l_var, signal_model, signal,
                                    waveform = W, logP = logP,
                                    temperature = temperature, cell = cell)
            l_E_model, l_E_theta, l_E_A, l_E_B \
                            = plt_result.plotEverythingEsperance(False, idx)
            for phi, theta_unorm in zip(l_obs_phi, l_E_theta):
                theta = theta_unorm * 2 *np.pi
                if theta-phi>-np.pi and theta-phi<np.pi:
                    delay = theta-phi
                elif theta-phi<-np.pi:
                    delay = theta+2*np.pi-phi
                else:
                    delay = theta-phi-2*np.pi

                l_phase_delay.append( delay)
        l_delay.append(np.mean(l_phase_delay))
        l_std_delay.append(np.std(l_phase_delay))

    plt.errorbar(l_real_T, l_delay, xerr = l_real_std_T,
                                                    yerr = l_std_delay, fmt='o')
    plt.xlabel(r'$T_\theta-T_\phi$')
    plt.ylabel("<"+r'$\theta$-$\phi$'+">")
    plt.savefig('../Results/PhaseBehavior/PhaseDelay_'+str(temperature)\
                                                            +"_" + cell+'.pdf')
    plt.show()
    plt.close()



""""""""""""""""""""" TEST """""""""""""""""""""
if __name__ == '__main__':
    phase_delay(cell = 'NIH3T3', temperature = 37, nb_traces = 200)
