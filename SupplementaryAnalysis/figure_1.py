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
from scipy import interpolate
import seaborn as sn
plt.style.use('seaborn-whitegrid')

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
def figure_1(cell = 'NIH3T3', temperature = 37, nb_traces = 1000,
             size_block = 100, division = True):
    """
    Create a nice plot to show data and fits.

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
    division : bool
        Plot dividing traces (True) or not (False).
    """
    path = '../Parameters/Real/opt_parameters_div_'+str(temperature)+"_"\
            +cell+'.p'
    ##################### LOAD OPTIMIZED PARAMETERS ##################
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
                      division = division, several_cell_cycles = False,
                      remove_odd_traces = False)
    (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
     ll_idx_cell_cycle_start, T_theta, T_phi)\
                    = dataClass.load(period_phi = None, load_annotation = True)

    print(len(ll_signal), " traces kept")


    ##################### CREATE HIDDEN VARIABLES ##################
    theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
    l_var = [theta_var_coupled, amplitude_var, background_var]

    ##################### CREATE AND RUN HMM ##################
    hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian, ll_obs_phi,
                        waveform = W, ll_nan_factor = ll_nan_circadian_factor,
                        pi = pi, crop = False )
    l_gamma_div, l_logP_div = hmm.run(project = False)

    #keep only long traces
    to_keep = [i for i, l_signal in enumerate(ll_signal) if len(l_signal)>100]
    ll_signal = [ll_signal[i][:100] for i in to_keep]
    ll_area = [ll_area[i][:100] for i in to_keep]
    if division:
        ll_nan_circadian_factor =[ll_nan_circadian_factor[i][:100] \
                                                              for i in to_keep]
        ll_obs_phi = [ll_obs_phi[i][:100] for i in to_keep]
        ll_peak = [ll_peak[i][:100] for i in to_keep]
        ll_idx_cell_cycle_start =  [ [j for j in ll_idx_cell_cycle_start[i] \
                                                     if j<100] for i in to_keep]
        ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v>0] \
                                                        for l_peak in ll_peak]
    else:
        ll_nan_circadian_factor =[[np.nan]*100 for i in to_keep]
        ll_obs_phi = [[np.nan]*100 for i in to_keep]
        ll_peak = [[np.nan]*100 for i in to_keep]
        ll_idx_cell_cycle_start =  [ [np.nan] for i in to_keep]
        ll_idx_peak = [[np.nan] for l_peak in ll_peak]
    l_gamma_div = [l_gamma_div[i][:100] for i in to_keep]

    ##################### PLOT #####################
    waveform = interpolate.interp1d(np.linspace(0,2*np.pi, len(W),
                                                            endpoint = True), W)
    current_palette = sn.color_palette()
    for (idx, l_signal), l_nan_circadian_factor, l_idx_peak, l_idx_cc, gamma \
                            in zip(enumerate(ll_signal),
                                             ll_nan_circadian_factor,
                                             ll_idx_peak,
                                             ll_idx_cell_cycle_start,
                                             l_gamma_div):
        if idx%2==0:
            plt.figure(figsize=(4,3))
        tspan = np.linspace(0,len(l_signal)/2, len(l_signal))
        plt.subplot(210+ idx%2+1)
        plt.plot(tspan, l_signal, '.', color = current_palette[0] \
                                            if division else current_palette[1])
        l_phase = []
        l_A = []
        l_B = []
        l_model = []
        for gamma_t in gamma:
            phase = np.angle(np.sum(np.multiply(np.sum(gamma_t, axis = (1,2)),
                      np.exp(1j*np.array(theta_var_coupled.domain)))))%(2*np.pi)
            A = np.sum(np.multiply(
                           np.sum(gamma_t, axis = (0,2)), amplitude_var.domain))
            B = np.sum(np.multiply(
                          np.sum(gamma_t, axis = (0,1)), background_var.domain))
            l_phase.append(phase)
            l_A.append(A)
            l_B.append(B)
            l_model.append(signal_model( [phase, A, B], waveform))
        #plt.plot(tspan, [phase/(2*np.pi) for phase in l_phase], '--',
                #color = 'grey')
        #plt.plot(tspan, l_model, '-', color = current_palette[2])

        for d in l_idx_cc:
            plt.axvline(d/2, color = 'black')
        plt.ylim(-0.1,1.1)
        if idx%2==1:
            plt.tight_layout()
            plt.savefig('../Results/Fits/GroupedFit_div_'+str(idx)+'_'\
                        +str(temperature)+"_"+cell+'.pdf' if division \
                        else '../Results/Fits/GroupedFit_nodiv_'+str(idx)\
                        +'_'+str(temperature)+"_"+cell+'.pdf')
            plt.show()
            plt.close()

        '''
        tspan = np.linspace(0,len(l_signal)/2, len(l_signal))
        plt.plot(tspan, l_signal, '.', color = current_palette[0] \
                                            if division else current_palette[1])
        l_phase = []
        l_A = []
        l_B = []
        l_model = []
        for gamma_t in gamma:
            phase = np.angle(np.sum(np.multiply(np.sum(gamma_t, axis = (1,2)),
                    np.exp(1j*np.array(theta_var_coupled.domain)))))%(2*np.pi)
            A = np.sum(np.multiply(np.sum(gamma_t, axis = (0,2)),
                                                        amplitude_var.domain))
            B = np.sum(np.multiply(np.sum(gamma_t, axis = (0,1)),
                                                        background_var.domain))
            l_phase.append(phase)
            l_A.append(A)
            l_B.append(B)
            l_model.append(signal_model( [phase, A, B], waveform))
        #plt.plot(tspan, [phase/(2*np.pi) for phase in l_phase], '--',
                                                                color = 'grey')
        #plt.plot(tspan, l_model, '-', color = 'black')

        for d in l_idx_cc:
            plt.axvline(d/2, color = 'black')
        plt.xlabel('Time (hours)')
        plt.ylabel(r'Revervb-$\alpha$-YFP fluorescence')
        plt.tight_layout()
        plt.savefig('../Results/Fits/Fit_div_'+str(idx)+'_'\
                                            +str(temperature)+"_"+cell+'.pdf')
        plt.show()
        plt.close()
        '''

""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':

    figure_1(cell = 'NIH3T3', temperature = 37, nb_traces = 100,
            division = True)
    #figure_1(cell = 'NIH3T3', temperature = 37, nb_traces = 20,
    #        division = False)
