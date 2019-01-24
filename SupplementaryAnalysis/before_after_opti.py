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
def before_after_opti(cell = 'NIH3T3', temperature = 37, nb_traces = 500,
                     size_block = 100):
    """
    Plot a trace fit before and after optimization of the parameters.

    Parameters
    ----------
    cell : string
        Cell condition.
    temperature : integer
        Temperature condition.
    nb_traces : integer
        Number of traces from which the inference is made.
    size_block : integer
        Size of the traces chunks (to save memory).
    """

    path = 'Parameters/Real/opt_parameters_div_'+str(temperature)+"_"+cell+'.p'
    """"""""""""""""""""" LOAD OPTIMIZED PARAMETERS """""""""""""""""
    with open(path, 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)

    """"""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
    display_parameters_from_file(path, show = False)

    """"""""""""""""""""" LOAD DATA """""""""""""""""
    if cell == 'NIH3T3':
        path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    else:
        path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
    dataClass=LoadData(path, nb_traces, temperature = temperature,
                        division = True, several_cell_cycles = False,
                        remove_odd_traces = True)
    (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
    ll_idx_cell_cycle_start, T_theta, T_phi) \
                    = dataClass.load(period_phi = None, load_annotation = True)
    ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v>0] for \
                                                            l_peak in ll_peak]
    print(len(ll_signal), " traces kept")

    """"""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""
    theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
    l_var = [theta_var_coupled, amplitude_var, background_var]

    """"""""""""""""""""" CREATE AND RUN HMM WITH OPTI """""""""""""""""
    hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian, ll_obs_phi,
                        waveform = W, ll_nan_factor = ll_nan_circadian_factor,
                        pi = pi, crop = True )
    l_gamma_div_1, l_logP_div_1 = hmm.run(project = False)


    """"""""""""""""""""" SET F TO 0 """""""""""""""""
    l_parameters[-1] = np.zeros((N_theta, N_phi))

    """"""""""""""""""""" RECREATE HIDDEN VARIABLES """""""""""""""""
    theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
    l_var = [theta_var_coupled, amplitude_var, background_var]

    """"""""""""""""""""" CREATE AND RUN HMM WITHOUT OPTI """""""""""""""""
    hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian, ll_obs_phi,
                        waveform = W, ll_nan_factor = ll_nan_circadian_factor,
                        pi = pi, crop = True )
    l_gamma_div_2, l_logP_div_2 = hmm.run(project = False)


    """"""""""""""""""""" CROP SIGNALS FOR PLOTTING """""""""""""""""
    l_first = [[it for it, obj in enumerate(l_obs_phi) if obj!=-1][0] \
                                                    for l_obs_phi in ll_obs_phi]
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




    """"""""""""""""""""" PLOT FITS WITH """""""""""""""""
    idx2 = 0
    vmax = 0
    imax = 0
    zp = zip(enumerate(ll_signal), ll_area, ll_obs_phi,
            ll_idx_cell_cycle_start, ll_idx_peak)
    for (idx, signal), area, l_obs_phi, l_idx_cell_cycle_start,l_idx_peak in zp:
        plt_result = PlotResults(l_gamma_div_2[idx], l_var, signal_model,signal,
                                waveform = W, logP = l_logP_div_2[idx],
                                temperature = temperature, cell = cell)
        E_model1, E_theta, E_A, E_B = plt_result.plotEverythingEsperance( True,
                                                                          idx2)
        plt_result = PlotResults(l_gamma_div_1[idx], l_var, signal_model,
                                signal, waveform = W, logP = l_logP_div_1[idx],
                                temperature = temperature, cell = cell)
        E_model2, E_theta, E_A, E_B = plt_result.plotEverythingEsperance(True,
                                                                        idx2+1)
        vcur = np.sum(np.abs(np.array(E_model1)\
                                            -np.array(E_model2)))/len(E_model1)
        if vcur>vmax:
            vmax = vcur
            imax = idx
        idx2+=2
    print(vmax, imax)


""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    wd = os.getcwd()[:-21]
    os.chdir(wd)
    #print(wd)
    before_after_opti(cell = 'NIH3T3', temperature = 37, nb_traces = 500)
