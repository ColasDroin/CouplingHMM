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
import Classes.EM as EM

from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.signal_model import signal_model

np.set_printoptions(threshold=np.nan)

#nice plotting style
sn.set_style("whitegrid", {
            'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

""""""""""""""""""""" FUNCTION """""""""""""""""""""
def test_sigma_theta(cell = 'NIH3T3', temperature = 37, nb_traces = 500,
                    size_block = 100):
    """
    Compute how the diffusion coefficient (inferred) evolves with the phase.

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
    ##################### LOAD OPTIMIZED PARAMETERS ##################
    path = '../Parameters/Real/opt_parameters_div_'+str(temperature)+"_"\
                                                                    +cell+'.p'
    with open(path , 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)


    ##################### DISPLAY PARAMETERS ##################
    display_parameters_from_file(path, show = False)

    ##################### LOAD DATA ##################
    if cell == 'NIH3T3':
        path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    else:
        path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"

    dataClass=LoadData(path, nb_traces, temperature = None, division = True,
                        several_cell_cycles = False, remove_odd_traces = True)
    (ll_area_tot_flat, ll_signal_tot_flat, ll_nan_circadian_factor_tot_flat,
    ll_obs_phi_tot_flat, T_theta, T_phi) = dataClass.load()
    print(len(ll_signal_tot_flat), " traces kept")

    ##################### SPECIFY F OPTIMIZATION CONDITIONS ##################
    #makes algorithm go faster
    only_F_and_pi = True

    ##################### CREATE HIDDEN VARIABLES ##################
    theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters)

    ##################### CREATE BLOCK OF TRACES ##################
    ll_area_tot = []
    ll_signal_tot = []
    ll_nan_circadian_factor_tot = []
    ll_obs_phi_tot = []
    first= True
    zp = enumerate(zip(ll_area_tot_flat, ll_signal_tot_flat,
                        ll_nan_circadian_factor_tot_flat, ll_obs_phi_tot_flat))
    for index, (l_area, l_signal, l_nan_circadian_factor, l_obs_phi) in zp:
        if index%size_block==0:
            if not first:
                ll_area_tot.append(ll_area)
                ll_signal_tot.append(ll_signal)
                ll_nan_circadian_factor_tot.append(ll_nan_circadian_factor)
                ll_obs_phi_tot.append(ll_obs_phi)

            else:
                first = False
            ll_area = [l_area]
            ll_signal = [l_signal]
            ll_nan_circadian_factor = [l_nan_circadian_factor]
            ll_obs_phi = [l_obs_phi]
        else:
            ll_area.append(l_area)
            ll_signal.append(l_signal)
            ll_nan_circadian_factor.append(l_nan_circadian_factor)
            ll_obs_phi.append(l_obs_phi)
    #get remaining trace
    ll_area_tot.append(ll_area)
    ll_signal_tot.append(ll_signal)
    ll_nan_circadian_factor_tot.append(ll_nan_circadian_factor)
    ll_obs_phi_tot.append(ll_obs_phi)



    ##################### OPTIMIZATION ##################

    l_jP_phase = []
    l_jP_amplitude = []
    l_jP_background = []
    l_gamma = []
    l_gamma_0 = []
    l_logP = []
    ll_signal_hmm = []
    ll_idx_phi_hmm = []
    ll_nan_circadian_factor_hmm = []


    for ll_signal, ll_obs_phi, ll_nan_circadian_factor in zip(ll_signal_tot,
                                                ll_obs_phi_tot,
                                                ll_nan_circadian_factor_tot):

        ### INITIALIZE AND RUN HMM ###
        l_var = [theta_var_coupled, amplitude_var, background_var]
        hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian,
                            ll_val_phi = ll_obs_phi, waveform = W ,
                            ll_nan_factor = ll_nan_circadian_factor,
                            pi = pi, crop = True )
        (l_gamma_0_temp, l_gamma_temp ,l_logP_temp,  ll_alpha,  ll_beta, l_E,
        ll_cnorm, ll_idx_phi_hmm_temp, ll_signal_hmm_temp,
        ll_nan_circadian_factor_hmm_temp) = hmm.run_em()

        #crop and create ll_mat_TR
        ll_signal_hmm_cropped_temp =[[s for s, idx in zip(l_s, l_idx) if idx>-1]\
                for l_s, l_idx in  zip(ll_signal_hmm_temp ,ll_idx_phi_hmm_temp)]
        ll_idx_phi_hmm_cropped_temp = [ [idx for idx in l_idx if idx>-1] \
                                            for l_idx in  ll_idx_phi_hmm_temp  ]
        ll_mat_TR = [np.array( [theta_var_coupled.TR[:,idx_phi,:] for idx_phi \
            in l_idx_obs_phi]) for l_idx_obs_phi in ll_idx_phi_hmm_cropped_temp]


        ### PLOT TRACE EXAMPLE ###
        zp = zip(enumerate(ll_signal_hmm_cropped_temp),l_gamma_temp,l_logP_temp)
        for (idx, signal), gamma, logP in zp:
            plt_result = PlotResults(gamma, l_var, signal_model, signal,
                                    waveform = W, logP = None,
                                    temperature = temperature, cell = cell)
            plt_result.plotEverythingEsperance(False, idx)
            if idx==0:
                break


        l_jP_phase_temp, l_jP_amplitude_temp,l_jP_background_temp \
            = EM.compute_jP_by_block(ll_alpha, l_E, ll_beta, ll_mat_TR,
                                    amplitude_var.TR, background_var.TR,
                                    N_theta, N_amplitude_theta,
                                    N_background_theta, only_F_and_pi)
        l_jP_phase.extend(l_jP_phase_temp)
        l_jP_amplitude.extend(l_jP_amplitude_temp)
        l_jP_background.extend(l_jP_background_temp)
        l_gamma.extend(l_gamma_temp)
        l_logP.extend(l_logP_temp)
        l_gamma_0.extend(l_gamma_0_temp)
        ll_signal_hmm.extend(ll_signal_hmm_temp)
        ll_idx_phi_hmm.extend(ll_idx_phi_hmm_temp)
        ll_nan_circadian_factor_hmm.extend(ll_nan_circadian_factor_hmm_temp)

    ##################### COMPUTE SIGMA(THETA) ##################
    l_mean = []
    l_std = []
    dic_phase = {}
    theta_domain = theta_var_coupled.domain
    for idx_theta in range(N_theta):
        dic_phase[idx_theta] = []
    for jP, ll_idx_obs_phi_trace in zip(l_jP_phase, ll_idx_phi_hmm):
        for t,jPt in enumerate(jP):
            for idx_theta_i, theta_i in enumerate(theta_domain):
                norm = np.sum(jPt[idx_theta_i])
                if norm==0:
                    continue
                jPt_i_norm = jPt[idx_theta_i]/norm

                theta_dest = theta_i + (w_theta+F[idx_theta_i,
                                                ll_idx_obs_phi_trace[t]])*dt

                var = 0
                for idx_theta_k, theta_k in enumerate(theta_domain):
                    var+= (min( abs( theta_dest - theta_k ),
                          abs( theta_dest - (theta_k+2*np.pi) ) ,
                          abs( theta_dest - (theta_k-2*np.pi) ) )**2) \
                                                    *jPt_i_norm[idx_theta_k]
                dic_phase[idx_theta_i].append((var, norm))


    for idx_theta in range(N_theta):
        p_theta = np.sum( [ tupl[1] for tupl in dic_phase[idx_theta]])
        sigma_theta = np.sum( [ (p_theta_t * sigma_theta_t)/p_theta \
                    for (p_theta_t, sigma_theta_t) in dic_phase[idx_theta] ])
        var_sigma_theta =  np.sum( [ (sigma_theta_t-sigma_theta)**2*p_theta_t \
                                                                      /p_theta \
                     for (p_theta_t, sigma_theta_t) in dic_phase[idx_theta] ])
        l_mean.append(sigma_theta)
        l_std.append(var_sigma_theta**0.5 )

    plt.errorbar(theta_domain/(2*np.pi), l_mean, yerr = l_std, fmt='o')
    plt.xlabel(r"Circadian phase $\theta$")
    plt.ylabel(r"Phase diffusion SD[$\theta$]")
    plt.tight_layout()
    plt.savefig('../Results/Correlation/Diffusion_'+cell+'_'\
                +str(temperature)+'.pdf')
    plt.show()
    plt.close()

""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    test_sigma_theta(cell = 'NIH3T3', temperature = 37, nb_traces = 200)
