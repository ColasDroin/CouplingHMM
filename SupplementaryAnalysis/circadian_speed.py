# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np
import scipy.stats as st
import matplotlib
matplotlib.use('Agg') #to run the script on a distant server
matplotlib.rcParams['pdf.fonttype'] = 42
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

sn.set_style("whitegrid", {'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

np.set_printoptions(threshold=np.nan)

""""""""""""""""""""" FUNCTION """""""""""""""""""""
def circadian_speed(cell = 'NIH3T3', temperature = 37, nb_traces = 500,
                    size_block = 100, T_phi = None):
    """
    Compute and plot the circadian speed in function of the circadian phase,
    both in the data and in simulations.

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
    if T_phi is not None and temperature is not None:
        temperature = None
        print('Since a T_phi was specified, temperature was set to None')
    #CAUTION, None temperature used
    path = '../Parameters/Real/opt_parameters_div_'+str(None)+"_"+cell+'.p'
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
                       division = True, several_cell_cycles = True,
                       remove_odd_traces = False,
                       several_circadian_cycles = False)
    try:
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta,
        _,T_phi,_) = dataClass.load(period_phi = T_phi, load_annotation = True)
    except:
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi,
        ll_peak, ll_idx_cell_cycle_start, T_theta,
        T_phi) = dataClass.load(period_phi = T_phi, load_annotation = True)
    ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v>0] for \
                                                            l_peak in ll_peak]
    print(len(ll_signal), " traces kept")

    ##################### CREATE HIDDEN VARIABLES ##################
    theta_var_coupled, amplitude_var, background_var = \
                        create_hidden_variables(l_parameters = l_parameters )
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
    l_first = [[it for it, obj in enumerate(l_obs_phi) if obj!=-1][0] \
                                                    for l_obs_phi in ll_obs_phi]
    l_last = [[len(l_obs_phi)-it-1 for it, obj in enumerate(l_obs_phi[::-1]) \
                                    if obj!=-1][0] for l_obs_phi in ll_obs_phi]
    ll_signal = [l_signal[first:last+1] for l_signal, first, last \
                                            in zip(ll_signal, l_first, l_last)]
    ll_area = [l_area[first:last+1] for l_area, first, last \
                                             in zip(ll_area, l_first, l_last)]
    ll_obs_phi = [l_obs_phi[first:last+1] for \
                    l_obs_phi, first, last in zip(ll_obs_phi, l_first, l_last)]
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


    ##################### COMPUTE CIRCADIAN SPEED ##################
    dic_circadian_speed = {}
    for idx_theta in range(N_theta):
        dic_circadian_speed[idx_theta] = []
    print(len(ll_signal), len(l_gamma_div), len(l_logP_div), len(ll_area),
            len(ll_obs_phi), len(ll_idx_cell_cycle_start), len(ll_idx_peak))
    for ((idx, signal), gamma, logP, area, l_obs_phi, l_idx_cell_cycle_start,
        l_idx_peak) in zip(enumerate(ll_signal),l_gamma_div, l_logP_div,
                            ll_area, ll_obs_phi, ll_idx_cell_cycle_start,
                            ll_idx_peak):
        plt_result = PlotResults(gamma, l_var, signal_model, signal,
                                waveform = W, logP = logP,
                                temperature = temperature, cell = cell)
        l_E_model, l_E_theta, l_E_A, l_E_B = \
                                plt_result.plotEverythingEsperance( False, idx)
        for theta_1_norm, theta_2_norm in zip(l_E_theta[0:-1], l_E_theta[1:]):
            theta_1 = theta_1_norm * 2*np.pi
            theta_2 = theta_2_norm * 2*np.pi
            if theta_2-theta_1>-np.pi and theta_2-theta_1<np.pi:
                speed = (theta_2-theta_1)/0.5
            elif theta_2-theta_1<-np.pi:
                speed = (theta_2+2*np.pi-theta_1)/0.5
            else:
                speed = (theta_2-theta_1-2*np.pi)/0.5
            if speed<-1. or speed>1.5: #bug with polar coordinates
                print(speed)
                continue
            idx_theta = int(round(theta_1/(2*np.pi) * \
                        len(theta_var_coupled.codomain )))\
                        %len(theta_var_coupled.codomain )
            dic_circadian_speed[idx_theta].append(speed)
    ##################### PLOT SPEED VS PHASE ##################
    l_mean_speed = []
    l_std_speed = []
    for idx_theta in range(N_theta):
        l_mean_speed.append(np.mean(dic_circadian_speed[idx_theta]))
        #l_std_speed.append(np.std(dic_circadian_speed[idx_theta]) \
                    #if np.std(dic_circadian_speed[idx_theta])<0.15 else np.nan)
        l_std_speed.append(np.std(dic_circadian_speed[idx_theta]) )



    plt.figure(figsize=(5,5))
    #plt.errorbar(theta_var_coupled.codomain, l_mean_speed, yerr = l_std_speed,
                    #fmt='o', label = 'Data estimation')
    plt.plot(theta_var_coupled.codomain/(2*np.pi), l_mean_speed,
            color = 'blue', label = 'Data estimation')
    plt.fill_between(theta_var_coupled.codomain/(2*np.pi),
                     np.array(l_mean_speed)+l_std_speed,
                     np.array(l_mean_speed)-l_std_speed,
                     facecolor='lightblue', alpha=0.5)
    plt.xlabel(r'Circadian phase $\theta$')
    plt.ylabel(r'Circadian speed ($rad.h^{-1})$')
    plt.ylim([0.15, 0.45])
    plt.xlim([0., 1])

    if T_phi is not None:
        ########### COMPUTE CIRCADIAN SPEED ON DETERMINISTIC ATTRACTOR ########
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, T_phi, l_boundaries_phi, 2*np.pi/T_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F]

        detSim = DetSim(l_parameters, cell, temperature)
        tspan, vect_Y = detSim.simulate(tf=10000, full_simulation = False,
                                        rand = False)

        ##################### PLOT SPEED VS PHASE ##################
        dic_circadian_speed_det = {}
        for idx_theta in range(N_theta):
            dic_circadian_speed_det[idx_theta] = []

        for theta_1, theta_2 in zip(vect_Y[5000:-1,0], vect_Y[5001:,0]):
            if theta_2-theta_1>-np.pi and theta_2-theta_1<np.pi:
                speed = (theta_2-theta_1)/0.5
            elif theta_2-theta_1<-np.pi:
                speed = (theta_2+2*np.pi-theta_1)/0.5
            else:
                speed = (theta_2-theta_1-2*np.pi)/0.5
            idx_theta = int(round(theta_1/(2*np.pi) * \
                        len(theta_var_coupled.codomain )))\
                        %len(theta_var_coupled.codomain )
            dic_circadian_speed_det[idx_theta].append(speed)

        l_mean_speed_det = []
        l_std_speed_det = []
        for idx_theta in range(N_theta):
            l_mean_speed_det.append(np.mean(dic_circadian_speed_det[idx_theta]))
            l_std_speed_det.append(np.std(dic_circadian_speed_det[idx_theta]))

        plt.plot(theta_var_coupled.codomain/(2*np.pi), l_mean_speed_det, \
                 label = 'Deterministic simulation')
        plt.fill_between(theta_var_coupled.codomain/(2*np.pi),
                        np.array(l_mean_speed_det)+l_std_speed_det,
                        np.array(l_mean_speed_det)-l_std_speed_det,
                        facecolor='orange', alpha=0.5)
        plt.plot([0,1],[2*np.pi/24,2*np.pi/24], '--', color = 'grey',
                 label = r'$\frac{2\pi}{24}$')
        plt.title(r'$T_\phi=$'+str(int(round(T_phi)))+'h')
        plt.legend()
    plt.tight_layout()
    plt.savefig('../Results/PhaseBehavior/CircadianSpeed_'+cell+'_'\
                +str(temperature)+'_'+str(T_phi)+'.pdf')
    plt.show()
    plt.close()



""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    #circadian_speed(cell = 'NIH3T3', temperature = 34, nb_traces = 5000)
    #circadian_speed(cell = 'NIH3T3', temperature = 37, nb_traces = 5000)
    #circadian_speed(cell = 'NIH3T3', temperature = 40, nb_traces = 200)
    for T_phi in range(17,29,4):
        circadian_speed(cell = 'NIH3T3', temperature = None,
                        nb_traces = 200, T_phi = T_phi)
    #circadian_speed(cell = 'NIH3T3', temperature = None, nb_traces = 100000,
                    #T_phi = 22)
