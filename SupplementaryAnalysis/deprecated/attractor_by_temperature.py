# -*- coding: utf-8 -*-
""""""""""""""""""""" WARNING : DEPRECATED FILE """""""""""""""""""""
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
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
from Classes.DetSim import DetSim

from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.signal_model import signal_model
from Functions.plot_phase_space_density import plot_phase_space_density

np.set_printoptions(threshold=np.nan)

#nice plotting style
sn.set_style("whitegrid",
             {'xtick.direction': 'out',
              'xtick.major.size': 6.0,
              'xtick.minor.size': 3.0,
              'ytick.color': '.15',
              'ytick.direction': 'out',
              'ytick.major.size': 6.0,
              'ytick.minor.size': 3.0})

""""""""""""""""""""" FUNCTIONS """""""""""""""""""""
def deterministic_attractor_by_temperature(cell = 'NIH3T3'):
    """
    Compute and plot the evolution of the deterministic attractor with
    temperature.

    Parameters
    ----------
    cell : string
        Cell condition.
    """

    lll_theta = []
    lll_phi = []
    l_temperature = [34,37,40]
    l_color = ['lightblue', 'grey', 'orange']
    for idx_t, temperature in enumerate(l_temperature):
        path = '../Parameters/Real/opt_parameters_div_'+str(temperature)\
                +"_"+cell+'.p'
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
        display_parameters_from_file(path, show = False)

        detSim = DetSim(l_parameters, cell, temperature)
        ll_phase_theta, ll_phase_phi = detSim.plot_trajectory(ti = 800,
                                                              tf = 1600,
                                                              rand = True,
                                                              save = False)
        lll_theta.append(ll_phase_theta)
        lll_phi.append(ll_phase_phi)

    plt.figure(figsize=(5,5))
    for idx_t, (ll_phase_theta,ll_phase_phi) in enumerate(zip(lll_theta,
                                                              lll_phi)):
        first = True
        for l_phase_theta, l_phase_phi in zip(ll_phase_theta,ll_phase_phi):
            if first:
                plt.plot(np.array(l_phase_theta)/(2*np.pi),
                        np.array(l_phase_phi)/(2*np.pi),
                        color = l_color[idx_t],
                        label = str(l_temperature[idx_t]))
                first = False
            else:
                plt.plot(np.array(l_phase_theta)/(2*np.pi),
                         np.array(l_phase_phi)/(2*np.pi),
                         color = l_color[idx_t])

    #plot identity
    x_domain = np.linspace(0,2*np.pi,100)
    f =  (lll_phi[1][2][0]-0.05 + np.linspace(0,2*np.pi,100))%(2*np.pi)
    abs_d_data_x = np.abs(np.diff(f))
    mask_x = np.hstack([abs_d_data_x > abs_d_data_x.mean()+3*abs_d_data_x.std(),
                                                                       [False]])
    masked_data_x = np.array([x if not m else np.nan for x,m in zip(f, mask_x)])


    #plt.plot(x_domain/(2*np.pi),masked_data_x/(2*np.pi) , '--'   )

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel(r'Circadian phase $\theta$')
    plt.ylabel(r'Cell-cycle phase $\phi$')
    #plt.plot([0,1],[0.22,0.22], '--', color = 'grey')
    #plt.text(x = 0.35, y = 0.14, s='G1', color = 'grey', fontsize=12)
    #plt.text(x = 0.36, y = 0.27, s='S', color = 'grey', fontsize=12)
    #plt.plot([0,1],[0.84,0.84], '--', color = 'grey')
    #plt.text(x = 0.25, y = 0.84-0.08, s='S/G2', color = 'grey', fontsize=12)
    #plt.text(x = 0.26, y = 0.84+0.05, s='M', color = 'grey', fontsize=12)


    plt.legend(loc=5)
    plt.tight_layout()
    plt.savefig('../Results/DetSilico/attractor_by_temperature.pdf')
    plt.show()
    plt.close()


def stochastic_attractor_by_temperature(cell = 'NIH3T3', nb_traces = 500,
                                        size_block = 100, expected = True):
    """
    Compute and plot the evolution of the stochastic attractor with
    temperature.

    Parameters
    ----------
    cell : string
        Cell condition.
    """
    ll_theta = []
    ll_phi = []
    l_temperature = [34,37,40]
    l_color = ['lightblue', 'grey', 'orange']
    for idx_t, temperature in enumerate(l_temperature):
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
        display_parameters_from_file(path, show = False)

        ##################### LOAD DATA ##################
        if cell == 'NIH3T3':
            path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
        else:
            path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
        dataClass=LoadData(path, nb_traces, temperature = temperature,
                            division = True, several_cell_cycles = True,
                            remove_odd_traces = True)
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, T_phi) = \
                       dataClass.load(period_phi = None, load_annotation = True)
        ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v>0] \
                                                        for l_peak in ll_peak]
        print(len(ll_signal), " traces kept")

        ##################### CREATE HIDDEN VARIABLES ##################
        theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
        l_var = [theta_var_coupled, amplitude_var, background_var]

        ##################### CREATE AND RUN HMM ##################
        hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian, ll_obs_phi,
                            waveform = W,
                            ll_nan_factor = ll_nan_circadian_factor,
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
        ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] for \
                                                               i in idx_to_keep]
        ll_idx_peak = [ll_idx_peak[i] for i in idx_to_keep]
        print("Kept traces with div: ", len(idx_to_keep))


        ##################### CROP SIGNALS FOR PLOTTING ##################
        l_first = [[it for it, obj in enumerate(l_obs_phi) if obj!=-1][0] \
                                                    for l_obs_phi in ll_obs_phi]
        l_last = [[len(l_obs_phi)-it-1 for it,obj in enumerate(l_obs_phi[::-1])\
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
        ll_idx_peak = [ [v for v in l_idx_peak if v>=first and v<=last] \
                        for l_idx_peak, first, last in \
                        zip(ll_idx_peak, l_first, l_last)]


        ##################### CREATE ll_idx_obs_phi and ll_val_phi############
        ll_idx_obs_phi = []
        for l_obs in ll_obs_phi:
            l_idx_obs_phi = []
            for obs in l_obs:
                l_idx_obs_phi.append( int(round(obs/(2*np.pi) \
                                    * len(theta_var_coupled.codomain )))\
                                    %len(theta_var_coupled.codomain )   )
            ll_idx_obs_phi.append(l_idx_obs_phi)

        if expected:
            ##################### GET EXPECTED PHASE ##################
            M_at = plot_phase_space_density(l_var, l_gamma_div, ll_idx_obs_phi,
                                            F_superimpose = F, save = False )
            l_theta = []
            l_phi = []
            for line,phi in zip(M_at.T, theta_var_coupled.domain):
                l_theta.append( np.angle(
                                np.sum(
                                np.multiply(
                                line,np.exp(1j*\
                                            np.array(theta_var_coupled.domain)
                                            ))))%(2*np.pi)
                                )
                l_phi.append(phi)
        else:
            ##################### GET MOST LIKELY PHASE ##################
            M_at = plot_phase_space_density(l_var, l_gamma_div, ll_idx_obs_phi,
                                            F_superimpose = F, save = False )
            l_theta = []
            l_phi = []
            for line,phi in zip(M_at.T, theta_var_coupled.domain):
                idx_best_phase = np.argmax(line)
                l_theta.append( theta_var_coupled.domain[idx_best_phase] )
                l_phi.append(phi)


        ######### REMOVE VERTICAL LINES AT BOUNDARIES  #########
        abs_d_data_x = np.abs(np.diff(l_theta))
        mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()\
                                                +3*abs_d_data_x.std(), [False]])
        masked_l_theta = np.array([x if not m else np.nan \
                                            for x,m in zip(l_theta, mask_x)])

        abs_d_data_x = np.abs(np.diff(l_phi))
        mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()\
                                                +3*abs_d_data_x.std(), [False]])
        masked_l_phi = np.array([x if not m else np.nan \
                                                for x,m in zip(l_phi, mask_x) ])

        ll_theta.append(masked_l_theta)
        ll_phi.append(masked_l_phi)

    ######### COMPUTE IDENDITIY  #########

    #plot identity
    x_domain = np.linspace(0,2*np.pi,100)
    f =  (1.15 + np.linspace(0,2*np.pi,100))%(2*np.pi)
    abs_d_data_x = np.abs(np.diff(f))
    mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()\
                                                +3*abs_d_data_x.std(), [False]])
    masked_identity = np.array([x if not m else np.nan \
                                                    for x, m in zip(f, mask_x)])


    ######### PLOT  #########

    plt.figure(figsize=(10,10))
    for idx_t, (l_phase_theta,l_phase_phi) in enumerate(zip(ll_theta,ll_phi)):
        plt.plot(l_phase_theta, l_phase_phi, color = l_color[idx_t],
                                            label = str(l_temperature[idx_t]))


    plt.plot(x_domain,masked_identity , '--'   )

    plt.xlim(10**-2,2*np.pi-10**-2)
    plt.ylim(10**-2,2*np.pi-10**-2)
    plt.xlabel("Circadian phase")
    plt.ylabel("Cell-cycle phase")
    plt.legend()
    plt.savefig('../Results/PhaseSpace/stochastic_attractor_by_temperature_' \
                +str(expected)+'.pdf')
    plt.show()
    plt.close()




""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    print("Deprecated file")
    #deterministic_attractor_by_temperature(cell = 'NIH3T3')
    #stochastic_attractor_by_temperature(cell = 'NIH3T3', nb_traces = 500, \
    #                                                          expected = True)
    #stochastic_attractor_by_temperature(cell = 'NIH3T3', nb_traces = 500, \
    #                                                         expected = False)
