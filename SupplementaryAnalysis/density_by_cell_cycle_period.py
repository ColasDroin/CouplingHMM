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
from Classes.HMMsim import HMMsim
from Classes.PlotStochasticSpeedSpace import PlotStochasticSpeedSpace

from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.signal_model import signal_model
from Functions.plot_phase_space_density import plot_phase_space_density

#nice plotting style
sn.set_style("whitegrid", {'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

np.set_printoptions(threshold=np.nan)
""""""""""""""""""""" FUNCTIONS """""""""""""""""""""
def density_by_cell_cycle_period(cell = 'NIH3T3', nb_traces = 500,
                                size_block = 100, stochastic = False):
    """
    Compute the phase space density for a given set of conditions, and
    superimpose either the deterministic attractor, either the stochastic one,
    one the final plot.

    Parameters
    ----------
    cell : string
        Cell condition.
    nb_traces : int
        How many traces to run the experiment on.
    size_block : integer
        Size of the traces chunks (to save memory).
    stochastic : bool
        If stochastic (True) or deterministic attractor.
    """

    temperature = 37
    #l_T_phi = [12, 15, 18, 21, 24, 27, 30, 33, 36, 39]
    l_T_phi = [21]

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

    for T_cell_cycle in l_T_phi:
        print('### T-CELL-CYCLE = ' + str(T_cell_cycle) + '###')
        ##################### LOAD DATA ##################
        if cell == 'NIH3T3':
            path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
        else:
            path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"

        dataClass=LoadData(path, nb_traces, temperature = temperature,
                            division = True, several_cell_cycles = True,
                            remove_odd_traces = True)
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, std_T_theta, T_phi, std_T_phi) = \
              dataClass.load(period_phi = T_cell_cycle, load_annotation = True)
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
        ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] for \
                                                               i in idx_to_keep]
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


        ##################### CREATE ll_idx_obs_phi and ll_val_phi##############
        ll_idx_obs_phi = []
        for l_obs in ll_obs_phi:
            l_idx_obs_phi = []
            for obs in l_obs:
                l_idx_obs_phi.append( int(round(obs/(2*np.pi) \
                * len(theta_var_coupled.codomain )))\
                % len(theta_var_coupled.codomain ))
            ll_idx_obs_phi.append(l_idx_obs_phi)

        ##################### GET PHASE DISTRIBUTION ##################
        M_at = plot_phase_space_density(l_var, l_gamma_div, ll_idx_obs_phi,
                                        F_superimpose = F, save = False )

        if stochastic:
            ##################### SIMULATE STOCHASTIC TRACES ##################
            sim = HMMsim(l_var, signal_model , sigma_em_circadian,
                        waveform = W , dt=0.5, uniform = True, T_phi = T_phi )
            ll_t_l_xi, ll_t_obs  =  sim.simulate_n_traces(nb_traces=1000,
                                                            tf = 1000)

            ### CROP BEGINNING OF THE TRACES ###
            ll_t_l_xi = [l_t_l_xi[-500:] for l_t_l_xi in ll_t_l_xi]
            ll_t_obs = [l_t_obs[-500:] for l_t_obs in ll_t_obs]
            ##################### REORDER VARIABLES ##################
            ll_obs_circadian = []
            ll_obs_nucleus = []
            lll_xi_circadian = []
            lll_xi_nucleus = []
            for idx, (l_t_l_xi, l_t_obs) in enumerate(zip(ll_t_l_xi, ll_t_obs)):
                ll_xi_circadian = [ t_l_xi[0] for t_l_xi in l_t_l_xi   ]
                ll_xi_nucleus = [ t_l_xi[1] for t_l_xi in l_t_l_xi   ]
                l_obs_circadian = np.array(l_t_obs)[:,0]
                l_obs_nucleus = np.array(l_t_obs)[:,1]
                ll_obs_circadian.append(l_obs_circadian)
                ll_obs_nucleus.append(l_obs_nucleus)
                lll_xi_circadian.append(ll_xi_circadian)
                lll_xi_nucleus.append(ll_xi_nucleus)

            w_phi = 2*np.pi/T_phi
            sim = PlotStochasticSpeedSpace((lll_xi_circadian, lll_xi_nucleus),
                                            l_var, dt,w_phi, cell,
                                            temperature, cmap = None)
            _, _, M_sim = sim.getPhaseSpace()
            M_sim_theta = np.array([l/np.sum(l) for l in M_sim]).T


            l_theta = []
            l_phi = []
            for idx_phi, phi in enumerate(theta_var_coupled.domain):
                # l_theta.append( np.angle(np.sum(
                #                         np.multiply(
                #                         M_sim_theta[idx_phi],
                #                         np.exp(1j*np.array(
                #                         theta_var_coupled.domain)))))\
                #                         %(2*np.pi) )
                l_theta.append(theta_var_coupled.domain[np.argmax(
                                                        M_sim_theta[idx_phi])])
                l_phi.append(phi)

            abs_d_data_x = np.abs(np.diff(l_theta))
            mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()\
                                                +3*abs_d_data_x.std(), [False]])
            masked_l_theta = np.array([x if not m else np.nan for x, m \
                                                    in zip(l_theta, mask_x)])

            abs_d_data_x = np.abs(np.diff(l_phi))
            mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()\
                                                +3*abs_d_data_x.std(), [False]])
            masked_l_phi = np.array([x if not m else np.nan for x, m \
                                                        in zip(l_phi, mask_x)])

            l_theta = masked_l_theta
            l_phi = masked_l_phi

            #plot M
            #plt.imshow(np.array(M_sim_theta), extent=[0, 1, 0, 1],
            #            origin='lower',  vmin=-0.0, vmax=0.1,  cmap='OrRd',
            #             alpha=0.5)
            #plt.imshow(np.array(M_sim_theta), extent=[0, 1, 0, 1],
            #            origin='lower',   cmap='OrRd')
            levels = np.arange(0.04, 0.08, 0.04)
            contours = plt.contour(np.array(M_sim_theta), levels,
                                    colors='black', origin='lower',
                                    extent=[0,1,0,1])
            plt.clabel(contours, inline=True, fontsize=8)
            plt.imshow(np.array(M_sim_theta), extent=[0, 1, 0, 1],
                        origin='lower',  vmin=-0.0, vmax=0.1,
                        cmap='RdGy', alpha=0.5)
            plt.plot(np.array(l_theta)/(2*np.pi), np.array(l_phi)/(2*np.pi),
                     lw = 2, color = 'green', alpha = 1.)
            plt.colorbar();
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.title("Phase density")
            plt.xlabel(r'Circadian phase $\theta$')
            plt.ylabel(r'Cell-cycle phase $\phi$')

            plt.legend()
            plt.savefig('../Results/PhaseSpace/generated_density_by_cell'\
                        +'_cycle_period_'+'stochastic' + '_'+str(T_phi)+'.pdf')
            plt.close()


        else:
            ##################### GET CORRESPONDING ATTRACTOR AND REPELLER ######
            detSim = DetSim(l_parameters, cell, temperature)
            ll_phase_theta, ll_phase_phi = \
                detSim.plot_trajectory( ti = 1000, tf = 1100, rand = True,
                                        save = False, K = 1, T_phi  = T_phi)

        ######### PLOT  #########

        #plt.figure(figsize=(5,5))

        #CLASSY PLOT
        levels = np.arange(0.04, 0.08, 0.04)
        contours = plt.contour(np.array(M_at.T), levels, colors='black',
                                            origin='lower' , extent=[0,1,0,1])
        plt.clabel(contours, inline=True, fontsize=8)
        plt.imshow(np.array(M_at.T), extent=[0, 1, 0, 1], origin='lower',
                                vmin=-0.0, vmax=0.1,  cmap='RdGy', alpha=0.5)
        plt.colorbar()

        #Different plot
        #plt.imshow(np.array(M_at.T), extent=[0, 1, 0, 1], origin='lower',
        #                        vmin=-0.0, vmax=0.1,  cmap='OrRd', alpha=0.5)

        #plot attractor

        # if stochastic:
        #     plt.plot(np.array(l_theta)/(2*np.pi), np.array(l_phi)/(2*np.pi),
        #             lw = 2, color = 'green', alpha = 1.)
        # else:
        #     for l_phase_theta,l_phase_phi in zip(ll_phase_theta,ll_phase_phi):
        #         plt.plot(np.array(l_phase_theta[:-1])/(2*np.pi),
        #                 np.array(l_phase_phi[:-1])/(2*np.pi), lw = 2,
        #                 color = 'green', alpha = 0.1)


        #OR plot attractor of the data
        l_theta = []
        l_phi = []
        for idx_phi, phi in enumerate(theta_var_coupled.domain):
            l_theta.append( np.angle(np.sum(np.multiply(M_at.T[idx_phi],
                            np.exp(1j*np.array(theta_var_coupled.domain)))))\
                            %(2*np.pi) )
            l_phi.append(phi)

        abs_d_data_x = np.abs(np.diff(l_theta))
        mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()\
                                                +3*abs_d_data_x.std(), [False]])
        masked_l_theta = np.array([x if not m else np.nan for x, m \
                                                    in zip(l_theta, mask_x)  ])

        abs_d_data_x = np.abs(np.diff(l_phi))
        mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()\
                                                +3*abs_d_data_x.std(), [False]])
        masked_l_phi = np.array([x if not m else np.nan \
                                            for x, m in zip(l_phi, mask_x)  ])

        l_theta = masked_l_theta
        l_phi = masked_l_phi
        plt.plot(np.array(l_theta)/(2*np.pi), np.array(l_phi)/(2*np.pi),
                    lw = 2, color = 'green', alpha = 1.)




        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title("Phase density")
        plt.xlabel(r'Circadian phase $\theta$')
        plt.ylabel(r'Cell-cycle phase $\phi$')


        plt.legend()
        if stochastic:
            plt.savefig('../Results/PhaseSpace/density_by_cell_cycle_period_'\
                        +'stochastic' + '_'+str(T_phi)+'.pdf')
        else:
            plt.savefig('../Results/PhaseSpace/density_by_cell_cycle_period_'\
                        +'deterministic' + '_'+str(T_phi)+'.pdf')
        plt.show()
        plt.close()

def generate_density(cell = 'NIH3T3', temperature = 37, T_phi = 22,
                    stochastic = False):
    """
    Generate and plot phase space density for a given set of conditions,
    either from deterministic or stochastic simulations.

    Parameters
    ----------
    cell : string
        Cell condition.
    temperature : integer
        Temperature condition
    T_phi : int
        Cell-cycle period
    stochastic : bool
        If stochastic (True) or deterministic simulation.
    """
    ##################### LOAD OPTIMIZED PARAMETERS ##################
    path = '../Parameters/Real/opt_parameters_div_'+str(temperature) \
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
        l_parameters[11] = T_phi
        l_parameters[13] = 2*np.pi/T_phi
    theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
    l_var = [theta_var_coupled, amplitude_var, background_var]

    if stochastic:

        ##################### SIMULATE STOCHASTIC TRACES ##################
        sim = HMMsim(  l_var, signal_model , sigma_em_circadian,  waveform = W,
                        dt=0.5, uniform = True, T_phi = T_phi )
        ll_t_l_xi, ll_t_obs  =  sim.simulate_n_traces(nb_traces=100, tf = 2000)

        ### CROP BEGINNING OF THE TRACES ###
        ll_t_l_xi = [l_t_l_xi[1000:] for l_t_l_xi in ll_t_l_xi]
        ll_t_obs = [l_t_obs[1000:] for l_t_obs in ll_t_obs]
        ##################### REORDER VARIABLES ##################
        ll_obs_circadian = []
        ll_obs_nucleus = []
        lll_xi_circadian = []
        lll_xi_nucleus = []
        for idx, (l_t_l_xi, l_t_obs) in enumerate(zip(ll_t_l_xi, ll_t_obs)):
            ll_xi_circadian = [ t_l_xi[0] for t_l_xi in l_t_l_xi   ]
            ll_xi_nucleus = [ t_l_xi[1] for t_l_xi in l_t_l_xi   ]
            l_obs_circadian = np.array(l_t_obs)[:,0]
            l_obs_nucleus = np.array(l_t_obs)[:,1]
            ll_obs_circadian.append(l_obs_circadian)
            ll_obs_nucleus.append(l_obs_nucleus)
            lll_xi_circadian.append(ll_xi_circadian)
            lll_xi_nucleus.append(ll_xi_nucleus)

        #print(lll_xi_nucleus)
        w_phi = 2*np.pi/T_phi
        sim = PlotStochasticSpeedSpace((lll_xi_circadian, lll_xi_nucleus),
                                        l_var, dt,w_phi, cell, temperature,
                                        cmap = None)
        _, _, M_sim = sim.getPhaseSpace()
        #M_sim_theta = np.array([l/np.sum(l) for l in M_sim]).T
        plot_phase_space_density(l_var, None, None, F_superimpose = None,
                                 save =True, cmap = 'bwr',
                                 temperature = temperature, cell = cell,
                                 period = T_phi,
                                 folder = '../Results/PhaseSpace',
                                 attractor = None, M_at = M_sim)
    else:
        detSim = DetSim(l_parameters, cell, temperature)
        ll_phase_theta, ll_phase_phi = detSim.plot_trajectory(ti = 1000,
                                                              tf = 20000,
                                                              rand = True,
                                                              save = False,
                                                              K = 1,
                                                              T_phi  = T_phi)
        M_sim = np.zeros((N_theta,N_phi))
        for l_phase_theta, l_phase_phi in zip(ll_phase_theta, ll_phase_phi):
            for theta, phi in zip(l_phase_theta, l_phase_phi):
                theta_idx = int( round((theta/(2*np.pi)*N_theta)))%(N_theta)
                phi_idx = int(  round((phi/(2*np.pi)*N_phi) ) )%(N_phi)
                M_sim[theta_idx, phi_idx] +=1
        plot_phase_space_density(l_var, None, None, F_superimpose = None,
                                 save =True, cmap = 'bwr',
                                 temperature = temperature, cell = cell,
                                 period = T_phi,
                                 folder = '../Results/PhaseSpace',
                                 attractor = None, M_at = M_sim)



""""""""""""""""""""" TEST """""""""""""""""""""
if __name__ == '__main__':
    #density_by_cell_cycle_period(cell = 'NIH3T3', nb_traces = 500,
    #                                                        stochastic = True)
    #generate_density(cell = 'NIH3T3', temperature = None, T_phi = 12,
    #                                                        stochastic = True)
    for T_phi in [16,24]:
        generate_density(cell = 'NIH3T3', temperature = None, T_phi = T_phi,
                                                            stochastic = True)
        generate_density(cell = 'NIH3T3', temperature = None, T_phi = T_phi,
                                                            stochastic = False)
