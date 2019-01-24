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
import copy
from matplotlib import colors as mcolors
from mpl_toolkits import axes_grid1
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
from Functions.display_parameters import display_parameters
from Functions.make_colormap import make_colormap


np.set_printoptions(threshold=np.nan)

#nice plotting style
sn.set_style("whitegrid", {'grid.color': 'white',
            'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})
""""""""""""""""""""" FUNCTION """""""""""""""""""""
def F_by_cell_cell_cycle_period(cell = 'NIH3T3', nb_traces = 10000,
                                size_block = 100, nb_iter = 15):
    """
    Optimize and plot the coupling function and phase-space density at
    different temperatures, but for a fixed distribution of cell-cycle periods.

    Parameters
    ----------
    cell : string
        Cell condition.
    nb_traces : int
        How many traces to run the experiment on.
    size_block : integer
        Size of the traces chunks (to save memory).
    nb_iter : int
        Number of EM iterations.
    """
    temperature = None

    ##################### LOAD COLORMAP ##################
    c = mcolors.ColorConverter().to_rgb
    bwr = make_colormap(  [c('blue'), c('white'), 0.48, c('white'),
                            0.52,c('white'),  c('red')])

    ##################### LOAD OPTIMIZED PARAMETERS ##################
    path = '../Parameters/Real/opt_parameters_nodiv_'+str(temperature)\
                                                                +"_"+cell+'.p'
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
    ##################### LOAD TRACES ##################
    dic_traces = {34:{}, 37:{}, 40:{}}
    cell_cycle_space = list(range(12,40))
    for T_cell_cycle in cell_cycle_space:
        for temperature in [34,37,40]:
            ##################### LOAD DATA ##################
            if cell == 'NIH3T3':
                path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
            else:
                path = "../Data/U2OS-2017-03-20/"\
                                          +"ALL_TRACES_INFORMATION_march_2017.p"

            dataClass=LoadData(path, nb_traces, temperature = temperature,
                                division = True, several_cell_cycles = False,
                                remove_odd_traces = True)
            (ll_area_tot_flat, ll_signal_tot_flat,
            ll_nan_circadian_factor_tot_flat, ll_obs_phi_tot_flat, ll_peak,
            ll_idx_cell_cycle_start, T_theta, std_T_theta, T_phi, std_T_phi)\
                = dataClass.load(period_phi = T_cell_cycle,
                                 load_annotation = True,
                                 force_temperature = True)
            dic_traces[temperature][T_cell_cycle] = [ll_area_tot_flat,
                                            ll_signal_tot_flat,
                                            ll_nan_circadian_factor_tot_flat,
                                            ll_obs_phi_tot_flat, ll_peak,
                                            ll_idx_cell_cycle_start, T_theta,
                                            std_T_theta, T_phi, std_T_phi]
    #check that there is the same number of traces for a given temperature and
    #given cell-cycle
    for T_cell_cycle in cell_cycle_space:
        max_nb = min(len(dic_traces[34][T_cell_cycle][0]),
                    len(dic_traces[37][T_cell_cycle][0]),
                    len(dic_traces[40][T_cell_cycle][0]))
        dic_traces[34][T_cell_cycle] = [l[:max_nb] \
                if type(l)==list else l for l in dic_traces[34][T_cell_cycle]]
        dic_traces[37][T_cell_cycle] = [l[:max_nb] \
                if type(l)==list else l for l in dic_traces[37][T_cell_cycle]]
        dic_traces[40][T_cell_cycle] = [l[:max_nb] \
                if type(l)==list else l for l in dic_traces[40][T_cell_cycle]]
        print(len(dic_traces[34][T_cell_cycle][0]),
              len(dic_traces[37][T_cell_cycle][0]),
              len(dic_traces[40][T_cell_cycle][0]))

    #merge all traces for a given temperature
    for temperature in [34,37,40]:
        print("Temperature : ", temperature)
        (ll_area_tot_flat, ll_signal_tot_flat, ll_nan_circadian_factor_tot_flat,
        ll_obs_phi_tot_flat, ll_peak, ll_idx_cell_cycle_start) \
                                                        = [], [], [], [], [], []
        for T_cell_cycle in cell_cycle_space:
            ll_area_tot_flat.extend(dic_traces[temperature][T_cell_cycle][0])
            ll_signal_tot_flat.extend(dic_traces[temperature][T_cell_cycle][1])
            ll_nan_circadian_factor_tot_flat.extend(
                                       dic_traces[temperature][T_cell_cycle][2])
            ll_obs_phi_tot_flat.extend(dic_traces[temperature][T_cell_cycle][3])
            ll_peak.extend(dic_traces[temperature][T_cell_cycle][4])
            ll_idx_cell_cycle_start.extend(
                                       dic_traces[temperature][T_cell_cycle][5])

        ##################### SPECIFY F OPTIMIZATION CONDITIONS ##################

        #makes algorithm go faster
        only_F_and_pi = True
        #we don't know the inital condiion when traces divide
        pi = None
        #we start with a random empty F
        F = (np.random.rand( N_theta, N_phi)-0.5)*0.01
        #regularization
        lambda_parameter = 2*10e-6
        lambda_2_parameter = 0.005


        ##################### CORRECT INFERENCE BIAS ##################
        try:
            F_no_coupling = pickle.load(open("../Parameters/Misc/F_no_coupling_"\
                                                +str(37)+"_"+cell+'.p', "rb" ) )
            for idx_theta in range(N_theta):
                F_no_coupling[idx_theta,:] = np.mean(F_no_coupling[idx_theta,:])

        except:
            print("F_no_coupling not found, no bias correction applied")
            F_no_coupling = None


        ##################### CORRECT PARAMETERS ACCORDINGLY ##################

        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F]


        ##################### CREATE HIDDEN VARIABLES ##################
        theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)

        ##################### CREATE BLOCK OF TRACES ##################
        ll_area_tot = []
        ll_signal_tot = []
        ll_nan_circadian_factor_tot = []
        ll_obs_phi_tot = []
        first= True
        zp = enumerate(zip(ll_area_tot_flat, ll_signal_tot_flat,
                           ll_nan_circadian_factor_tot_flat,
                           ll_obs_phi_tot_flat))
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
        def buildObsPhiFromIndex(ll_index):
            ll_obs_phi = []
            for l_index in ll_index:
                l_obs_phi = []
                for index in l_index:
                    if index==-1:
                        l_obs_phi.append(-1)
                    else:
                        l_obs_phi.append(index/N_theta*2*np.pi)
                ll_obs_phi.append(l_obs_phi)
            return ll_obs_phi


        lP=0
        l_lP=[]
        l_idx_to_remove = []
        for it in range(nb_iter):
            print("Iteration :", it)
            l_jP_phase = []
            l_jP_amplitude = []
            l_jP_background = []
            l_gamma = []
            l_gamma_0 = []
            l_logP = []
            ll_signal_hmm = []
            ll_idx_phi_hmm = []
            ll_nan_circadian_factor_hmm = []



            ### CLEAN TRACES AFTER FIRST ITERATION ###
            if it==1:
                ll_signal_tot_flat = ll_signal_hmm_clean
                ll_idx_phi_tot_flat = ll_idx_phi_hmm_clean
                ll_nan_circadian_factor_tot_flat = \
                                               ll_nan_circadian_factor_hmm_clean
                print("nb traces apres 1ere iteration : ",
                                                        len(ll_signal_tot_flat))
                ll_signal_tot = []
                ll_idx_phi_tot = []
                ll_nan_circadian_factor_tot = []
                ll_obs_phi_tot = []
                first= True
                zp = zip(enumerate(ll_signal_tot_flat),
                                    ll_idx_phi_tot_flat,
                                    ll_nan_circadian_factor_tot_flat)
                for (index, l_signal), l_idx_phi, l_nan_circadian_factor in zp:
                    if index%size_block==0:
                        if not first:
                            ll_signal_tot.append(ll_signal)
                            ll_idx_phi_tot.append(ll_idx_phi)
                            ll_nan_circadian_factor_tot.append(
                                                        ll_nan_circadian_factor)
                            ll_obs_phi_tot.append(
                                            buildObsPhiFromIndex(ll_idx_phi))
                        else:
                            first = False
                        ll_signal = [l_signal]
                        ll_idx_phi = [l_idx_phi]
                        ll_nan_circadian_factor = [l_nan_circadian_factor]
                    else:
                        ll_signal.append(l_signal)
                        ll_idx_phi.append(l_idx_phi)
                        ll_nan_circadian_factor.append(l_nan_circadian_factor)

                #get remaining trace
                ll_signal_tot.append(ll_signal)
                ll_idx_phi_tot.append(ll_idx_phi)
                ll_nan_circadian_factor_tot.append(ll_nan_circadian_factor)
                ll_obs_phi_tot.append( buildObsPhiFromIndex(ll_idx_phi) )

            zp = zip(ll_signal_tot, ll_obs_phi_tot, ll_nan_circadian_factor_tot)
            for ll_signal, ll_obs_phi, ll_nan_circadian_factor in zp:

                ### INITIALIZE AND RUN HMM ###
                l_var = [theta_var_coupled, amplitude_var, background_var]
                hmm = HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian,
                                      ll_val_phi = ll_obs_phi, waveform = W ,
                                      ll_nan_factor = ll_nan_circadian_factor,
                                      pi = pi, crop = True )
                (l_gamma_0_temp, l_gamma_temp ,l_logP_temp,  ll_alpha,  ll_beta,
                l_E, ll_cnorm, ll_idx_phi_hmm_temp, ll_signal_hmm_temp,
                ll_nan_circadian_factor_hmm_temp) = hmm.run_em()

                #crop and create ll_mat_TR
                ll_signal_hmm_cropped_temp = [[s for s, idx in zip(l_s, l_idx) \
                                                                    if idx>-1]\
                                              for l_s, l_idx \
                                              in zip(ll_signal_hmm_temp,
                                                           ll_idx_phi_hmm_temp)]
                ll_idx_phi_hmm_cropped_temp = [[idx for idx in l_idx if idx>-1]\
                                              for l_idx in  ll_idx_phi_hmm_temp]
                ll_mat_TR = [np.array( [theta_var_coupled.TR[:,idx_phi,:] \
                            for idx_phi in l_idx_obs_phi]) for l_idx_obs_phi \
                            in ll_idx_phi_hmm_cropped_temp]


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
                ll_nan_circadian_factor_hmm.extend(
                                               ll_nan_circadian_factor_hmm_temp)

            t_l_jP = (l_jP_phase, l_jP_amplitude,l_jP_background)

            ### REMOVE BAD TRACES IF FIRST ITERATION###
            if it==0:
                Plim = np.percentile(l_logP, 10)
                #Plim = -10000
                for idx_trace, P in enumerate(l_logP):
                    if P<=Plim:
                        l_idx_to_remove.append(idx_trace)
                for index in sorted(l_idx_to_remove, reverse=True):
                    del t_l_jP[0][index]
                    del t_l_jP[1][index]
                    del t_l_jP[2][index]
                    del l_gamma[index]
                    del l_logP[index]
                    del l_gamma_0[index]
                    del ll_signal_hmm[index]
                    del ll_idx_phi_hmm[index]
                    del ll_nan_circadian_factor_hmm[index]
                ll_signal_hmm_clean = copy.deepcopy(ll_signal_hmm)
                ll_idx_phi_hmm_clean = copy.deepcopy(ll_idx_phi_hmm)
                ll_nan_circadian_factor_hmm_clean \
                                    = copy.deepcopy(ll_nan_circadian_factor_hmm)


            ### PARAMETERS UPDATE ###

            [F_up, pi_up, std_theta_up, sigma_em_circadian_up, ll_coef,
            std_amplitude_theta_up, std_background_theta_up, mu_amplitude_theta_up,
            mu_background_theta_up, W_up] = EM.run_EM(l_gamma_0, l_gamma,
                                      t_l_jP, theta_var_coupled, ll_idx_phi_hmm,
                                      F, ll_signal_hmm, amplitude_var,
                                      background_var, W,
                                      ll_idx_coef = F_no_coupling,
                                      only_F_and_pi = only_F_and_pi,
                                      lambd_parameter = lambda_parameter,
                                      lambd_2_parameter = lambda_2_parameter)

            if np.mean(l_logP)-lP<10**-9:
                print("diff:", np.mean(l_logP)-lP)
                print(print("average lopP:", np.mean(l_logP)))
                break
            else:
                lP = np.mean(l_logP)
                print("average lopP:", lP)
                l_lP.append(lP)

            ### CHOOSE NEW PARAMETERS ###
            F = F_up
            pi_up = pi

            l_parameters = [dt, sigma_em_circadian, W, pi,
            N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
            N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
            N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
            gamma_amplitude_theta, l_boundaries_amplitude_theta,
            N_background_theta, mu_background_theta, std_background_theta,
            gamma_background_theta, l_boundaries_background_theta,
            F]


            theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)

            ### PLOT COUPLING FUNCTION ###
            plt.pcolormesh(theta_var_coupled.domain, theta_var_coupled.codomain,
                            F.T, cmap=bwr, vmin=-0.3, vmax=0.3)
            plt.xlim([0, 2*np.pi])
            plt.ylim([0, 2*np.pi])
            plt.colorbar()
            plt.xlabel("theta")
            plt.ylabel("phi")
            plt.show()
            plt.close()


        plt.plot(l_lP)
        #plt.savefig("../Parameters/Real/opt_parameters_div_"+str(temperature)
                     #+"_"+cell+'_'+str(T_cell_cycle)+'.pdf')
        plt.show()
        plt.close()



        ##################### PLOT FINAL COUPLING ##################
        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            ###Add a vertical color bar to an image plot.###
            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            current_ax = plt.gca()
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.sca(current_ax)
            return im.axes.figure.colorbar(im, cax=cax, **kwargs)

        plt.figure(figsize=(5*1.2,5*1.2))
        im = plt.imshow(F.T, cmap=bwr, vmin=-0.3, vmax=0.3,
                        interpolation='spline16', origin='lower',
                        extent=[0, 2*np.pi,0, 2*np.pi])
        add_colorbar(im, label = r'Acceleration ($rad.h^{-1}$)')
        plt.xlabel(r'Circadian phase $\theta$')
        plt.ylabel(r'Cell-cycle phase $\phi$')
        #plt.colorbar()
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\phi$')
        plt.title('T = ' + str(temperature))
        plt.tight_layout()
        plt.savefig("../Results/PhaseSpace/Coupling_"+str(temperature)\
                    +"_"+cell+'_demoind.pdf')
        plt.close()

        ##################### PLOT PHASE SPACE DENSITY ##################
        plot_phase_space_density(l_var, l_gamma, ll_idx_phi_tot_flat,
                                 F_superimpose = None, save = True, cmap = bwr,
                                 temperature = temperature, cell = cell,
                                 period = T_cell_cycle,
                                 folder = '../Results/PhaseSpace/' )

        path = "../Results/PhaseSpace/Coupling_"+str(temperature)+"_"\
                +cell+'_demoind.p'
        with open(path, 'wb') as f:
            pickle.dump(F, f)

def load_F_by_cell_cycle(cell, nb_traces = 100, period = 22):
    """
    Load the previously computed coupling function, recompute density and
    superimpose determinstic attractor.

    Parameters
    ----------
    cell : string
        Cell condition.
    nb_traces : int
        How many traces to run the experiment on.
    period : int
        Cell-cycle period used to simulate attractor.
    """
    ##################### LOAD COLORMAP ##################
    c = mcolors.ColorConverter().to_rgb
    bwr = make_colormap(  [c('blue'), c('white'), 0.48, c('white'),
                          0.52,c('white'),  c('red')])

    ##################### DISPLAY PARAMETERS ##################
    #display_parameters_from_file(path, show = True)

    dic_traces = {34:{}, 37:{}, 40:{}}
    cell_cycle_space = list(range(12,40))
    for T_cell_cycle in cell_cycle_space:
        for temperature in [34,37,40]:
            ##################### LOAD DATA ##################
            if cell == 'NIH3T3':
                path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
            else:
                path = "../Data/U2OS-2017-03-20/"\
                        +"ALL_TRACES_INFORMATION_march_2017.p"

            dataClass=LoadData(path, nb_traces, temperature = temperature,
                               division = True, several_cell_cycles = False,
                               remove_odd_traces = True)
            (ll_area_tot_flat, ll_signal_tot_flat,
            ll_nan_circadian_factor_tot_flat, ll_obs_phi_tot_flat, ll_peak,
            ll_idx_cell_cycle_start, T_theta, std_T_theta, T_phi, std_T_phi)\
                    = dataClass.load(period_phi = T_cell_cycle,
                                    load_annotation = True,
                                    force_temperature = True)
            dic_traces[temperature][T_cell_cycle] = [ll_area_tot_flat,
                                            ll_signal_tot_flat,
                                            ll_nan_circadian_factor_tot_flat,
                                            ll_obs_phi_tot_flat, ll_peak,
                                            ll_idx_cell_cycle_start, T_theta,
                                            std_T_theta, T_phi, std_T_phi]
    #check that there is the same number of traces for a given temperature
    #and given cell-cycle
    for T_cell_cycle in cell_cycle_space:
        max_nb = min(len(dic_traces[34][T_cell_cycle][0]),
                    len(dic_traces[37][T_cell_cycle][0]),
                    len(dic_traces[40][T_cell_cycle][0]))
        dic_traces[34][T_cell_cycle] = [l[:max_nb] if type(l)==list \
                                   else l for l in dic_traces[34][T_cell_cycle]]
        dic_traces[37][T_cell_cycle] = [l[:max_nb] if type(l)==list \
                                   else l for l in dic_traces[37][T_cell_cycle]]
        dic_traces[40][T_cell_cycle] = [l[:max_nb] if type(l)==list \
                                   else l for l in dic_traces[40][T_cell_cycle]]
        print(len(dic_traces[34][T_cell_cycle][0]),
              len(dic_traces[37][T_cell_cycle][0]),
              len(dic_traces[40][T_cell_cycle][0]))

    #merge all traces for a given temperature
    for temperature in [34,37,40]:
        print("Temperature : ", temperature)
        (ll_area_tot_flat, ll_signal_tot_flat, ll_nan_circadian_factor_tot_flat,
        ll_obs_phi_tot_flat, ll_peak, ll_idx_cell_cycle_start)\
                                                        = [], [], [], [], [], []
        for T_cell_cycle in cell_cycle_space:
            ll_area_tot_flat.extend(dic_traces[temperature][T_cell_cycle][0])
            ll_signal_tot_flat.extend(dic_traces[temperature][T_cell_cycle][1])
            ll_nan_circadian_factor_tot_flat.extend(
                                       dic_traces[temperature][T_cell_cycle][2])
            ll_obs_phi_tot_flat.extend(dic_traces[temperature][T_cell_cycle][3])
            ll_peak.extend(dic_traces[temperature][T_cell_cycle][4])
            ll_idx_cell_cycle_start.extend(
                                       dic_traces[temperature][T_cell_cycle][5])


        with open("../Results/PhaseSpace/Coupling_"+str(temperature)\
                                            +"_"+cell+'_demoind.p', 'rb') as f:
            F = pickle.load(f)

        ##################### PLOT FINAL COUPLING ##################
        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            ###Add a vertical color bar to an image plot.###
            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            current_ax = plt.gca()
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.sca(current_ax)
            return im.axes.figure.colorbar(im, cax=cax, **kwargs)


        plt.figure(figsize=(5*1.2,5*1.2))
        im = plt.imshow(F.T, cmap=bwr, vmin=-0.3, vmax=0.3,
                        interpolation='spline16', origin='lower',
                        extent=[0, 1,0, 1])
        add_colorbar(im, label = r'Acceleration ($rad.h^{-1}$)')
        plt.xlabel(r'Circadian phase $\theta$')
        plt.ylabel(r'Cell-cycle phase $\phi$')
        #plt.colorbar()
        plt.title('T = ' + str(temperature))
        plt.tight_layout()
        plt.savefig("../Results/PhaseSpace/Coupling_"+str(temperature)+"_"\
                    +cell+'_demoind.pdf')
        plt.close()


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
            Ft] = pickle.load(f)
        l_parameters[-1] = F


        ##################### CREATE HIDDEN VARIABLES ##################
        theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
        l_var = [theta_var_coupled, amplitude_var, background_var]

        ##################### CREATE AND RUN HMM ##################
        hmm=HMM_SemiCoupled(l_var, ll_signal_tot_flat, sigma_em_circadian,
                            ll_obs_phi_tot_flat, waveform = W,
                            ll_nan_factor = ll_nan_circadian_factor_tot_flat,
                            pi = pi, crop = True )
        l_gamma_div, l_logP_div = hmm.run(project = False)


        ##################### REMOVE BAD TRACES #####################
        Plim = np.percentile(l_logP_div, 1)
        idx_to_keep = [i for i, logP in enumerate(l_logP_div) if logP>Plim ]
        l_gamma_div = [l_gamma_div[i] for i in idx_to_keep]
        ll_signal_tot_flat = [ll_signal_tot_flat[i] for i in idx_to_keep]
        ll_area_tot_flat = [ll_area_tot_flat[i] for i in idx_to_keep]
        ll_obs_phi_tot_flat = [ll_obs_phi_tot_flat[i] for i in idx_to_keep]
        ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] \
                                                           for i in idx_to_keep]
        print("Kept traces with div: ", len(idx_to_keep))


        ##################### CROP SIGNALS FOR PLOTTING ##################
        l_first = [[it for it, obj in enumerate(l_obs_phi) if obj!=-1][0] \
                                           for l_obs_phi in ll_obs_phi_tot_flat]
        l_last = [[len(l_obs_phi)-it-1 for it, obj \
                        in enumerate(l_obs_phi[::-1]) if obj!=-1][0] \
                        for l_obs_phi in ll_obs_phi_tot_flat]
        ll_signal = [l_signal[first:last+1] for l_signal, first, last \
                                    in zip(ll_signal_tot_flat, l_first, l_last)]
        ll_area = [l_area[first:last+1] for l_area, first, last \
                                      in zip(ll_area_tot_flat, l_first, l_last)]
        ll_obs_phi = [l_obs_phi[first:last+1] for l_obs_phi, first, last \
                                   in zip(ll_obs_phi_tot_flat, l_first, l_last)]
        ll_idx_cell_cycle_start = [[v for v in l_idx_cell_cycle_start \
                            if v>=first and v<=last  ] \
                            for l_idx_cell_cycle_start, first, last \
                            in zip(ll_idx_cell_cycle_start, l_first, l_last)]


        ##################### CREATE ll_idx_obs_phi and ll_val_phi##################
        ll_idx_obs_phi = []
        for l_obs in ll_obs_phi:
            l_idx_obs_phi = []
            for obs in l_obs:
                l_idx_obs_phi.append(int(round(obs/(2*np.pi) * \
                                     len(theta_var_coupled.codomain )))\
                                     %len(theta_var_coupled.codomain ))
            ll_idx_obs_phi.append(l_idx_obs_phi)


        ##################### COMPUTE DETERMINISTIC ATTRACTOR ##################
        #l_parameters[6] = T_theta
        if period is not None:
            l_parameters[11] = period
            l_parameters[13] = 2*np.pi/period

        #print(T_theta, T_phi)
        detSim = DetSim(l_parameters, cell, temperature)
        l_theta, l_phi = detSim.plot_trajectory(ti = 2500, tf = 3000,
                                                rand = True, save = False )
        print(T_phi)

        ##################### PLOT COUPLING AND PHASE SPACE DENSITY ############
        plot_phase_space_density(l_var, l_gamma_div, ll_idx_obs_phi,
                                 F_superimpose = F, save = True, cmap = bwr,
                                 temperature = temperature, cell = cell,
                                 period = T_phi, attractor = (l_theta, l_phi ),
                                 folder = '../Results/PhaseSpace/' )

""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    F_by_cell_cell_cycle_period(cell = 'NIH3T3', nb_traces = 100,
                                 nb_iter = 3)
    load_F_by_cell_cycle(cell = 'NIH3T3', nb_traces = 100, period = 22)
