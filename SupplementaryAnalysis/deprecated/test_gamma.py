# -*- coding: utf-8 -*-
""""""""""""""""""""" WARNING : DEPRECATED FILE """""""""""""""""""""
""" This file was intially used to compute the impact of the gamma parameter
on the resulting inference."""
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
import matplotlib.colors as mcolors


#Import internal modules
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))
sys.path.insert(0, os.path.realpath('..'))


from Classes.LoadData import LoadData
from Classes.PlotResults import PlotResults
from Classes.HMM_SemiCoupled import HMM_SemiCoupled
import Classes.EM as EM

from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.signal_model import signal_model
from Functions.plot_phase_space_density import plot_phase_space_density
from Functions.make_colormap import make_colormap

from RawDataAnalysis.estimate_OU_par import estimate_OU_par


np.set_printoptions(threshold=np.nan)
""""""""""""""""""""" FUNCTION """""""""""""""""""""
def test_gamma(cell = 'NIH3T3', temperature = 37, nb_trace = 500,
                size_block = 100, nb_iter = 15):

    ##################### LOAD COLORMAP ##################
    c = mcolors.ColorConverter().to_rgb
    bwr = (  [c('blue'), c('white'), 0.48, c('white'),
                0.52,c('white'),  c('red')])


    ### GUESS OR ESTIMATE PARAMETERS ON NON-DIVIDING TRACES ###
    wd = os.getcwd()[:-21]
    os.chdir(wd)

    #remove potential previous results
    path = 'Parameters/Real/init_parameters_nodiv_'+str(temperature)\
            +"_"+cell+".p"
    if os.path.isfile(path) :
        os.remove(path)
    #run script
    print("1_wrap_initial_parameters launched")
    os.system("python 1_wrap_initial_parameters.py " + cell + " "\
                + str(temperature) + \
                ' > TextOutputs/1_wrap_initial_parameters_'+cell\
                +'_'+str(temperature)+'.txt')
    #check if final result has been created
    if os.path.isfile(path) :
        print("Initial set of parameters successfully created")
    else:
        print("BUG, initial set of parameters not created")


    for gamma_reg in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
        print("gamma =", gamma_reg, " started ")
        ### REPLACE GAMMA AND STD OU  AND RECORD AS OPTIMIZED PARAMETERS NODIV###
        gamma_A = gamma_reg
        gamma_B = gamma_reg
        mu_A, std_A, mu_B, std_B = estimate_OU_par(cell, temperature, \
                                        gamma_A = gamma_A, gamma_B = gamma_B)

        path = 'Parameters/Real/init_parameters_nodiv_'+str(temperature)\
                                                                +"_"+cell+".p"
        with open(path, 'rb') as f:
            [dt, sigma_em_circadian, W, pi,
            N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
            N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
            N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
            gamma_amplitude_theta, l_boundaries_amplitude_theta,
            N_background_theta, mu_background_theta, std_background_theta,
            gamma_background_theta, l_boundaries_background_theta,
            F] = pickle.load(f)

        gamma_amplitude_theta = gamma_A
        gamma_background_theta = gamma_B
        std_amplitude_theta = std_A
        std_background_theta = std_B

        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F]

        ##################### WRAP PARAMETERS ##################
        path = "Parameters/Real/opt_parameters_nodiv_"+str(temperature)\
                                                                +"_"+cell+".p"
        pickle.dump( l_parameters, open( path, "wb" ) )


        ### COMPUTE WAVEFORM BIAS ###
        #remove potential previous results
        path = "Parameters/Misc/F_no_coupling_"+str(temperature)+"_"+cell+'.p'
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("7_bias launched")
        os.system("python 7_validate_inference_dividing.py " + str(nb_iter) \
                    + " "+ str(nb_trace) + " "+  str(size_block) + " "+cell \
                    + " "+ str(temperature) + " True"+ ' > TextOutputs/7_bias_'\
                    +cell+'_'+str(temperature)+'.txt')
        #check if final result has been created
        if os.path.isfile(path) :
            print("Bias successfully computed")
        else:
            print("BUG, bias not computed")

        ### OPTIMIZE COUPLING ON DIVIDING TRACES ###
        #remove potential previous results
        path = "Parameters/Real/opt_parameters_div_"+str(temperature)+"_"\
                                                                    +cell+".p"
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("4_optimize_parameters_dividing launched")
        os.system("python 4_optimize_parameters_dividing.py " + str(nb_iter) \
                    + " "+ str(nb_trace) + " "+  str(size_block) + " "+cell \
                    + " "+ str(temperature)\
                    + ' > TextOutputs/4_optimize_parameters_dividing_'+cell\
                    +'_'+str(temperature)+'.txt')
         #check if final result has been created
        if os.path.isfile(path) :
            print("Optimized set of parameters on dividing traces"\
                                                    +" successfully created")
        else:
            print("BUG, optimized set of parameters on"\
                                                +" dividing traces not created")


        ### COMPLUTE/PLOT COUPLING AND PHASE DENSITY ###
        ##################### LOAD OPTIMIZED PARAMETERS ##################
        path = 'Parameters/Real/opt_parameters_div_'+str(temperature)\
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
        period = None

        ##################### LOAD DATA ##################
        if cell == 'NIH3T3':
            path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
        else:
            path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
        dataClass=LoadData(path, nb_trace, temperature = temperature,
                            division = True, several_cell_cycles = False,
                            remove_odd_traces = True)
        if period is None:
            (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
            ll_idx_cell_cycle_start, T_theta, T_phi) \
                = dataClass.load(period_phi = period, load_annotation = True)
        else:
            (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
            ll_idx_cell_cycle_start, T_theta, std_T_theta, T_phi, std_T_phi) \
                = dataClass.load(period_phi = period, load_annotation = True)

        ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v>0] for \
                                                            l_peak in ll_peak]
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
        ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] \
                                                        for i in idx_to_keep]
        ll_idx_peak = [ll_idx_peak[i] for i in idx_to_keep]
        print("Kept traces with div: ", len(idx_to_keep))


        ##################### CROP SIGNALS FOR PLOTTING ##################
        l_first = [[it for it, obj in enumerate(l_obs_phi) \
                    if obj!=-1][0] for l_obs_phi in ll_obs_phi]
        l_last = [[len(l_obs_phi)-it-1 for it, obj \
                    in enumerate(l_obs_phi[::-1]) if obj!=-1][0] \
                    for l_obs_phi in ll_obs_phi]
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


        ##################### CREATE ll_idx_obs_phi and ll_val_phi###############
        ll_idx_obs_phi = []
        for l_obs in ll_obs_phi:
            l_idx_obs_phi = []
            for obs in l_obs:
                l_idx_obs_phi.append( int(round(obs/(2*np.pi) *\
                                    len(theta_var_coupled.codomain )))\
                                    %len(theta_var_coupled.codomain )   )
            ll_idx_obs_phi.append(l_idx_obs_phi)


        ##################### PLOT FITS ##################
        zp = zip(enumerate(ll_signal),l_gamma_div, l_logP_div, ll_area,
            ll_obs_phi, ll_idx_cell_cycle_start, ll_idx_peak)
        for ((idx, signal), gamma, logP, area, l_obs_phi,
            l_idx_cell_cycle_start, l_idx_peak) in zp:
            plt_result = PlotResults(gamma, l_var, signal_model, signal,
                                    waveform = W, logP = logP,
                                    temperature = temperature, cell = cell)
            E_model, E_theta, E_A, E_B = \
                                plt_result.plotEverythingEsperance( True, idx)

        ##################### PLOT COUPLING AND PHASE SPACE DENSITY ############
        #plot phase density
        plot_phase_space_density(l_var, l_gamma_div, ll_idx_obs_phi,
                                F_superimpose = F, save = True, cmap = bwr,
                                temperature = temperature, cell = cell,
                                period = gamma_reg,
                                folder = 'Results/TestGamma' )

        #plot coupling
        plt.imshow(F.T, cmap=bwr, vmin=-0.3, vmax=0.3,
                    interpolation='nearest', origin='lower',
                    extent=[0, 2*np.pi,0, 2*np.pi])
        plt.colorbar()
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\phi$')
        plt.title('Coupling Function')
        plt.savefig("Results/TestGamma/Coupling_"+str(temperature)+"_"\
                    +cell+'_'+str(gamma_reg)+'.pdf')
        plt.close()



""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':

    test_gamma(cell = 'NIH3T3', temperature = None, nb_trace = 100000,
                nb_iter = 20)
