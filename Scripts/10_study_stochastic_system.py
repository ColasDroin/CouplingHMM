# -*- coding: utf-8 -*-
""" This script is used to study the system with stochastic simulations. """
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import pickle
import numpy as np
import sys
import copy
import matplotlib
matplotlib.use('Agg') #to run the script on a distant server
import matplotlib.pyplot as plt
import os

sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

### Import internal modules
from Classes.PlotStochasticSpeedSpace import PlotStochasticSpeedSpace
from Classes.HMMsim import HMMsim
from Classes.LoadData import LoadData

###Import internal functions
from Functions.create_hidden_variables import create_hidden_variables
from Functions.signal_model import signal_model

#access main directory
os.chdir('..')
""""""""""""""""""""" LOAD SHELL ARGUMENTS """""""""""""""""
try:
    nb_traces = int(sys.argv[1])
    cell = sys.argv[2]
    if sys.argv[3]=="None":
        temperature = None
    else:
        temperature = int(sys.argv[3])
except:
    print("No shell input given, default arguments used")
    nb_traces = 30
    cell = 'NIH3T3'
    temperature = None

""""""""""""""""""""" LOAD OPTIMIZED PARAMETERS """""""""""""""""

with open('Parameters/Real/opt_parameters_div_'+str(temperature)+"_"\
                                                        +cell+'.p', 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)

""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""
theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )
l_var = [theta_var_coupled, amplitude_var, background_var]



""""""""""""""""""""" COMPUTE NB_TRACES_MAX """""""""""""""""
if cell == 'NIH3T3':
    path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
else:
    path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
dataClass=LoadData(path, nb_traces, temperature = temperature,
                   division = True, several_cell_cycles = True)
(ll_area_tot_flat, ll_signal_tot_flat, ll_nan_circadian_factor_tot_flat,
    ll_obs_phi_tot_flat, T_theta, T_phi) = dataClass.load()
nb_traces_max = len(ll_signal_tot_flat)
if nb_traces>nb_traces_max:
    print("CAUTION : too many traces, number of traces generated reduced to: ",
          nb_traces_max)
    nb_traces = nb_traces_max


""""""""""""""""""""" SIMULATE STOCHASTIC TRACES """""""""""""""""
sim = HMMsim(l_var, signal_model , sigma_em_circadian,  waveform = W ,
             dt=0.5, uniform = True )
ll_t_l_xi, ll_t_obs  =  sim.simulate_n_traces(nb_traces=nb_traces, tf = 60)

""""""""""""""""""""" REORDER VARIABLES """""""""""""""""
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

ll_val_phi =[np.array(ll_xi)[:,0] for  ll_xi in lll_xi_nucleus]


""""""""""""""""""""" CREATE ll_idx_obs_phi """""""""""""""""
ll_idx_obs_phi = []
for l_obs in ll_val_phi:
    l_idx_obs_phi = []
    for obs in l_obs:
        l_idx_obs_phi.append( int(round(obs/(2*np.pi) *\
            len(theta_var_coupled.codomain )))%len(theta_var_coupled.codomain ))
    ll_idx_obs_phi.append(l_idx_obs_phi)

""""""""""""""""""""" PLOT PHASE SPACE OBTAINED FROM STOCHASTIC DATA """""""""""
plt_space_theo = PlotStochasticSpeedSpace((lll_xi_circadian, lll_xi_nucleus),
                                            l_var, dt,w_phi, cell, temperature,
                                            cmap = None)
space_theta_th, space_phi_th, space_count_th \
                                = plt_space_theo.plotPhaseSpace(save_plot=True)
