# -*- coding: utf-8 -*-
""" This script is used to validate the estimation process for the parameters
obtained from non-dividing traces, i.e. the coupling parameters. """
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import pickle
import numpy as np
import sys
import copy
import matplotlib
matplotlib.use('Agg') #to run the script on a distant server
import matplotlib.pyplot as plt
from scipy import interpolate
import random
import seaborn as sn
import os
plt.style.use('seaborn-whitegrid')

sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

### Import internal modules
from Classes.LoadData import LoadData
from Classes.HMM_SemiCoupled import HMM_SemiCoupled
import Classes.EM as EM
from Classes.HMMsim import HMMsim

###Import internal functions
from Functions.create_hidden_variables import create_hidden_variables
from Functions.make_colormap import make_colormap
from Functions.signal_model import signal_model
from Functions.display_parameters import display_parameters_from_file
from RawDataAnalysis.estimate_OU_par import estimate_OU_par_from_signal
from RawDataAnalysis.estimate_phase_dev import estimate_phase_dev_from_signal
from RawDataAnalysis.estimate_waveform import estimate_waveform
from RawDataAnalysis.estimate_waveform import estimate_waveform_from_signal

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
    nb_traces = 2000
    cell = 'NIH3T3'
    temperature = None

""""""""""""""""""""" LOAD OPTIMIZED PARAMETERS """""""""""""""""
path = 'Parameters/Real/opt_parameters_nodiv_'+str(temperature)+"_"+cell+'.p'
with open(path, 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)

pi = None #to start with distributions at equilibrium
W_init = copy.copy(W) #save to make a plot in the end
""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
display_parameters_from_file(path, show = True)


""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""
theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )
l_var = [theta_var_coupled, amplitude_var, background_var]

""""""""""""""""""""" COMPUTE NB_TRACE_MAX """""""""""""""""
if cell == 'NIH3T3':
    path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    dataClass=LoadData(path, nb_traces, temperature = temperature,
                        division = False)
else:
    path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
    dataClass=LoadData(path, nb_traces, temperature = temperature,
                        division = True) #all thz U2OS divide

(ll_area_tot_flat, ll_signal_tot_flat, ll_nan_circadian_factor_tot_flat,
    ll_obs_phi_tot_flat, T_theta, T_phi) = dataClass.load()
nb_traces_max = len(ll_signal_tot_flat)
if nb_traces>nb_traces_max:
    print("CAUTION : too many traces, number of traces generated reduced to: ",
            nb_traces_max)
    nb_traces = nb_traces_max

""""""""""""""""""""" GENERATE UNCOUPLED TRACES """""""""""
sim = HMMsim( l_var, signal_model , sigma_em_circadian,  waveform = W ,
              dt=0.5, uniform = True )
ll_t_l_xi, ll_t_obs  =  sim.simulate_n_traces(nb_traces=nb_traces, tf=60)

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

""""""""""""""""""""" COMPUTE ll_peak and possibly ll_idx_cell_cycle_start """""
ll_peak = []
ll_idx_cell_cycle_start=[]
#phase 0 is taken as reference for peak
for ll_xi_circadian in lll_xi_circadian:
    l_peak = []
    zp = zip(ll_xi_circadian[:-1], ll_xi_circadian[1:])
    for l_xi_circadian_1, l_xi_circadian_2 in zp:
        if l_xi_circadian_2[0]-l_xi_circadian_1[0]<-np.pi \
                                                    and not 1 in l_peak[-20:]:
            l_peak.append(1)
        else:
            l_peak.append(0)
    l_peak.append(0)
    ll_peak.append(l_peak)
if cell=='U2OS':
    for ll_xi_nucleus in lll_xi_nucleus:
        l_idx = []
        zp = zip(enumerate(ll_xi_nucleus[:-1]), ll_xi_nucleus[1:])
        for (idx, l_xi_nucleus_1), l_xi_nucleus_2 in zp:
            if l_xi_nucleus_2[0]-l_xi_nucleus_1[0] < -np.pi:
                l_idx.append(idx)
        ll_idx_cell_cycle_start.append(l_idx)

""""""""""""""""""""" PLOT TRACE EXAMPLES """""""""""""""""
zp = zip(enumerate(lll_xi_nucleus), lll_xi_circadian, ll_obs_circadian)
for (idx_trace, ll_xi_nucleus), ll_xi_circadian, l_obs_circadian in zp:
    tspan = np.linspace(0, len(l_obs_circadian)/2, len(l_obs_circadian),
                        endpoint=False)
    ll_xi_nucleus = np.array(ll_xi_nucleus)
    ll_xi_circadian = np.array(ll_xi_circadian)
    plt.plot(tspan, l_obs_circadian, '-')
    plt.plot(tspan, ll_xi_circadian[:,0]/(2*np.pi), '--')
    plt.plot(tspan, np.exp(ll_xi_circadian[:,1]), '--')
    plt.plot(tspan, ll_xi_circadian[:,2], '--')
    plt.plot(tspan, ll_xi_nucleus[:,0]/(2*np.pi), '--')
    #plt.show()
    plt.close()
    if idx_trace>1:
        break

""""""""""""""""""""" ESTIMATE PARAMETERS """""""""""""""""
mu_A, std_A, mu_B, std_B = estimate_OU_par_from_signal(ll_obs_circadian, W,
                                            gamma_A = gamma_amplitude_theta,
                                            gamma_B = gamma_background_theta)
std_phase = estimate_phase_dev_from_signal(ll_peak)
W = estimate_waveform_from_signal(ll_signal = ll_obs_circadian,
                            ll_peak = ll_peak,
                            domain_theta = theta_var_coupled.domain,
                            ll_idx_cell_cycle_start = ll_idx_cell_cycle_start)

""""""""""""""""""""" COMPARE ESTIMATES AND TRUE VALUES """""""""""""""""
print("Phase deviation (theoretical vs estimated): ", std_theta, std_phase)
print("Mean amplitude (theoretical vs estimated): ", mu_amplitude_theta, mu_A)
print("Std amplitude (theoretical vs estimated): ", std_amplitude_theta, std_A)
print("Mean background (theoretical vs estimated): ", mu_background_theta, mu_B)
print("Std background (theoretical vs estimated): ",
                                                    std_background_theta, std_B)

""""""""""""""""""""" SAVE ESTIMATES AND TRUE VALUES IN A TEXT FILE"""""""""""
with open('Results/Validation/estimation_non_dividing_'\
                                  +str(temperature)+"_"+cell+'.txt', 'w') as f:
    f.write("Phase deviation (theoretical vs estimated): " \
                                  + str(std_theta) + ' '+ str(std_phase) + '\n')
    f.write("Mean amplitude (theoretical vs estimated): "\
                              + str(mu_amplitude_theta) + ' '+ str(mu_A) + '\n')
    f.write("Std amplitude (theoretical vs estimated): "\
                            + str(std_amplitude_theta) + ' '+ str(std_A) + '\n')
    f.write("Mean background (theoretical vs estimated): "\
                             + str(mu_background_theta) + ' '+ str(mu_B) + '\n')
    f.write("Std background (theoretical vs estimated): "\
                           + str(std_background_theta) + ' '+ str(std_B) + '\n')

""""""""""""""""""""" COMPARE TRUE AND INFERRED WAVEFORMS """""""""""""""""
plt.plot(theta_var_coupled.domain/(2*np.pi), W, label = 'Inferred')
plt.plot(theta_var_coupled.domain/(2*np.pi), W_init, label = 'Simulation')
plt.xlabel(r"circadian phase $\theta$")
plt.ylabel(r"Waveform $\omega(\theta)$")
plt.legend()
plt.ylim([-0.1,1.1])
plt.savefig('Results/Validation/W_estimated_'+str(temperature)+"_"+cell+'.pdf')
plt.show()
plt.close()

""""""""""""""""""""" WRAP PARAMETERS """""""""""""""""
l_parameters = [dt, sigma_em_circadian, W, pi,
                N_theta, std_phase, period_theta, l_boundaries_theta, w_theta,
                N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
                N_amplitude_theta, mu_A, std_A,  gamma_amplitude_theta,
                l_boundaries_amplitude_theta,
                N_background_theta, mu_B, std_B, gamma_background_theta,
                l_boundaries_background_theta,
                F]
pickle.dump( l_parameters, open( "Parameters/Silico/est_parameters_nodiv_"\
                                +str(temperature)+"_"+cell+'.p', "wb" ) )
