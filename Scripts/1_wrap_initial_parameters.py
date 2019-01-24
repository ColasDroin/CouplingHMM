# -*- coding: utf-8 -*-
""" This script is used to set and record in a pickle all the parameters which
are not optimized under the EM algorithm """
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import pickle
import numpy as np
import sys
import os

sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

###Import internal functions
from RawDataAnalysis.estimate_OU_par import estimate_OU_par
from RawDataAnalysis.estimate_phase_dev import estimate_phase_dev
from RawDataAnalysis.estimate_waveform import estimate_waveform
from Functions.display_parameters import display_parameters_from_file
from Functions.display_parameters import display_parameters

#access main directory
os.chdir('..')

""""""""""""""""""""" DEFINE CONDITIONS """""""""""""""""
try:
    cell = sys.argv[1]
    if sys.argv[2]=="None":
        temperature = None
    else:
        temperature = int(sys.argv[2])
except:
    print("No shell input given, default arguments used")
    cell = 'NIH3T3'
    temperature = None

""""""""""""""""""""" DEFINE PARAMETERS """""""""""""""""
#fixed simulation parameters
dt = 0.5

#theta phase parameters
N_theta = 48
std_theta, std_T = estimate_phase_dev(cell, temperature)
period_theta = 24
l_boundaries_theta = (0, 2*np.pi)
w_theta =  2*np.pi/period_theta

#phi phase parameters
N_phi =  48
std_phi = 0.1 #useless for inference
period_phi = 22
l_boundaries_phi = (0, 2*np.pi)
w_phi = 2*np.pi/period_phi

#coupling function
F = None

#emission parameters
sigma_em_circadian = 0.15
W = estimate_waveform(cell = 'all', temperature = temperature,
                     domain_theta = np.linspace(0,2*np.pi, N_theta,
                                                endpoint = False))
#W = estimate_waveform(cell = cell, temperature = temperature,
#                    domain_theta = np.linspace(0,2*np.pi, N_theta,
#                                                endpoint = False))
pi = None

#OU parameters
if cell=='NIH3T3':
    gamma_amplitude_theta = 0.075
    gamma_background_theta = 0.075
else:
    gamma_amplitude_theta = 0.1
    gamma_background_theta = 0.1

mu_A, std_A, mu_B, std_B = estimate_OU_par(cell, temperature, W = W,
                                            gamma_A = gamma_amplitude_theta,
                                            gamma_B = gamma_background_theta)

#amplitude parameters
N_amplitude_theta = 30
mu_amplitude_theta = mu_A
std_amplitude_theta = std_A
l_boundaries_amplitude_theta = (mu_amplitude_theta-5*std_amplitude_theta,
                                mu_amplitude_theta+5*std_amplitude_theta)

#background parameters
N_background_theta = 30
mu_background_theta = mu_B
std_background_theta = std_B
l_boundaries_background_theta = (mu_background_theta-5*std_background_theta,
                                mu_background_theta+5* std_background_theta)

""""""""""""""""""""" WRAP PARAMETERS """""""""""""""""
path = 'Parameters/Real/init_parameters_nodiv_'+str(temperature)+"_"+cell+".p"
with open(path, 'wb') as f:
    pickle.dump([dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F], f)

""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
display_parameters_from_file(path, show = False)
