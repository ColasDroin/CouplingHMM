# -*- coding: utf-8 -*-
""" This script is used to study the system with deterministic simulations (e.g
Arnold tongues, attractor location, etc.). """
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import os
import sys
import matplotlib
matplotlib.use('Agg') #to run the script on a distant server
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

### Import internal modules
from Classes.DetSim import DetSim
from Classes.LoadData import LoadData

#access main directory
os.chdir('..')

""""""""""""""""""""" LOAD SHELL ARGUMENTS """""""""""""""""
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

with open('Parameters/Real/opt_parameters_div_'+str(temperature)\
                                                    +"_"+cell+'.p', 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)


""""""""""""""""""""" CREATE SIMULATION CLASS """""""""""""""""
detSim = DetSim(l_parameters, cell, temperature, upfolder = False)

""""""""""""""""""""" PLAY WITH SIMULATIONS """""""""""""""""

for T_phi in range(10,49,1):
    print('Tphi=', T_phi)
    detSim.plot_vectorfield_bis(T_phi, save =  True)

#plot speed vectorifield
detSim.plot_vectorfield_bis(T_phi = 12, save =  True)


#plot temporal trajectory
detSim.plot_signal(tf = 96, full_simulation = True)


#plot phase space trajectory
for K in np.linspace(0,2,6):
    print(K)
    detSim.plot_trajectory(ti = 400, tf = 1600,
                            rand = True, save = False, K= K )

for T_phi in range(14,35):
    print(T_phi)
    detSim.plot_trajectory(ti = 400, tf = 1600, rand = True,
                            save = True, T_phi = T_phi )

#plot devil staircase
detSim.plot_devil_staircase(l_arg = [np.linspace(2*np.pi/(2.5*24),
                                      2*np.pi/(24/3), 100) , 1, 500, False])
#
# #plot Arnold Tongues
# detSim.plot_arnold_tongues( 5, 80, tf = 200, random_init = True)
#
# #plot circadian period histogram
# detSim.plot_hist_period(100)
