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
from scipy.integrate import dblquad
from scipy import interpolate

#Import internal modules
sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

from Classes.LoadData import LoadData
from Classes.PlotResults import PlotResults
from Classes.HMM_SemiCoupled import HMM_SemiCoupled

from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.signal_model import signal_model

np.set_printoptions(threshold=np.nan)
""""""""""""""""""""" FUNCTION """""""""""""""""""""
def test_potential(cell = 'NIH3T3', temperature = 37):
    """
    Compute the potential associated to the coupling function in the system of
    coupled oscillators.

    Parameters
    ----------
    cell : string
        Cell condition.
    temperature : integer
        Temperature condition.
    """
    ##################### LOAD OPTIMIZED PARAMETERS ##################
    path = '../Parameters/Real/opt_parameters_div_'+str(temperature)+"_"\
                                                                    +cell+'.p'
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
    ##################### CREATE HIDDEN VARIABLES ##################
    theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )

    ##################### COMPUTE POTENTIAL ##################
    Fbis = F+w_theta
    F_temp = np.hstack((Fbis,Fbis,Fbis))
    F_temp = np.vstack((F_temp, F_temp, F_temp))
    f = interpolate.interp2d(np.linspace(-2*np.pi,4*np.pi, 3*F.shape[0],
                            endpoint = False),
                            np.linspace(-2*np.pi,4*np.pi, 3*F.shape[1],
                            endpoint = False), F_temp.T,
                            kind='cubic', bounds_error= True)

    U = np.zeros((N_theta,N_phi))
    for idx_theta in range(N_theta):
        for idx_phi in range(N_phi):
            print(idx_theta, idx_phi)
            U[idx_theta, idx_phi] = dblquad(f, 0,
                            theta_var_coupled.domain[idx_theta],
                            lambda x: 0,
                            lambda x: theta_var_coupled.codomain[idx_phi]
                                            )[0]


    path = "../Results/Potential/U_"+str(temperature)+"_"+cell+'.p'
    pickle.dump( U, open( path, "wb" ))

    plt.figure(figsize=(5*1.2,5*1.2))
    ax = plt.gca()
    im = plt.imshow(U.T, cmap='bwr', vmin=-0.3, vmax=0.3,
                    interpolation='spline16', origin='lower')
    #add_colorbar(im, label = r'Acceleration ($rad.h^{-1}$)')
    plt.xlabel(r'Circadian phase $\theta$')
    plt.ylabel(r'Cell-cycle phase $\phi$')
    plt.tight_layout()
    plt.savefig("../Results/Potential/U_"+str(temperature)+"_"+cell+'.pdf')
    plt.show()
    plt.close()


""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    test_potential(cell = 'NIH3T3', temperature = None)
