# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

### Import internal modules
sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
from Classes.StateVar import StateVar
from Classes.StateVarSemiCoupled import StateVarSemiCoupled

""""""""""""""""""""" FUNCTION """""""""""""""""""""
def create_hidden_variables(path_parameters_file = None, l_parameters = None):
    """
    Create and store the hidden variables (circadian phase, amplitude,
    background) in a list. The parametrization can be given either as a
    filepath, either as a list.

    Parameters
    ----------
    path_parameters_file : string
        Path of the file containing the paramters to build the hidden variables
        from.
    l_parameters : list
        List of parameters to build the hidden variables
        from.

    Returns
    -------
    The list of hidden variables.
    """

    ### LOAD PARAMETERS ###
    if path_parameters_file is not None:
        with open(path_parameters_file, 'rb') as f:
            [dt, sigma_em_circadian, W, pi,
            N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
            N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
            N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
            gamma_amplitude_theta, l_boundaries_amplitude_theta,
            N_background_theta, mu_background_theta, std_background_theta,
            gamma_background_theta, l_boundaries_background_theta,
            F] = pickle.load(f)
    elif l_parameters is not None:
        [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = l_parameters
    else:
        print("Error: a filename or a list of parameters must be given")

    ### DEFINE TRANSITION KERNELS ###

    def f_trans_OU(s, l_parameters, dt):
        """
        l_parameters[0]: mu
        l_parameters[1]: gamma
        l_parameters[1]: std
        """
        return (l_parameters[0]+(s-l_parameters[0])*np.exp(-l_parameters[1]*dt),
                l_parameters[2]/(2*l_parameters[1])**0.5\
                                        *(1-np.exp(-2*l_parameters[1]*dt))**0.5)

    def f_trans_theta(theta, l_parameters, dt, phi = None, F = None):
        """
        l_parameters[0]: frequency
        l_parameters[1]: std
        """
        if F is not None and phi is not None:
            c1 = int(np.floor( theta*N_theta/(2*np.pi)))
            c2 = int(np.floor(phi*N_phi/(2*np.pi)))
            return theta+(F[c1,c2] + l_parameters[0])*dt,l_parameters[1]*dt**0.5
        else:
            return theta+ l_parameters[0]*dt, l_parameters[1]*dt**0.5

    ### CREATE HIDDEN VARIABLES ###

    l_parameters_theta = [w_theta, std_theta]
    theta_var_coupled = StateVarSemiCoupled("Theta", l_boundaries_theta,
                                            N_theta, f_trans_theta,
                                            l_parameters_theta,  F,
                                            l_boundaries_phi, N_phi,  dt)
    #plt.matshow(theta_var_coupled.TR[:,0,:])
    #plt.show()


    l_parameters_amplitude_theta = [mu_amplitude_theta, gamma_amplitude_theta,
                                                            std_amplitude_theta]
    amplitude_var_theta = StateVar("Amplitude", l_boundaries_amplitude_theta,
                                    N_amplitude_theta, f_trans_OU,
                                    l_parameters_amplitude_theta, dt)
    #plt.matshow(amplitude_var.TR)
    #plt.show()

    l_parameters_background_theta = [mu_background_theta,
                                    gamma_background_theta,
                                    std_background_theta]
    background_var_theta = StateVar("Background", l_boundaries_background_theta,
                                    N_background_theta, f_trans_OU,
                                    l_parameters_background_theta, dt)
    #plt.matshow(background_var.TR)
    #plt.show()

    return theta_var_coupled, amplitude_var_theta, background_var_theta


""""""""""""""""""""" TEST """""""""""""""""""""
if __name__ == '__main__':
    theta_var_coupled, amplitude_var, background_var = \
                        create_hidden_variables('../Parameters/Real/opt_parameters_div_None_NIH3T3.p')
    plt.matshow(theta_var_coupled.TR[:,0,:])
    plt.show()
    plt.matshow(amplitude_var.TR)
    plt.show()
    plt.matshow(background_var.TR)
    plt.show()
