# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
import seaborn as sn
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from mpl_toolkits import axes_grid1
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.realpath('..'))
from Functions.create_hidden_variables import create_hidden_variables
from Functions.make_colormap import make_colormap

#colormap
c = mcolors.ColorConverter().to_rgb
bwr = make_colormap(  [c('blue'), c('white'), 0.48, c('white'),
                                                    0.52,c('white'),  c('red')])

#nice plotting style
sn.set_style("whitegrid", { 'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

""""""""""""""""""""" FUNCTION """""""""""""""""""""

def load_path(type_sim = 'Real', type_param = 'init', type_cell = 'NIH3T3',
              temperature = '37', type_div = 'nodiv', local = False):
    """
    Return the correct path for the file of parameters given a set of
    conditions.

    Parameters
    ----------
    type_sim : string
        Simulation type. 'Real' or 'Silico'.
    type_param : string
        Parameters type. 'init' or 'opt' for initial or optimal.
    type_cell : string
        Cell type. 'NIH3T3' or 'U2OS'.
    temperature : integer
        Temperature condition . 34, 37 or 40.
    type_div : string
        Dividing cells or not. 'div' or 'nodiv'.
    local : bool
        Specify it the script is launched locally or not (to have the correct
        path)

    Returns
    -------
    The desired path.
    """
    if local:
        pref = "../"
    else:
        pref =''
    try:
        with open(pref+'Parameters/'+type_sim+'/'+type_param+'_parameters_'\
                +type_div+'_'+str(temperature)+"_"+type_cell+'.p', 'rb') as f:
            path = pref+'Parameters/'+type_sim+'/'+type_param+'_parameters_'\
                   +type_div+'_'+str(temperature)+"_"+type_cell+'.p'
    except:
        print("Error, file " + pref+'Parameters/'+type_sim+'/'+type_param+\
              '_parameters_'+type_div+'_'+str(temperature)+"_"+type_cell+'.p'\
              + "doesn't exist")
        path = None
    return path

def display_parameters_from_file(path, show = False, temperature = None,
                                                                   cell = None):
    """
    Display all the parameters from a given file.

    Parameters
    ----------
    path : string
        Path for the parameters file.
    show : bool
        Display the figures (e.g. coupling function, intial condition) or not.
    temperature : integer
        Temperature condition . 34, 37 or 40.
    cell : string
        Cell type. 'NIH3T3' or 'U2OS'.
    """

    with open(path, 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)
    display_parameters(l_parameters, show, temperature, cell)

def display_parameters(l_parameters, show, temperature = None, cell = None):
    """
    Display all the parameters from a list.

    Parameters
    ----------
    l_parameters : list
        List of parameters.
    show : bool
        Display the figures (e.g. coupling function, intial condition) or not.
    temperature : integer
        Temperature condition . 34, 37 or 40.
    cell : string
        Cell type. 'NIH3T3' or 'U2OS'.
    """
    [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = l_parameters

    print("")
    print("########## SIMULATION PARAMETERS ##########")
    print("dt = ", dt)
    print("sigma_em_circadian = ", sigma_em_circadian)
    print("")
    print("########## THETA PARAMETERS ##########")
    print("N_theta = ", N_theta)
    print("std_theta = ", std_theta)
    print("period_theta = ", period_theta)
    print("l_boundaries_theta = ", l_boundaries_theta)
    print("w_theta = ", w_theta)
    print("")
    print("########## PHI PARAMETERS ##########")
    print("N_phi = ", N_phi)
    print("std_phi = ", std_phi)
    print("period_phi = ", period_phi)
    print("l_boundaries_phi = ", l_boundaries_phi)
    print("w_phi = ", w_phi)
    print("")
    print("########## AMPLITUDE PARAMETERS ##########")
    print("N_amplitude_theta = ", N_amplitude_theta)
    print("mu_amplitude_theta = ", mu_amplitude_theta)
    print("std_amplitude_theta = ", std_amplitude_theta)
    print("gamma_amplitude_theta = ", gamma_amplitude_theta)
    print("l_boundaries_amplitude_theta = ", l_boundaries_amplitude_theta)
    print("")
    print("########## BACKGROUND PARAMETERS ##########")
    print("N_background_theta = ", N_background_theta)
    print("mu_background_theta = ", mu_background_theta)
    print("std_background_theta = ", std_background_theta)
    print("gamma_background_theta = ", gamma_background_theta)
    print("l_boundaries_background_theta = ", l_boundaries_background_theta)
    print("")

    theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)

    if show:
        if W is None:
            print("No waveform recorded, default waveform used")
            W= [((1+np.cos( np.array(theta)))/2)**1.6 for theta \
                                                    in theta_var_coupled.domain]
        plt.plot(theta_var_coupled.domain, W)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$s(\theta)$')
        plt.title('circadian waveform')
        plt.show()
        plt.close()

        W = np.append(W[int(N_theta/2):], W[:int(N_theta/2)])
        plt.plot(theta_var_coupled.domain, W)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$s(\theta)$')
        plt.title('circadian waveform')
        plt.show()
        plt.close()

        if F is None:
            print("No coupling recorded, blank coupling used")
            F = np.zeros((N_theta, N_phi))


        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            """Add a vertical color bar to an image plot."""
            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            current_ax = plt.gca()
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.sca(current_ax)
            return im.axes.figure.colorbar(im, cax=cax, **kwargs)

        plt.figure(figsize=(5*1.2,5*1.2))
        ax = plt.gca()
        #ax.grid(False)
        im = plt.imshow(F.T, cmap=bwr, vmin=-0.3, vmax=0.3, \
                        interpolation='spline16', origin='lower',
                        extent=[0, 1,0, 1])
        add_colorbar(im, label = r'Acceleration ($rad.h^{-1}$)')
        plt.xlabel(r'Circadian phase $\theta$')
        plt.ylabel(r'Cell-cycle phase $\phi$')
        plt.plot([0,1],[0.22,0.22], '--', color = 'grey')
        plt.text(x = 0.05, y = 0.14, s='G1', color = 'grey', fontsize=12)
        plt.text(x = 0.06, y = 0.27, s='S', color = 'grey', fontsize=12)
        plt.plot([0,1],[0.84,0.84], '--', color = 'grey')
        plt.text(x = 0.05, y = 0.84-0.08, s='S/G2', color = 'grey', fontsize=12)
        plt.text(x = 0.06, y = 0.84+0.05, s='M', color = 'grey', fontsize=12)

        #plt.xlabel("Circadian phase")
        #plt.ylabel("Cell-cycle phase")

        #plt.title('Coupling Function')
        plt.tight_layout()
        try:
            plt.savefig("../Results/Disp/F_inferred_"+str(temperature)+"_"\
                                                                   +cell+'.pdf')
        except:
            pass
        plt.show()
        plt.close()


        theta_domain = theta_var_coupled.domain
        amplitude_domain = amplitude_var.domain
        background_domain = background_var.domain
        if pi is None:
            print("No pi recorded, uniform distribution used")
            pi_theta = theta_var_coupled.pi
            pi_amplitude = amplitude_var.pi
            pi_background = background_var.pi
        else:
            pi_theta = np.sum(pi, axis = (1,2))
            pi_amplitude = np.sum(pi, axis = (0,2))
            pi_background = np.sum(pi, axis = (0,1))

        plt.plot(theta_domain, pi_theta)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$pi(\theta)$')
        plt.title('Initial circadian phase distribution')
        plt.show()
        plt.close()

        plt.plot(amplitude_domain, pi_amplitude)
        plt.xlabel(r'$A$')
        plt.ylabel(r'$pi(A)$')
        plt.title('Initial amplitude distribution')
        plt.show()
        plt.close()

        plt.plot(background_domain, pi_background)
        plt.xlabel(r'$B$')
        plt.ylabel(r'$pi(B)$')
        plt.title('Initial background distribution')
        plt.show()
        plt.close()


""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    type_sim = 'Real'
    type_param = 'opt'
    temperature = None
    type_cell = 'NIH3T3'
    type_div = 'div'

    path = load_path (type_sim = type_sim, type_param = type_param, \
                      type_cell = type_cell, temperature = temperature,
                      type_div = type_div, local = True)
    print(path)
    display_parameters_from_file(path, show = True, temperature = temperature,
                                 cell = type_cell)
