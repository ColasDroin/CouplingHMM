# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import seaborn as sn
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
import copy
from mpl_toolkits import axes_grid1

### Import internal modules
sys.path.insert(0, os.path.realpath('..'))
from Functions.create_hidden_variables import create_hidden_variables

#to edit figure text
matplotlib.rcParams['pdf.fonttype'] = 42

#nice plotting style
sn.set_style("whitegrid", { 'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

""""""""""""""""""""" FUNCTION """""""""""""""""""""
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot.
    See https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    for more information."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def plot_phase_space_density(l_var, l_gamma, ll_idx_phi, F_superimpose = None,
                             save =False, cmap = 'bwr', temperature = 37,
                             cell = 'NIH3T3', period = None, folder = None ,
                             attractor = None, M_at = None):
    """
    Plot the phase-space density for a given list of gamma matrices, with the
    possibility of superimposing the plot on  a given coupling, and also
    plotting a simulated attractor on top of it.

    Parameters
    ----------
    l_var : list
        List of hidden variables.
    l_gamma : list
        List of gamma matrices obtained from the forward-backward algorithm.
    ll_idx_phi : list
        List of list of cell-cycle indexes (1st dim : traces, 2nd dim : time).
    F_superimpose : bool
        Make a plot in which the phase-space density is superimposed with the
        coupling function.
    save : bool
        Save or not the plots.
    cmap : string or colormap
        Custom colormap.
    temperature : integer
        Temperature condition . 34, 37 or 40.
    cell : string
        Cell type. 'NIH3T3' or 'U2OS'.
    period : int
        Cell-cycle period, to print on the plot.
    folder : string
        To specify a particular location for the plot.
    attractor : list of lists
        Simulated attractor to be superimposed on the data.
    M_at : matrix
        Previously computed density matrix.

    Returns
    -------
    A density matrix.
    """
    N_theta = l_var[0].nb_substates
    N_phi= len(l_var[0].codomain)
    if M_at is None:
        M_num = np.zeros((N_theta, N_phi ))
        for l_idx_phi, gamma in zip(ll_idx_phi, l_gamma):
            gamma_phase = list(np.sum(gamma, axis=(2,3)))
            if period is not None:
                if period<26:
                    #print('phase space correction')
                    #do linear interpolation between the missing phi
                    l_diff_idx = list(np.array(l_idx_phi[1:])\
                                                        -np.array(l_idx_phi[:-1]))
                    idx_diff=0
                    #print(l_idx_phi)
                    while idx_diff<len(l_idx_phi)-1:
                        #print('idx diff', idx_diff)
                        #print('len idx phi -1 ', len(l_idx_phi)-1)
                        #print('len diff', len(l_diff_idx))
                    #for idx_diff, diff in enumerate(l_diff_idx):
                        diff = l_diff_idx[idx_diff]
                        if diff==2 or diff==-46:
                            l_idx_phi.insert(idx_diff+1,
                                                      (l_idx_phi[idx_diff]+1)%N_phi)
                            gamma_phase.insert(idx_diff+1,
                                  (gamma_phase[idx_diff]+gamma_phase[idx_diff+1])/2)
                            l_diff_idx.insert(idx_diff+1,1)
                            idx_diff+=2
                            continue

                        if diff==3 or diff==-45:
                            l_idx_phi.insert(idx_diff+1,
                                                      (l_idx_phi[idx_diff]+2)%N_phi)
                            l_idx_phi.insert(idx_diff+1,
                                                      (l_idx_phi[idx_diff]+1)%N_phi)

                            gamma_phase_before=copy.deepcopy(gamma_phase[idx_diff])
                            gamma_phase_after=copy.deepcopy(gamma_phase[idx_diff+1])
                            gamma_phase.insert(idx_diff+1,
                                       gamma_phase_before*1/3+gamma_phase_after*2/3)
                            gamma_phase.insert(idx_diff+1,
                                       gamma_phase_before*2/3+gamma_phase_after*1/3)

                            l_diff_idx.insert(idx_diff+1,1)
                            l_diff_idx.insert(idx_diff+1,1)

                            idx_diff+=3
                            continue


                        idx_diff+=1
                    #print(l_idx_phi)
            #remove last value so every cell-cycle is represented once
            for idx_phi, gamma_t in zip(l_idx_phi[:-1], gamma_phase[:-1]):
                M_num[:, idx_phi]+=gamma_t
        M_at = M_num
    else:
        pass

    #divide by two density of first line
    #M_at[:,0] = M_at[:,0]/2
    '''
    #normalize by cell-cycle density
    for i in range(N_phi):
        if np.sum(M_at[:,i])<10:
            M_at[:,i] = M_at[:,(i-1)%N_theta]+M_at[:,(i+1)%N_theta]
        M_at[:,i] = M_at[:,i]/np.sum(M_at[:,i])
    '''

    '''
    #solve cell-cycle linearization artifacts
    M_at_new = copy.deepcopy(M_at)
    for i in range(N_theta):
    #for j in range(N_phi):
        vertical_mean = np.sum(M_at[:,(i-1)%N_theta])/2\
                                                +np.sum(M_at[:,(i+1)%N_theta])/2
        print("current line", i, np.sum(M_at[:,(i)%N_theta]))
        print("above line", i+1, np.sum(M_at[:,(i+1)%N_theta]))
        print("below line", i-1, np.sum(M_at[:,(i-1)%N_theta]))
        print("average", vertical_mean)
        if np.sum(M_at[:,i])<vertical_mean/1.5:
            print("###passed condition###")
            #M_at_new[:,i] = (M_at[:,(i-1)%N_theta]+M_at[:,(i+1)%N_theta])/2

            #mean_vertical = (M_at[(i-1)%N_theta,(j-1)%N_phi]\
                                +M_at[(i)%N_theta,(j-1)%N_phi]\
                                +M_at[(i+1)%N_theta,(j-1)%N_phi]\
                                +M_at[(i-1)%N_theta,(j+1)%N_phi]\
                                +M_at[(i)%N_theta,(j+1)%N_phi]\
                                +M_at[(i+1)%N_theta,(j+1)%N_phi])/6
            #if M_at[i,j]<mean_vertical/1.5 or M_at[i,j]>mean_vertical*1.5:
            #M_at_new[i,j] = mean_vertical
    M_at = M_at_new
    '''
    #normalize by total sum, such that each line in average sums to 1
    M_at = M_at/np.sum(M_at)*N_phi

    #print(M_at)
    plt.figure(figsize=(5*1.0,5*1.0))


    '''
    plt.imshow(np.array(M_at.T), cmap='inferno',  origin='lower', vmin=-0.0,
                                        vmax=0.1, extent=[0,2*np.pi,0,2*np.pi])
    '''
    levels = np.arange(0.02, 0.10, 0.02)
    contours = plt.contour(np.array(M_at.T), levels, colors='black',
                           origin='lower',
                           extent=[0,1,0,1])
    plt.clabel(contours, inline=True, fontsize=8, fmt = '%.2f')
    #im = plt.imshow(np.array(M_at.T), extent=[0, 1, 0, 1],
    #                origin='lower',  vmin=-0.0, vmax=0.10,cmap='Reds',
                     #alpha=0.8, interpolation = 'bilinear')
    im = plt.imshow(np.array(M_at.T), extent=[0, 1, 0, 1], origin='lower',
                    vmin=-0.0, vmax=0.1,  cmap='YlGn', alpha=0.8,
                    interpolation = 'bilinear')
    #im = plt.imshow(np.array(M_at.T), extent=[0, 1, 0, 1], origin='lower',
                     #vmin=-0.0, vmax=0.1,  cmap='YlGn', alpha=0.8,
                     #interpolation = 'none')
    #add_colorbar(im, label = r'Phase density')
    #plt.colorbar()
    if period is not None and not np.isnan(period):
        plt.title(r"$T_{\phi} = $" + str(int(round(period))))

    #plt.plot([0,1],[0.22,0.22], '--', color = 'grey')
    #plt.text(x = 0.45, y = 0.14, s='G1', color = 'grey', fontsize=12)
    #plt.text(x = 0.46, y = 0.27, s='S', color = 'grey', fontsize=12)
    #plt.plot([0,1],[0.84,0.84], '--', color = 'grey')
    #plt.text(x = 0.05, y = 0.84-0.08, s='S/G2', color = 'grey', fontsize=12)
    #plt.text(x = 0.06, y = 0.84+0.05, s='M', color = 'grey', fontsize=12)

    plt.xlabel(r'Circadian phase $\theta$')
    plt.ylabel(r'Cell-cycle phase $\phi$')
    plt.tight_layout()
    if save:
        if folder is None:
            plt.savefig("Results/PhaseSpace/PhaseSpaceDensity_"\
                            +str(temperature)+"_"+cell+"_"+str(period)+'.pdf')
        else:
            plt.savefig(folder + "/PhaseSpaceDensity_"\
                            +str(temperature)+"_"+cell+"_"+str(period)+'.pdf')
    else:
        #plt.show()
        pass
    plt.close()

    if attractor is not None and F_superimpose is None:
        print("BUG, attractor can only be plotted if coupling function given")
    if F_superimpose is not None:
        plt.figure(figsize=(5*1.0,5*1.0))

        im = plt.imshow(F_superimpose.T, cmap=cmap, vmin=-0.3,
                        vmax=0.3,origin='lower',interpolation='spline16',
                        extent=[0,1,0,1])
        plt.xlabel(r'Circadian phase $\theta$')
        plt.ylabel(r'Cell-cycle phase $\phi$')
        plt.title(r'Coupling function')
        levels = np.arange(0.02, 0.10, 0.02)
        add_colorbar(im, label = r'Acceleration ($rad.h^{-1}$)')
        #plt.colorbar()
        plt.contour(np.array(M_at.T), levels, colors='k', origin='lower',
                    extent=[0,1,0,1])
        if attractor is not None:
            for l_phase_theta, l_phase_phi in zip(attractor[0],attractor[1]):
                plt.plot(np.array(l_phase_theta)/(2*np.pi),
                        np.array(l_phase_phi)/(2*np.pi),
                        color = 'green')
        plt.tight_layout()
        if save:
            if folder is None:
                plt.savefig("Results/PhaseSpace/PhaseSpaceDensitySuperimposed_"\
                            +str(temperature)+"_"+cell+"_"+str(period)+'.pdf')
            else:
                plt.savefig(folder + "/PhaseSpaceDensitySuperimposed_"\
                            +str(temperature)+"_"+cell+"_"+str(period)+'.pdf')
        else:
            #plt.show()
            pass
        plt.close()

    return M_at
