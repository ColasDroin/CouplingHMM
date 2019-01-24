# -*- coding: utf-8 -*-
""" This script is used to compute the fits once the coupling and all parameters
have been optimized/estimated. From the phase inference, once can also plot the
phase-space density estimation. """
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import pickle
import numpy as np
import sys
import copy
import matplotlib
matplotlib.use('Agg') #to run the script on a distant server
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

### Import internal modules
from Classes.LoadData import LoadData
from Classes.HMM_SemiCoupled import HMM_SemiCoupled
from Classes.PlotResults import PlotResults
from Classes.DetSim import DetSim

###Import internal functions
from Functions.signal_model import signal_model
from Functions.create_hidden_variables import create_hidden_variables
from Functions.make_colormap import make_colormap
from Functions.plot_phase_space_density import plot_phase_space_density
from Functions.display_parameters import display_parameters_from_file

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
    if sys.argv[4]=="None":
        period = None
    else:
        period = int(sys.argv[4])

except:
    #for period in np.arange(14,31,2):
    print("No shell input given, default arguments used")
    nb_traces = 500
    cell = 'NIH3T3'
    temperature = None
    period = None

#if a given period is selected, then the None set of parameters is chosen
if period is not None:
    temp = temperature
    temperature = None

""""""""""""""""""""" LOAD OPTIMIZED PARAMETERS """""""""""""""""
path = 'Parameters/Real/opt_parameters_div_'+str(temperature)+"_"+cell+'.p'
with open(path, 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)

#correct for temperature
#    temperature = temp
""""""""""""""""""""" LOAD COLORMAP """""""""""""""""
c = mcolors.ColorConverter().to_rgb
bwr = make_colormap(  [c('blue'), c('white'), 0.48, c('white'),
                        0.52,c('white'),  c('red')])

""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
#display_parameters_from_file(path, show = True)

""""""""""""""""""""" LOAD DATA """""""""""""""""
if cell == 'NIH3T3':
    path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
else:
    path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
dataClass=LoadData(path, nb_traces, temperature = temperature, division = True,
                    several_cell_cycles = False, remove_odd_traces = True)
if period is None:
    (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, T_phi) \
                = dataClass.load(period_phi = period, load_annotation = True)
else:
    (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, std_T_theta,
        T_phi, std_T_phi) = dataClass.load(period_phi = period,
                                          load_annotation = True,
                                          force_temperature = False)

ll_idx_peak = [[idx for idx, v in enumerate(l_peak) \
                                                if v>0] for l_peak in ll_peak]
print(len(ll_signal), " traces kept")

""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""
theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )
l_var = [theta_var_coupled, amplitude_var, background_var]

""""""""""""""""""""" CREATE AND RUN HMM """""""""""""""""
hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian, ll_obs_phi,
                    waveform = W, ll_nan_factor = ll_nan_circadian_factor,
                    pi = pi, crop = True )
l_gamma_div, l_logP_div = hmm.run(project = False)

'''
""""""""""""""""""""" REMOVE BAD TRACES """""""""""""""""""""
Plim = np.percentile(l_logP_div, 00)
idx_to_keep = [i for i, logP in enumerate(l_logP_div) if logP>Plim ]
l_gamma_div = [l_gamma_div[i] for i in idx_to_keep]
ll_signal = [ll_signal[i] for i in idx_to_keep]
ll_area = [ll_area[i] for i in idx_to_keep]
l_logP_div = [l_logP_div[i] for i in idx_to_keep]
ll_obs_phi = [ll_obs_phi[i] for i in idx_to_keep]
ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] for i in idx_to_keep]
ll_idx_peak = [ll_idx_peak[i] for i in idx_to_keep]
print("Kept traces with div: ", len(idx_to_keep))
'''

""""""""""""""""""""" CROP SIGNALS FOR PLOTTING """""""""""""""""
l_first = [[it for it, obj in enumerate(l_obs_phi) \
                                    if obj!=-1][0] for l_obs_phi in ll_obs_phi]

l_last = [[len(l_obs_phi)-it-1 for it, obj \
      in enumerate(l_obs_phi[::-1]) if obj!=-1][0] for l_obs_phi in ll_obs_phi]

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


""""""""""""""""""""" CREATE ll_idx_obs_phi and ll_val_phi"""""""""""""""""
ll_idx_obs_phi = []
for l_obs in ll_obs_phi:
    l_idx_obs_phi = []
    for obs in l_obs:
        l_idx_obs_phi.append( int(round(obs/(2*np.pi) \
                            * len(theta_var_coupled.codomain )))\
                            %len(theta_var_coupled.codomain ) )
    ll_idx_obs_phi.append(l_idx_obs_phi)


""""""""""""""""""""" COMPUTE DETERMINISTIC ATTRACTOR """""""""""""""""
#l_parameters[6] = T_theta
if period is not None:
    l_parameters[11] = period
    l_parameters[13] = 2*np.pi/period

#print(T_theta, T_phi)
detSim = DetSim(l_parameters, cell, temperature)
l_theta, l_phi = detSim.plot_trajectory(ti = 2500, tf = 3000, rand = True,
                                        save = False )
print(T_phi)

""""""""""""""""""""" PLOT FITS """""""""""""""""
ll_esp_theta = []
zp = zip(enumerate(ll_signal),l_gamma_div, l_logP_div, ll_area, ll_obs_phi,
                    ll_idx_cell_cycle_start, ll_idx_peak)
for ((idx, signal), gamma, logP, area, l_obs_phi,
                                    l_idx_cell_cycle_start, l_idx_peak) in zp:
    plt_result = PlotResults(gamma, l_var, signal_model, signal, waveform = W,
                            logP = logP, temperature = temperature, cell = cell)
    E_model, E_theta, E_A, E_B \
        = plt_result.plotEverythingEsperance(True, idx,
                                             l_obs_phi = ll_obs_phi[idx])
    ll_esp_theta.append(E_theta)
    if idx<20:
        plt_result.plot3D( save = True, index_trace = idx)
        plt_result.plotEsperancePhaseSpace(
                np.array(ll_idx_obs_phi[idx])/N_phi*(2*np.pi),
                save = True, index_trace = idx,
                tag = str(period), attractor = (l_theta, l_phi ) )




""""""""""""""""""""" PLOT COUPLING AND PHASE SPACE DENSITY """""""""""""""""
plot_phase_space_density(l_var, l_gamma_div, ll_idx_obs_phi, F_superimpose = F,
                         save = True, cmap = bwr, temperature = temperature,
                         cell = cell, period = period,
                         attractor = (l_theta, l_phi ) )

""""""""""""""""""""" RECORD PHASES IN A TXT FILE """""""""""""""""
longest_trace = np.max([len(trace) for trace in ll_signal])
with open('Results/RawPhases/traces_theta_'+str(cell)+'_'+str(temperature)\
                                    +'_'+str(period)+'.txt', 'w') as f_theta:
    with open('Results/RawPhases/traces_phi_'+str(cell)+'_'+str(temperature)\
                                        +'_'+str(period)+'.txt', 'w') as f_phi:

        zp = zip(enumerate(ll_signal), ll_esp_theta, ll_obs_phi)
        for (idx, signal) , l_esp_theta, l_obs_phi in zp:
            f_theta.write(str(idx))
            f_phi.write(str(idx))
            for s in np.concatenate((l_esp_theta,[-1]\
                                                *(longest_trace-len(signal)))):
                f_theta.write('\t'+str(s*2*np.pi) if s!=-1 else '\t'+str(s))
            for s in np.concatenate((l_obs_phi,[-1]\
                                                *(longest_trace-len(signal)))):
                f_phi.write('\t'+str(s))
            f_theta.write('\n')
            f_phi.write('\n')
