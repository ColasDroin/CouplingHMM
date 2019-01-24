# -*- coding: utf-8 -*-
""""""""""""""""""""" WARNING : DEPRECATED FILE """""""""""""""""""""
''' This file was initially used to compute the circadian speed with H2B cells,
to show that there is a deceleration around mitosis'''
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
print('This file is deprecated!')
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
from os import listdir
from os.path import isfile, join


#Import internal modules
sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

from Classes.LoadData import LoadData
from Classes.PlotResults import PlotResults
from Classes.HMM_SemiCoupled import HMM_SemiCoupled
import Classes.EM as EM

from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.signal_model import signal_model

np.set_printoptions(threshold=np.nan)

""""""""""""""""""""" LOAD TRACES AND ANNOTATION """""""""""""""""""""
mypath = '../Data/H2B.3T3.37.2017-12-12/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) \
                                                  if f!='README' and f[0]=='t']

ll_t = []
ll_signal = []
ll_peak = []
ll_neb = []
ll_cc_start = []
ll_nan_circadian_factor = []
ll_idx_cell_cycle_start =[]
l_idx_trace = []

for file in onlyfiles:
    l_t = []
    l_signal = []
    l_peak = []
    l_neb = []
    l_cc_start = []
    with open(mypath+file, 'r') as f:
        for idx, line in enumerate(f):
            if idx==0:  continue
            t, signal, h2b, area, peak, cc_start, neb, pro, met, cyto = \
                                                                    line.split()
            if True:#float(t)%0.5==0:
                l_signal.append(float(signal))
                l_t.append(float(t))
                l_peak.append(int(peak))
                l_cc_start.append(int(cc_start))
                l_neb.append(int(neb))

        l_signal = np.array(l_signal) - np.percentile(l_signal, 5)
        l_signal = l_signal/np.percentile(l_signal, 95)

    garbage, idx_trace, garbage = file.split('.')
    if np.sum(l_cc_start)>=2:
        l_idx_trace.append(int(idx_trace))
        ll_t.append(l_t)
        ll_signal.append(l_signal)
        ll_peak.append(l_peak)
        ll_neb.append(l_neb)
        ll_cc_start.append(l_cc_start)
        ll_peak.append(l_peak)


""" NaN FOR CIRCADIAN SIGNAL """
for l_mitosis_start, l_cell_start, l_peak in zip(ll_neb, ll_cc_start, ll_peak):
    l_temp = [False]*len(l_mitosis_start)
    NaN = False
    for ind, (m, c) in enumerate(zip(l_mitosis_start, l_cell_start)):
        if m==1:
            NaN = True
        if c==1:
            NaN = False
        if NaN:
            try:
                l_temp[ind-1] = True #TO BE REMOVED POTENTIALLY
            except:
                pass
            try:
                l_temp[ind+1] = True
            except:
                pass
            l_temp[ind] = True
    ll_nan_circadian_factor.append(  l_temp   )


""" COMPUTE IDX PHI """
for (idx, l_mitosis_start), l_cell_cycle_start in zip(enumerate(ll_neb),
                                                      ll_cc_start) :
    l_idx_cell_cycle_start = [idx for idx, i in enumerate(l_cell_cycle_start) \
                                                                        if i==1]
    ll_idx_cell_cycle_start.append(l_idx_cell_cycle_start)

""" GET PHI OBS """
ll_obs_phi = []
for l_signal, l_idx_cell_cycle_start in zip(ll_signal, ll_idx_cell_cycle_start):
    l_obs_phi = [-1]*l_idx_cell_cycle_start[0]
    first = True
    for idx_div_1, idx_div_2 in zip(l_idx_cell_cycle_start[:-1],
                                    l_idx_cell_cycle_start[1:]):
        if not first:
            del l_obs_phi[-1]
        l_obs_phi.extend(   [i%(2*np.pi) for i in \
                                np.linspace(0,2*np.pi,idx_div_2-idx_div_1+1)])
        first = False

    #l_obs_phi.append(0) #first phase avec the last spike
    l_obs_phi.extend( [-1] *  (len(l_signal)-l_idx_cell_cycle_start[-1]-1))
    ll_obs_phi.append(l_obs_phi)


""""""""""""""""""""" LOAD OPTIMIZED PARAMETERS """""""""""""""""
path = '../Parameters/Real/opt_parameters_div_'+str(37)+"_"+'NIH3T3'+'.p'
with open(path, 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)
    dt = l_parameters[0] = 1/6

""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
display_parameters_from_file(path, show = True)

""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""
theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
l_var = [theta_var_coupled, amplitude_var, background_var]

""""""""""""""""""""" CREATE AND RUN HMM """""""""""""""""
hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian, ll_obs_phi,
                    waveform = W, ll_nan_factor = ll_nan_circadian_factor,
                    pi = pi, crop = True )
l_gamma_div, l_logP_div = hmm.run(project = False)


""""""""""""""""""""" CROP SIGNALS FOR PLOTTING """""""""""""""""
l_first = [[it for it, obj in enumerate(l_obs_phi) if obj!=-1][0] \
                                                    for l_obs_phi in ll_obs_phi]
l_last = [[len(l_obs_phi)-it-1 for it, obj in enumerate(l_obs_phi[::-1]) \
                                    if obj!=-1][0] for l_obs_phi in ll_obs_phi]
ll_signal = [l_signal[first:last+1] for l_signal, first, last \
                                            in zip(ll_signal, l_first, l_last)]
ll_obs_phi = [l_obs_phi[first:last+1] for l_obs_phi, first, last \
                                        in zip(ll_obs_phi, l_first, l_last)]
ll_idx_cell_cycle_start = [ [v for v in l_idx_cell_cycle_start \
                            if v>=first and v<=last  ] \
                            for l_idx_cell_cycle_start, first, last \
                            in zip(ll_idx_cell_cycle_start, l_first, l_last)]


""""""""""""""""""""" COMPUTE CIRCADIAN SPEED """""""""""""""""
zp = zip(l_idx_trace, ll_signal,l_gamma_div, l_logP_div, ll_obs_phi,
         ll_idx_cell_cycle_start)
for idx, signal, gamma, logP,  l_obs_phi, l_idx_cell_cycle_start in zp:
    plt_result = PlotResults(gamma, l_var, signal_model, signal, waveform = W,
                            logP = logP)
    l_E_model, l_E_theta, l_E_A, l_E_B = \
                                  plt_result.plotEverythingEsperance(False, idx)
    with open('../Results/H2B/'+str(idx)+'.txt', 'w') as f:
        f.write('PhaseTheta' + '\t' + 'PhasePhi' + '\n')
        for E_theta, phi in zip(l_E_theta, l_obs_phi):
            f.write(str(E_theta*2*np.pi)+ '\t' + str(phi) + '\n')
