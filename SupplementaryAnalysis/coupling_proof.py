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

#Import internal modules
sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
from Classes.LoadData import LoadData
from Functions.create_hidden_variables import create_hidden_variables
from Classes.HMM_SemiCoupled import HMM_SemiCoupled

np.set_printoptions(threshold=np.nan)
""""""""""""""""""""" FUNCTION """""""""""""""""""""
def test_coupling(cell = 'NIH3T3', temperature = 37, test = 'cycle',
                  n_bins = 15, nb_traces = 500):
    """
    Compute the phase distribution of the cell-cycle/circadian clock for a given
    circadian/cell-cycle phase, and compute the period distribution of the
    cell-cycle/circadian clock for a given starting circadian/cell-cycle phase.

    Parameters
    ----------
    cell : string
        Cell conditionself.
    temperature : integer
        Temperature condition.
    test : string
        'cycle' or clock, depending in which direction the coupling wants to be
        test.
    n_bins : int
        How many bins to divide the phase domain in.
    nb_traces : int
        How many traces to run the experiment on.
    """
    ### LOAD DATA ###
    if cell == 'NIH3T3':
        path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    else:
        path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
    dataClass=LoadData(path, nb_traces, temperature = temperature,
                        division = True, several_cell_cycles = False)
    (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
    ll_idx_cell_cycle_start, T_theta, T_phi) = \
                                        dataClass.load(load_annotation = True)
    print(len(ll_signal), " traces in the dataset")

    ### KEEP ONLY REQUIRED TRACES ###
    if test=='cycle':
        #print(ll_idx_cell_cycle_start[0])
        to_keep = [idx for idx, l_idx_cc in enumerate(ll_idx_cell_cycle_start) \
                                                        if len(l_idx_cc)>=2  ]
    elif test=='clock':
        l_first = [[it for it, obj in enumerate(l_obs_phi) if obj!=-1][0] \
                                                    for l_obs_phi in ll_obs_phi]
        l_last =[[len(l_obs_phi)-it-1 for it, obj in enumerate(l_obs_phi[::-1])\
                                    if obj!=-1][0] for l_obs_phi in ll_obs_phi]
        ll_peak_after_crop =[[idx for idx, x in enumerate(l_peak[first:last+1])\
               if x>0] for l_peak, first, last in zip(ll_peak, l_first, l_last)]
        to_keep = [idx for idx, l_idx_cc in enumerate(ll_peak_after_crop) \
                                                            if len(l_idx_cc)>=2]
    else:
        print("Either clock or cycle can be tested")
    ll_signal = [ll_signal[i] for i in to_keep]
    ll_area = [ll_area[i] for i in to_keep]
    ll_nan_circadian_factor = [ll_nan_circadian_factor[i] for i in to_keep]
    ll_obs_phi = [ll_obs_phi[i] for i in to_keep]
    ll_peak = [ll_peak[i] for i in to_keep]
    ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] for i in to_keep]
    ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v>0] \
                                                          for l_peak in ll_peak]

    print(len(ll_signal), " traces kept")

    ### LOAD PARAMETERS ###
    with open('../Parameters/Real/opt_parameters_nodiv_'+str(temperature)\
              +"_"+cell+'.p', 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)

    ##################### CREATE HIDDEN VARIABLES ###################
    theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
    l_var = [theta_var_coupled, amplitude_var, background_var]
    domain_theta = theta_var_coupled.domain
    domain_phi = theta_var_coupled.codomain

    ##################### CREATE AND RUN HMM ###################
    hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian, ll_obs_phi,
                        waveform = W, ll_nan_factor = ll_nan_circadian_factor,
                        pi = pi, crop = True )
    l_gamma_div, l_logP_div = hmm.run(project = False)

    ##################### REMOVE BAD TRACES ###################
    Plim = np.percentile(l_logP_div, 10)
    idx_to_keep = [i for i, logP in enumerate(l_logP_div) if logP>Plim ]
    l_gamma_div = [l_gamma_div[i] for i in idx_to_keep]
    ll_signal = [ll_signal[i] for i in idx_to_keep]
    ll_area = [ll_area[i] for i in idx_to_keep]
    l_logP_div = [l_logP_div[i] for i in idx_to_keep]
    ll_obs_phi = [ll_obs_phi[i] for i in idx_to_keep]
    ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] for i in idx_to_keep]
    print("Kept traces with div: ", len(idx_to_keep))


    ### CROP SIGNAL ###
    l_first = [[it for it, obj in enumerate(l_obs_phi) if obj!=-1][0] \
                                                    for l_obs_phi in ll_obs_phi]
    l_last = [[len(l_obs_phi)-it-1 for it, obj in enumerate(l_obs_phi[::-1]) \
                                    if obj!=-1][0] for l_obs_phi in ll_obs_phi]
    ll_signal = [l_signal[first:last+1] for l_signal, first, last \
                                            in zip(ll_signal, l_first, l_last)]
    ll_area = [l_area[first:last+1] for l_area, first, last \
                                              in zip(ll_area, l_first, l_last)]
    ll_obs_phi = [l_obs_phi[first:last+1] for l_obs_phi, first, last \
                                            in zip(ll_obs_phi, l_first, l_last)]
    ll_peak = [l_peak[first:last+1] for l_peak, first, last \
                                            in zip(ll_peak, l_first, l_last)]
    ll_idx_peak = [  [idx for idx, i in enumerate(l_peak) if i==1] \
                                                        for l_peak in ll_peak]
    ll_idx_cc_start = [np.array(l_idx_cell_cycle_start)-first \
                            for l_idx_cell_cycle_start, first, last \
                            in zip(ll_idx_cell_cycle_start, l_first, l_last)]



    ### COMPUTE THE DISTRIBUTION OF CIRCADIAN PHASES AT PHI = 0 ###
    ll_phase_clock = []
    for gamma in l_gamma_div:
        l_phase=[]
        for gamma_t in gamma:
            p_phase = np.sum(gamma_t, axis = (1,2))
            phase = np.angle(np.sum(np.multiply(p_phase,
                                            np.exp(1j*domain_theta))))%(2*np.pi)
            l_phase.append(phase)
        ll_phase_clock.append(l_phase)


    #keep only traces with at least a full cycle

    if test=="cycle":
        ll_idx = ll_idx_cc_start
        ll_phase = ll_phase_clock
    elif test=="clock":
        ll_idx = ll_idx_peak
        ll_phase = ll_obs_phi
    else:
        print("Either clock or cycle can be tested")

    l_nb_cc = [ (idx, len(l)) for idx, l in enumerate(ll_idx) if len(l)>1]

    #plot circadian ditribution at mitosis, or cycle distribution at peak
    l_phase_mitosis = []
    for idx_corr, (idx, nb_cc) in enumerate(l_nb_cc):
        for i in range(nb_cc-1):
            l_phase_mitosis.append( ll_phase[idx][ll_idx[idx][i]]    )


    df1 = pd.DataFrame({})

    n, bins, patches = plt.hist(l_phase_mitosis, bins = n_bins,
                                normed = False, color = 'steelblue', alpha=0.5)
    df1['g1'] = pd.Series(l_phase_mitosis)
    hist, edges = np.histogram(l_phase_mitosis)
    max_val = max(hist)
    plt.ylim([0, max_val])
    plt.xlim([0, 2*np.pi+0.01])
    #plt.title("N = " + str(len(l_phase_mitosis)))
    if test=="cycle":
        plt.xlabel('Circadian phase at mitosis')
        plt.savefig('../Results/TestCoupling/'+cell+'_'\
                    +str(temperature)+'_circadian_phase_at_mitosis.pdf')
    else:
        plt.xlabel('Cell-cycle phase at circadian peak')
        plt.savefig('../Results/TestCoupling/'+cell+'_'\
                    +str(temperature)+'_cell_phase_at_peak.pdf')
    plt.ylabel('Frequency')
    plt.show()
    plt.close()

    ### ASSOCIATE EACH TRACE TO A GIVEN PHASE AT MITOSIS/PEAK ###
    d_idx_traces_uniform = {b:[] for b in bins[:-1]}
    for (idx, nb_cc) in l_nb_cc:
        for i in range(nb_cc-1):
            for b1, b2 in zip(bins[:-1], bins[1:]):
                c1 = ll_phase[idx][ll_idx[idx][i]] >=b1
                c2 = ll_phase[idx][ll_idx[idx][i]] <b2
                if c1 and c2:
                    d_idx_traces_uniform[b1].append(  (idx,i)  )
                    break


    ### COMPUTE DISTRIBUTION OF PERIODS DEPENDING ON THE INITIAL PHASE ###
    l_mean = []
    l_std = []
    for b, l_idx_to_keep in d_idx_traces_uniform.items():
        l_period = []
        for idx, i in l_idx_to_keep:
            l_period.append(  (ll_idx[idx][i+1]-ll_idx[idx][i])/2     )
        l_mean.append(np.mean(l_period))
        l_std.append(np.std(l_period))

    objects = [str( ((b1+b2)/2)/(2*np.pi)  )[:3] for b1,b2 in zip(bins[:-1],
                                                                    bins[1:])]
    y_pos = range(len(objects)) #np.linspace(0,3,(len(objects)))
    plt.bar(y_pos,  l_mean, yerr=l_std, color = 'steelblue', alpha=0.5)
    plt.xticks(y_pos, objects)
    if test=='cycle':
        plt.ylabel('Cell-cycle period')
        plt.xlabel('Circadian phase at mitosis')
        plt.savefig('../Results/TestCoupling/'+cell+'_'+str(temperature)\
                +'_Period_cycle_distribution_depending_on_circadian_phase.pdf')
    else:
        plt.ylabel('Circadian period')
        plt.xlabel('Cell-cycle phase at previous circadian peak')
        plt.savefig('../Results/TestCoupling/'+cell+'_'+str(temperature)\
                +'_Period_clock_distribution_depending_on_cycle_phase.pdf')
    plt.show()
    plt.close()


""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    test_coupling(cell = 'U2OS', temperature = 37, test = 'cycle',
                    n_bins = 10, nb_traces = 50)
    test_coupling(cell = 'U2OS', temperature = 37, test = 'clock',
                   n_bins = 20, nb_traces = 100)
