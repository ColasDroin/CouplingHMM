# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import interp1d
import seaborn as sn

### Import internal modules
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))
sys.path.insert(0, os.path.realpath('Classes'))
sys.path.insert(0, os.path.realpath('Functions'))
from clean_waveform import clean_waveform
from LoadData import LoadData

#nice plotting style
sn.set_style("whitegrid", { 'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

""""""""""""""""""""" FUNCTIONS """""""""""""""""""""
def running_mean(l, N):
    """
    Compute a moving average on list l with filter of size N.

    Parameters
    ----------
    l : list
        Signal on which the moving average is computed.
    N : integer
        Filter size.

    Returns
    -------
    The smoothed signal.
    """
    sum = 0
    result = list( 0 for x in l)

    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)

    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N

    return result[int(N/2):]+result[-int(N/2):]

def compute_intervals_no_div(ll_signal, ll_peak, domain_theta):
    """
    Extract a list of peak-to-peak intervals.

    Parameters
    ----------
    ll_signal : list
        List of list of traces.
    ll_peak : list
        List of list of peak indexes.
    domain_theta : array
        Domain of the circadian phase.

    Returns
    -------
    A list of peak-to-peak intervals.
    """

    l_w = []
    for l_signal, l_peak in zip(ll_signal, ll_peak):

        l_idx_peak = [idx for idx, i in enumerate(l_peak) if i==1]
        for t_peak_1, t_peak_2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
            try:
                s = l_signal[t_peak_1:t_peak_2+1]
            except:
                s = l_signal[t_peak_1:t_peak_2]
            s = interp1d(np.linspace(0,2*np.pi, len(s), endpoint = True), s)

            l_w.append(s(domain_theta))

    return l_w


def compute_intervals_div(ll_signal, ll_peak, ll_idx_cell_cycle_start,
                          domain_theta):
    """
    Extract a list of peak-to-peak intervals with division in-between.

    Parameters
    ----------
    ll_signal : list
        List of list of traces.
    ll_peak : list
        List of list of peak indexes.
    ll_idx_cell_cycle_start : list
        List of list of mitosis start indexes.
    domain_theta : array
        Domain of the circadian phase.

    Returns
    -------
    A list of peak-to-peak intervals.
    """
    #look for pp events
    ll_idx_peak = [[idx for idx, i in enumerate(l_peak) if i==1]\
                                                          for l_peak in ll_peak]
    l_w = []
    for l_idx_peak, l_idx_cc, l_signal in zip(ll_idx_peak,
                                              ll_idx_cell_cycle_start,
                                              ll_signal):

        l_t_p = set([(p1,p2) for p1,p2 in zip(l_idx_peak[:-1], l_idx_peak[1:])])
        #print(l_t_p)
        #print(l_idx_cc)
        for d in l_idx_cc:
            for p1, p2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
                if p1<d and d<p2:
                    l_t_p.discard(  (p1,p2)    )
        #print(l_t_p)
        for (p1,p2) in l_t_p:
            try:
                s = l_signal[p1+1:p2+1]
            except:
                s = l_signal[p1+1:p2]
            s = interp1d(   np.linspace(0,2*np.pi, len(s), endpoint = True), s)
            l_w.append(s(domain_theta))
    return l_w

def estimate_waveform_from_signal(ll_signal, ll_peak, domain_theta,
                                  ll_idx_cell_cycle_start = []):
    """
    Given a list of signal, extract peak-to-peak intervals and estimate waveform
    from them.

    Parameters
    ----------
    ll_signal : list
        List of list of traces.
    ll_peak : list
        List of list of peak indexes.
    domain_theta : array
        Domain of the circadian phase.
    ll_idx_cell_cycle_start : list
        List of list of mitosis start indexes.

    Returns
    -------
    Waveform as a list.
    """

    #smooth signal
    ll_signal = [running_mean(l_signal,3) for l_signal in ll_signal]


    if len(ll_idx_cell_cycle_start)==0:
        l_w = compute_intervals_no_div(ll_signal, ll_peak, domain_theta)
    else:
        l_w = compute_intervals_div(ll_signal, ll_peak, ll_idx_cell_cycle_start,
                                    domain_theta)

    #l_w = compute_intervals_no_div(ll_signal, ll_peak, domain_theta)

    W = np.mean(np.array(l_w), axis = 0)
    W_std = np.std(np.array(l_w), axis = 0)

    W = clean_waveform(W)

    return W

def estimate_waveform(cell, temperature, domain_theta):
    """
    Given a list of conditions, get a list of corresponding traces, extract
    peak-to-peak intervals and estimate waveform from them.

    Parameters
    ----------
    cell : string
        Cell condition.
    temperature : int
        Temperature condition.
    domain_theta : array
        Domain of the circadian phase.
    Returns
    -------
    Waveform as a list.
    """

    ######### CORRECTION BECAUSE NOT ENOUGH TRACES AT 34°C AND 40°C #########
    print('CAUTION : Parameters for None temperature selected since not enough \
           traces at 34°C and 40°C')
    temperature = None

    if not cell=='all':
        ##################### LOAD DATA ##################
        if cell == 'NIH3T3':
            path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
            dataClass=LoadData(path, 10000000, temperature = temperature,
                               division = False)
        elif cell == 'U2OS':
            path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
            dataClass=LoadData(path, 1000000, temperature = 37, division = True)
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
         ll_idx_cell_cycle_start, T_theta, T_phi) \
                                          = dataClass.load(load_annotation=True)
        ##################### ESTIMATE WAVEFORM ##################
        return estimate_waveform_from_signal(ll_signal, ll_peak, domain_theta,
                                             ll_idx_cell_cycle_start)
    else:
        W1 = estimate_waveform('U2OS', temperature, domain_theta)
        W2 = estimate_waveform('NIH3T3', temperature, domain_theta)
        W = clean_waveform( (W1+W2)/2 )
        return W


""""""""""""" TEST """""""""""""
if __name__ == '__main__':

    N_theta = 48
    domain_theta = np.linspace(0,2*np.pi, N_theta, endpoint = False)
    os.chdir('..')
    print("***NIH3T3***")
    print("34°C no coupling:")
    W_34 = estimate_waveform(cell = "NIH3T3", temperature = 34,
                             domain_theta = domain_theta)
    print("37°C no coupling:")
    W_37 = estimate_waveform(cell = "NIH3T3", temperature = 37,
                             domain_theta = domain_theta)
    print("40°C no coupling:")
    W_40 = estimate_waveform(cell = "NIH3T3", temperature = 40,
                             domain_theta = domain_theta)
    print("all T no coupling:")
    W_None = estimate_waveform(cell = "NIH3T3", temperature = None,
                               domain_theta = domain_theta)


    print("***U2OS***")
    print("37°C:")
    W_37_U2OS = estimate_waveform(cell = "U2OS", temperature = 37,
                                  domain_theta = domain_theta)

    print("***U2OS***")
    print("all T:")
    W_None_U2OS = estimate_waveform(cell = "U2OS", temperature = None,
                                    domain_theta = domain_theta)



    print("***all***")
    print("None")
    W_None_all = estimate_waveform(cell = "all", temperature = 34,
                                   domain_theta = domain_theta)


    ######### PLOT RESULT #########
    cell = "NIH3T3"
    temperature = None
    plt.plot(domain_theta, W_34, label = '34°C')
    plt.plot(domain_theta, W_37, label = '37°C')
    plt.plot(domain_theta, W_40, label = '40°C')
    plt.xlabel(r"circadian phase $\theta$")
    plt.ylabel(r"Waveform $\omega(\theta)$")
    plt.xlim((0,2*np.pi))
    plt.legend()
    plt.savefig('Results/RawData/Waveform_'+cell+'.pdf')
    plt.show()
    plt.close()

    cell = "U2OS"
    temperature = None

    plt.plot(domain_theta, W_37_U2OS, label = '37°C')
    plt.xlabel(r"circadian phase $\theta$")
    plt.ylabel(r"Waveform $\omega(\theta)$")
    plt.xlim((0,2*np.pi))
    plt.legend()
    plt.savefig('Results/RawData/Waveform_'+cell+'.pdf')
    plt.show()
    plt.close()

    cell = "all"
    temperature = None
    W_NIH = W_None#np.append(W_None[int(N_theta/2):],W_None[:int(N_theta/2)])
    W_U2 = W_None_U2OS#np.append(W_None_U2OS[int(N_theta/2):],
                                #W_None_U2OS[:int(N_theta/2)])
    W_all =W_None_all #np.append(W_None_all[int(N_theta/2):],
                                #W_None_all[:int(N_theta/2)])

    plt.plot(domain_theta/(2*np.pi), W_NIH, color = "green", label = 'NIH3T3')
    plt.plot(domain_theta/(2*np.pi), W_U2, color = 'lightblue', label = 'U2OS')
    plt.plot(domain_theta/(2*np.pi), W_all, color = 'orange', label = 'Both')
    plt.xlabel(r"circadian phase $\theta$")
    plt.ylabel(r"Waveform $\omega(\theta)$")
    plt.xlim((0,1))
    plt.legend()
    plt.savefig('Results/RawData/Waveform_'+cell+'_'+str(temperature)+'.pdf')
    plt.show()
    plt.close()
