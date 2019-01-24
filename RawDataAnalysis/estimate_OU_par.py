# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import savgol_filter

### Import internal modules
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))
sys.path.insert(0, os.path.realpath('Classes'))
sys.path.insert(0, os.path.realpath('Functions'))
from LoadData import LoadData
from peak_det import peakdet

""""""""""""""""""""" FUNCTION """""""""""""""""""""
def estimate_OU_par_from_signal(ll_signal, W, gamma_A = 0.03, gamma_B = 0.03):
    """
    Estimate mean and variance of OU processes from a given set of circadian
    traces.

    Parameters
    ----------
    W : list
        Waveform.
    gamma_A : float
        Regression parameter for the amplitude.
    gamma_b : float
        Regression parameter for the background.

    Returns
    -------
    The mean and standard deviations of the amplitude and the background.
    """

    ##################### SMOOTH SIGNAL ################
    ll_signal = [savgol_filter(l_signal, 21, 3) for l_signal in ll_signal]

    ##################### COMPUTE AMPLITUDE WAVEFORM ################
    if W is None:
        amp_waveform = 1
    else:
        amp_waveform = np.max(W)-np.min(W)

    ##################### LOOK FOR MAXIMA/MINIMA ################
    l_min_spikes = []
    l_max_spikes = []
    l_amp_spikes = []
    for i, l_signal in enumerate(ll_signal):
        l_max, l_min = peakdet(l_signal, 0.3)
        if l_max.shape[0]<2 or l_min.shape[0]<1:
            continue

        ##################### REMOVE ARTIFACT ################
        idx=0
        to_del_lmax=[]
        to_del_lmin=[]
        for t1, t2, t3 in zip(l_max[:,0],l_min[:,0], l_max[1:,0]) :
            if t3-t1<14 or abs(t1-t2)<7 or abs(t3-t2)<7:
                if l_max[idx,1]>l_max[idx+1,1]:
                    to_del_lmax.append(idx+1)
                else:
                    to_del_lmax.append(idx)
                to_del_lmin.append(idx)
            idx+=1
        l_max = [v for i,v in enumerate(list(l_max)) if i not in to_del_lmax]
        l_min = [v for i,v in enumerate(list(l_min)) if i not in to_del_lmax]

        if len(l_min)>0:
            l_min_spikes.extend( np.array(l_min)[:,1] )
        if len(l_max)>0:
            l_max_spikes.extend( np.array(l_max)[:,1] )

        if len(l_max)<2 or len(l_min)<1:
            continue

        ##################### PLOT MIN/MAX IDENTIFICATION ################
        l_max = np.array(l_max)
        l_min = np.array(l_min)
        if i<0:
            plt.scatter(l_max[:,0], l_max[:,1], color='blue')
            plt.scatter(l_min[:,0], l_min[:,1], color='red')
            plt.plot(l_signal)
            #for idx, v  in enumerate(l_peak[i]):
                #if v!=0:
                #    plt.axvline(idx)
            plt.show()
            plt.close()


        ##################### COMPUTE AMPLITUDES ################
        first_max=False
        last_max=False
        if l_max[0,0]<l_min[0,0]:
            first_max=True
        if l_max[-1,0]>l_min[-1,0]:
            last_max=True


        #create a list of spike amplitude
        l_amplitude = []
        if l_max.shape[0]>=2 and l_min.shape[0]>=2:
            if first_max:
                for min1, max1, min2 in zip(l_min[:,1], l_max[1:,1],
                                                                  l_min[1:,1]):
                    l_amplitude.append( max1-(min2+min1)/2)
            else:
                for min1, max1, min2 in zip(l_min[0:,1], l_max[0:,1],
                                                                  l_min[1:,1]):
                    l_amplitude.append( max1-(min2+min1)/2)

        elif l_max.shape[0]==1 and l_min.shape[0]==2:
            l_amplitude.append( l_max[0,1]-(l_min[1,1]+l_min[0,1])/2     )


        if i==-1:
            print(l_amplitude)
        if len(l_amplitude)>0:
            l_amp_spikes.extend(l_amplitude)

    ##################### REMOVE TOO SMALL AMPLITUDES (BUGS) ################
    l_amp_spikes = [a for a in l_amp_spikes if a>0.3]


    #compute stationnary mean and variance for amplitude
    l_log_amp_spikes = np.log(np.array(l_amp_spikes)/amp_waveform)
    var_A_stat = np.var(l_log_amp_spikes)
    std_A = (var_A_stat * 2*gamma_A)**0.5
    mu_A = np.mean(l_log_amp_spikes)


    var_B_stat = np.var(l_min_spikes)
    std_B = (var_B_stat * 2*gamma_B)**0.5
    mu_B = np.mean(l_min_spikes)

    '''
    print("mu_A =", mu_A)
    print("std_A =", std_A)
    print("mu_B =", mu_B)
    print("std_B =", std_B)
    '''

    return mu_A, std_A, mu_B, std_B

def estimate_OU_par(cell, temperature, W = None, gamma_A = 0.03, gamma_B =0.03):
    """
    Estimate mean and variance of OU processes given a set of conditions,
    according to which a set of traces is filtered.

    Parameters
    ----------
    cell : string
        Cell type.
    temperature : integer
        Temperature condition.
    W : list
        Waveform.
    gamma_A : float
        Regression parameter for the amplitude.
    gamma_b : float
        Regression parameter for the background.

    Returns
    -------
    The mean and standard deviations of the amplitude and the background.
    """
    ######### CORRECTION BECAUSE NOT ENOUGH TRACES AT 34°C AND 40°C #########
    print('CAUTION : Parameters for None temperature selected since not enough \
            traces at 34°C and 40°C')
    temperature = None

    ##################### LOAD DATA ################
    if cell == 'NIH3T3':
        path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
        dataClass=LoadData(path, 10000000, temperature = temperature,
                            division = False)
    elif cell == 'U2OS':
        path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
        dataClass=LoadData(path, 10000000, temperature = temperature,
                            division = True)

    try:
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, T_phi) = \
                                            dataClass.load(load_annotation=True)
    except:
        dataClass.path = '../'+dataClass.path
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, T_phi) = \
                                            dataClass.load(load_annotation=True)

    return estimate_OU_par_from_signal(ll_signal, W, gamma_A, gamma_B)


""""""""""""" TEST """""""""""""
if __name__ == '__main__':
    os.chdir('..')

    print("NIH3T3 34°C no coupling:")
    mu_A_34, std_A_34, mu_B_34, std_B_34 = estimate_OU_par(cell = "NIH3T3",
                                                            temperature = 34)
    print("NIH3T3 37°C no coupling:")
    mu_A_37, std_A_37, mu_B_37, std_B_37 = estimate_OU_par(cell = "NIH3T3",
                                                             temperature = 37)
    print("NIH3T3 40°C no coupling:")
    mu_A_40, std_A_40, mu_B_40, std_B_40 = estimate_OU_par(cell = "NIH3T3",
                                                             temperature = 40)


    ### PLOT RESULTS ###
    plt.errorbar([34,37,40], [mu_A_34, mu_A_37, mu_A_40],
                                yerr = [std_A_34, std_A_37, std_A_40], fmt='o')
    plt.xlim([33,41])
    plt.xlabel("Temperature")
    plt.ylabel("Amplitude mean and deviation")
    plt.savefig('Results/RawData/OU_A_NIH3T3.pdf')
    plt.show()
    plt.close()

    plt.errorbar([34,37,40], [mu_B_34, mu_B_37, mu_B_40],
                                yerr = [std_B_34, std_B_37, std_B_40], fmt='o')
    plt.xlim([33,41])
    plt.xlabel("Temperature")
    plt.ylabel("Background mean and deviation")
    plt.savefig('Results/RawData/OU_B_NIH3T3.pdf')
    plt.show()
    plt.close()


    print("U20S 34°C no coupling:")
    mu_A_34, std_A_34, mu_B_34, std_B_34 = estimate_OU_par(cell = "U2OS",
                                                            temperature = 34)
    print("U20S 37°C no coupling:")
    mu_A_37, std_A_37, mu_B_37, std_B_37 = estimate_OU_par(cell = "U2OS",
                                                            temperature = 37)
    print("U20S 40°C no coupling:")
    mu_A_40, std_A_40, mu_B_40, std_B_40 = estimate_OU_par(cell = "U2OS",
                                                            temperature = 40)


    ### PLOT RESULTS ###
    plt.errorbar([34,37,40], [mu_A_34, mu_A_37, mu_A_40],
                                yerr = [std_A_34, std_A_37, std_A_40], fmt='o')
    plt.xlim([33,41])
    plt.xlabel("Temperature")
    plt.ylabel("Amplitude mean and deviation")
    plt.savefig('Results/RawData/OU_A_U20S.pdf')
    plt.show()
    plt.close()

    plt.errorbar([34,37,40], [mu_B_34, mu_B_37, mu_B_40],
                                yerr = [std_B_34, std_B_37, std_B_40], fmt='o')
    plt.xlim([33,41])
    plt.xlabel("Temperature")
    plt.ylabel("Background mean and deviation")
    plt.savefig('Results/RawData/OU_B_U20S.pdf')
    plt.show()
    plt.close()
