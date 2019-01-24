# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import kurtosis

### Import internal modules
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('Classes'))

from LoadData import LoadData

""""""""""""""""""""" FUNCTIONS """""""""""""""""""""
def estimate_phase_dev_from_div(ll_idx_div):
    """
    Estimate phase deviation from a list of division positions, using an inverse
    gaussian.

    Parameters
    ----------
    ll_idx_div : list
        For each trace, list of indeces of divisions.

    Returns
    -------
    The standard deviation for the phase progression, and the periods.
    """
    ################## COMPUTE CIRCADIAN CLOCK PP DISTRIBUTION ##############
    l_T_cycle=[]
    for l_idx_div in ll_idx_div:
        for t_div_1, t_div_2 in zip(l_idx_div[:-1], l_idx_div[1:]):
            #remove outliers
            T = (t_div_2-t_div_1)/2
            if T<12 or T>38:
                #double or missing annotation
                pass
            else:
                l_T_cycle.append(T )

    std_period = np.std(l_T_cycle)
    mean_period = np.mean(l_T_cycle)
    std_phase = std_period*2*np.pi/mean_period**(3/2)

    return std_phase, std_period



def estimate_phase_dev_from_signal(ll_peak):
    """
    Estimate phase deviation from a list of peaks positions, using an inverse
    gaussian.

    Parameters
    ----------
    ll_peak : list
        For each trace, list of indexes of circdian peaks.

    Returns
    -------
    The standard deviation for the phase progression, and the periods.
    """
    ################## COMPUTE CIRCADIAN CLOCK PP DISTRIBUTION ###############
    l_T_clock=[]
    for l_peak in ll_peak:
        l_idx_peak = [idx for idx, i in enumerate(l_peak) if i==1]
        for t_peak_1, t_peak_2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
            #remove outliers
            T = (t_peak_2-t_peak_1)/2
            if T<12 or T>38:
                #double or missing annotation
                pass
            else:
                l_T_clock.append(T )

    ##################### COMPUTE PHASE DEVIATION ##################
    std_period = np.std(l_T_clock)
    mean_period = np.mean(l_T_clock)
    std_phase = std_period*2*np.pi/mean_period**(3/2)

    return std_phase, std_period


def compute_phase_variance_with_confidence(ll_peak):
    """
    Alternative estimate for the phase deviation from a list of peaks positions.

    Parameters
    ----------
    ll_peak : list
        For each trace, list of indexes of circdian peaks.

    Returns
    -------
    The standard deviation for the phase progression, and the corresponding
    confidence interval.
    """
    ################## COMPUTE CIRCADIAN CLOCK PP DISTRIBUTION ###############
    l_T_clock=[]
    for l_peak in ll_peak:
        l_idx_peak = [idx for idx, i in enumerate(l_peak) if i==1]
        for t_peak_1, t_peak_2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
            #remove outliers
            T = (t_peak_2-t_peak_1)/2
            if T<12 or T>38:
                #double or missing annotation
                pass
            else:
                l_T_clock.append(T )

    ##################### COMPUTE PHASE VARIANCE ##################
    var_period = np.var(l_T_clock)
    mean_period = np.mean(l_T_clock)
    var_phase = var_period*4*np.pi**2/mean_period**3

    ##################### COMPUTE PHASE VARIANCE CONFIDENCE ##################
    var_var_period = var_period**4/len(l_T_clock)*(kurtosis(l_T_clock)-1\
                                                          +2/(len(l_T_clock)-1))
    var_var_phase = var_var_period * 16 * np.pi**4 / mean_period**6

    return var_phase, var_var_phase



def estimate_phase_dev_from_div_signal(ll_peak, ll_idx_cell_cycle_start):
    """
    Alternative estimate for the phase deviation from a list of peaks positions,
    removing dpd intervals.

    Parameters
    ----------
    ll_peak : list
        For each trace, list of indexes of circdian peaks.
    ll_peak : list
        For each trace, list of indexes of the start of mitosis.

    Returns
    -------
    The standard deviation for the phase progression, and the periods.
    """
    ######## COMPUTE CIRCADIAN CLOCK PP DISTRIBUTION REMOVING PDP ########
    l_T_clock=[]
    ll_idx_peak = [[idx for idx, i in enumerate(l_peak) if i==1] \
                                                          for l_peak in ll_peak]
    for l_idx_peak, l_idx_cc in zip(ll_idx_peak, ll_idx_cell_cycle_start):

        l_t_p = set([(p1,p2) for p1,p2 in zip(l_idx_peak[:-1], l_idx_peak[1:])])
        #print(l_t_p)
        #print(l_idx_cc)
        for d in l_idx_cc:
            for p1, p2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
                if p1<d and d<p2:
                    l_t_p.discard(  (p1,p2)    )
        #print(l_t_p)
        for (p1,p2) in l_t_p:
            l_T_clock.append( (p2-p1)/2 )

    ##################### COMPUTE PHASE DEVIATION ##################
    std_period = np.std(l_T_clock)
    mean_period = np.mean(l_T_clock)
    std_phase = std_period*2*np.pi/mean_period**(3/2)

    return std_phase, std_period

def estimate_phase_dev(cell, temperature):
    """
    Final estimate for the circadian phase deviation.

    Parameters
    ----------
    cell : string
        Cell condition.
    temperature : int
        Temperature condition.

    Returns
    -------
    The standard deviation for the phase progression, and the periods.
    """

    ######### CORRECTION BECAUSE NOT ENOUGH TRACES AT 34°C AND 40°C #########
    print('CAUTION : Parameters for None temperature selected since not enough \
                                                       traces at 34°C and 40°C')
    temperature = None

    ##################### LOAD DATA ##################
    if cell == 'NIH3T3':
        path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
        dataClass=LoadData(path, 10000000, temperature = temperature,
                            division = False)
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, T_phi) = \
                                            dataClass.load(load_annotation=True)
        #print(len(ll_area))
        std, std_T = estimate_phase_dev_from_signal(ll_peak)

    elif cell == 'U2OS':
        path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
        dataClass=LoadData(path, 10000000, temperature = temperature,
                            division = True)
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, T_phi) = \
                                            dataClass.load(load_annotation=True)
        std, std_T = estimate_phase_dev_from_signal(ll_peak)
        #correction for the neglected coupling since dividing traces
        std = std*0.65
        std_T  = std_T *0.65

    else:
        print("Cell type doesn't exist")

    '''
    for (idx, l_signal), l_peak in zip(enumerate(ll_signal), ll_peak):
        plt.plot(l_signal)
        plt.plot(l_peak)
        plt.show()
        plt.close()
        if idx>17:
            break
    '''
    return std, std_T

def estimate_cycle_dev(cell, temperature):
    """
    Final estimate for the cell-cycle phase deviation.

    Parameters
    ----------
    cell : string
        Cell condition.
    temperature : int
        Temperature condition.

    Returns
    -------
    The standard deviation for the phase progression, and the periods.
    """
    ##################### LOAD DATA ##################
    if cell == 'NIH3T3':
        path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    elif cell == 'U2OS':
        path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
    dataClass=LoadData(path, 10000000, temperature = temperature,
                        division = True)
    (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak, \
    ll_idx_cell_cycle_start, T_theta, T_phi) = \
                                            dataClass.load(load_annotation=True)
    std, std_T = estimate_phase_dev_from_div(ll_idx_cell_cycle_start)

    return std, std_T


def show_no_temperature_difference():
    """
    Plot the estimate of the phase at different temperatures.
    """
    l_var = []
    l_var_var = []
    l_temp = [34,37,40]
    for temperature in l_temp:
        path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
        dataClass=LoadData(path, 10000000, temperature = temperature,
                            division = False)
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak, \
        ll_idx_cell_cycle_start, T_theta, T_phi) = \
                                            dataClass.load(load_annotation=True)
        var,var_var = compute_phase_variance_with_confidence(ll_peak)
        l_var.append(var)
        l_var_var.append(var_var)

    plt.errorbar(l_temp, l_var, yerr = l_var_var, fmt='o')
    plt.xlim([33,41])
    plt.xlabel("Temperature")
    plt.ylabel("Phase diffusion variance mean and deviation")
    plt.savefig('Results/RawData/var_diffusion.pdf')
    plt.show()
    plt.close()

def invgauss (x, mu, lambd):
    """
    Inverse gaussian density function.

    Parameters
    ----------
    x : float
        Point for which the probability must be estimated.
    mu : float
        Mean parameter.
    lambda : float
        Scale parameter.

    Returns
    -------
    The probability of the point x according to the specified distribution.
    """
    return (lambd / 2*np.pi*x**3)**0.5 * np.exp(-lambd * (x-mu)**2/(2*x*mu**2))


def compute_likelihood_sigma():
    """
    Compute and plot the likelihood of the phase diffusion parameter,
    depending on the temperature.
    """
    l_T = [34,37,40]
    l_likelihood_T = []
    mean_IG = 24
    domain_sigma = np.linspace(0.05, 0.3, 100)

    for T in l_T:

        path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
        dataClass=LoadData(path, 10000000, temperature = T, division = False)
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, T_phi) = \
                                            dataClass.load(load_annotation=True)

        l_T_clock=[]
        for l_peak in ll_peak:
            l_idx_peak = [idx for idx, i in enumerate(l_peak) if i==1]
            for t_peak_1, t_peak_2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
                #remove outliers
                T = (t_peak_2-t_peak_1)/2
                if T<12 or T>38:
                    #double or missing annotation
                    pass
                else:
                    l_T_clock.append(T )

        l_likelihood = []
        for sigma_theta in domain_sigma:
            lpx = np.log(1/sigma_theta)
            #lpx = 1
            for T in l_T_clock:
                lpx = lpx + np.log(invgauss(T, mean_IG,
                                            4*np.pi**2/sigma_theta**2))
            l_likelihood.append( lpx/len(l_T_clock))
        l_likelihood_T.append(l_likelihood)


    plt.plot(domain_sigma,l_likelihood_T[0], c = 'red', label = '34' )
    plt.plot(domain_sigma,l_likelihood_T[1], c = 'blue',label = '37' )
    plt.plot(domain_sigma,l_likelihood_T[2], c = 'orange',label = '40' )
    plt.axvline(domain_sigma[np.argmax(l_likelihood_T[0])], c= 'red')
    plt.axvline(domain_sigma[np.argmax(l_likelihood_T[1])], c= 'blue')
    plt.axvline(domain_sigma[np.argmax(l_likelihood_T[2])], c= 'orange')


    plt.ylabel(r'$log(L(\sigma_\theta))$')
    plt.xlabel(r'$\sigma_\theta$' )
    plt.legend()
    plt.savefig('Results/RawData/likelihood_sigma_theta.pdf')
    plt.show()
    plt.close()




""""""""""""" TEST """""""""""""
if __name__ == '__main__':
    os.chdir('..')

    #show_no_temperature_difference()
    compute_likelihood_sigma()



    std_34, std_T_34 = estimate_phase_dev(cell = "NIH3T3", temperature = 34)
    print("theta NIH3T3 34°C no coupling:", std_34, std_T_34)
    std_37 , std_T_37 = estimate_phase_dev(cell = "NIH3T3", temperature = 37)
    print("theta NIH3T3 37°C no coupling:", std_37, std_T_37)
    std_40, std_T_40  = estimate_phase_dev(cell = "NIH3T3", temperature = 40)
    print("theta NIH3T3 40°C no coupling:", std_40, std_T_40)


    plt.errorbar([34,37,40], [std_34, std_37, std_40], fmt='o')
    plt.xlabel("Temperature")
    plt.ylabel("Circadian phase deviation")
    plt.savefig('Results/RawData/Phase_theta_std_NIH3T3.pdf')
    plt.show()
    plt.close()


    std_34, std_T_34  = estimate_phase_dev(cell = "U2OS", temperature = 34)
    print("theta U20S 34°C no coupling:", std_34, std_T_34)
    std_37, std_T_37  = estimate_phase_dev(cell = "U2OS", temperature = 37)
    print("theta U20S 37°C no coupling:", std_37, std_T_37)
    std_40, std_T_40  = estimate_phase_dev(cell = "U2OS", temperature = 40)
    print("theta U20S 40°C no coupling:", std_40, std_T_40)


    plt.errorbar([34,37,40], [std_34, std_37, std_40], fmt='o')
    plt.xlabel("Temperature")
    plt.ylabel("Circadian phase deviation")
    plt.savefig('Results/RawData/Phase_theta_std_U20S.pdf')
    plt.show()
    plt.close()

    std_34, std_T_34  = estimate_cycle_dev(cell = "NIH3T3", temperature = 34)
    print("phi NIH3T3 34°C no coupling:", std_34, std_T_34)
    std_37, std_T_37  = estimate_cycle_dev(cell = "NIH3T3", temperature = 37)
    print("phi NIH3T3 37°C no coupling:", std_37, std_T_37)
    std_40, std_T_40  = estimate_cycle_dev(cell = "NIH3T3", temperature = 40)
    print("phi NIH3T3 40°C no coupling:", std_40, std_T_40)

    plt.errorbar([34,37,40], [std_34, std_37, std_40], fmt='o')
    plt.xlabel("Temperature")
    plt.ylabel("Cell-cycle phase deviation")
    plt.savefig('Results/RawData/Phase_phi_std_NIH3T3.pdf')
    plt.show()
    plt.close()


    std_34, std_T_34  = estimate_cycle_dev(cell = "U2OS", temperature = 34)
    print("phi U20S 34°C no coupling:", std_34, std_T_34)
    std_37, std_T_37  = estimate_cycle_dev(cell = "U2OS", temperature = 37)
    print("phi U20S 37°C no coupling:", std_37, std_T_37)
    std_40, std_T_40  = estimate_cycle_dev(cell = "U2OS", temperature = 40)
    print("phi U20S 40°C no coupling:", std_40, std_T_40)

    plt.errorbar([34,37,40], [std_34, std_37, std_40], fmt='o')
    plt.xlabel("Temperature")
    plt.ylabel("Cell-cycle phase deviation")
    plt.savefig('Results/RawData/Phase_phi_std_U20S.pdf')
    plt.show()
    plt.close()
