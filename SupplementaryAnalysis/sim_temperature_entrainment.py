# -*- coding: utf-8 -*-
''' This file enables to simulate a system of three oscillators, to study the
circadian-clock/cell-cycle system under physiological conditions. It wasn't used
in the end, hence it is not properly commented.
It can be extremely long to run...'''
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import interpolate
import sys
import random
import scipy.integrate as spi
import scipy.stats as st
from scipy.integrate import odeint
import warnings
from multiprocessing import Pool
import signal
import seaborn as sn
#Import internal modules
sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))

from Functions.create_hidden_variables import create_hidden_variables
from Functions.signal_model import signal_model

from Classes.StateVar import StateVar
from Classes.HMMsim import HMMsim

#nice plotting style
sn.set_style("whitegrid", {'grid.color': 'white',
            'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})
""""""""""""""""""""" CHOOSE DATA """""""""""""""""""""
temperature = 37
cell = 'NIH3T3'

""""""""""""""""""""" LOAD PARAMETERS """""""""""""""""""""
with open('../Parameters/Real/opt_parameters_div_'+str(temperature)+"_"\
                                                        +cell+'.p', 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)

""""""""""""""""""""" SET SIMULATION PARAMETERS """""""""""""""""""""
N_trajectories = 5
tf = 1000
ti = 200
T_theta = period_theta
T_phi = period_phi
T_temp = np.nan
F_temp = np.hstack((F,F,F))
F_temp = np.vstack((F_temp, F_temp, F_temp))
F_func = interpolate.interp2d(np.linspace(-2*np.pi,4*np.pi, 3*F.shape[0],
                            endpoint = False),
                            np.linspace(-2*np.pi,4*np.pi, 3*F.shape[1],
                            endpoint = False),
                            F_temp, kind='cubic', bounds_error= True)

space_K2 = np.linspace(0.01,0.08,15)
space_K3 = np.linspace(0.01,0.2,15)
space_gamma_0 = np.linspace(0,2*np.pi, 20, endpoint = False)


""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""""""
theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters = l_parameters)
l_var = [theta_var_coupled, amplitude_var, background_var]
domain_theta = theta_var_coupled.domain
domain_phi = theta_var_coupled.codomain
""""""""""""""""""""" FUNCTION TO SIMULATE TRACES """""""""""""""""""""
def simulate(tf=80, K1 =1, K2 = 1, K3 = 1, gamma_0 = 0, gamma_1 = 0):

    #define intial condition
    theta_0 = random.random()*2*np.pi
    phi_0 = random.random()*2*np.pi
    T_0 = 0#random.random()*2*np.pi
    Y0 = [theta_0, phi_0, T_0]
    tspan = np.linspace(0, tf, abs(2*tf), endpoint = False)

    def system(Y, t):
        theta, phi,T = Y
        d_theta_dt = 2*np.pi/T_theta + K1*F_func(Y[0]%(2*np.pi),Y[1]%(2*np.pi))\
        + K2*np.cos( Y[0]%(2*np.pi) - (2*np.pi/T_temp*t)%(2*np.pi) + gamma_0)
        d_phi_dt = 2*np.pi/T_phi+ K3*np.cos( Y[1]%(2*np.pi) \
                - (2*np.pi/T_temp*t)%(2*np.pi) + gamma_1)
        d_T_dt = 2*np.pi/T_temp
        return [d_theta_dt, d_phi_dt, d_T_dt]

    vect_Y = odeint(system, Y0, tspan)

    return tspan, vect_Y

def compute_dephasing_mp(K2):
    dic_dephasing = {}
    print("Currently computing K2 = ", K2)
    for K3 in space_K3:
        for gamma_0 in space_gamma_0:
            for gamma_1 in space_gamma_0:
                #plt.figure(figsize=(20,20))
                l_res_K1 = [np.nan, np.nan]
                for K1 in [0,1]:
                    lll_theta = []
                    lll_phi = []
                    lll_T= []
                    l_dephasing = []
                    for n in range(N_trajectories):
                        tspan, vect_Y = simulate(tf, K1, K2, K3, gamma_0,
                                                gamma_1)
                        tspan = tspan[ti:]
                        vect_Y = vect_Y[ti:]

                        l_phase_theta_flat =  vect_Y[:,0]%(2*np.pi)
                        l_phase_phi_flat =  vect_Y[:,1]%(2*np.pi)
                        l_phase_T_flat =  vect_Y[:,2]%(2*np.pi)

                        final_theta = l_phase_theta_flat[-1]
                        final_phi = l_phase_phi_flat[-1]
                        final_Temp = l_phase_T_flat[-1]
                        dephasing = (final_Temp-final_theta)%(2*np.pi)
                        l_dephasing.append(dephasing)
                    final_dephasing = np.angle(np.sum([np.exp(1j*phase)\
                                /len(l_dephasing) for phase in l_dephasing]))
                    final_std = st.circstd(l_dephasing)
                    if final_std>0.5: #no stable state reached
                        #print('fail', final_std)
                        break
                    else:
                        #print(K1, K2, K3, gamma_0, 'OK !!')
                        l_res_K1[K1] = [final_dephasing%(2*np.pi), final_std]

                '''
                    #cut traces so that each time a boundary is crossed
                    #it creates a new plot
                    ll_phase_theta=[]
                    ll_phase_phi=[]
                    ll_phase_T = []
                    l_phase_theta=[]
                    l_phase_phi=[]
                    l_phase_T = []
                    zp = zip(l_phase_theta_flat[1:], l_phase_phi_flat[1:],
                            l_phase_theta_flat[:-1], l_phase_phi_flat[:-1],
                            l_phase_T_flat[1:], l_phase_T_flat[:-1])
                    for theta2, phi2, theta1, phi1, T2, T1 in zp:
                        l_phase_theta.append(theta1)
                        l_phase_phi.append(phi1)
                        l_phase_T.append(T1)
                        c1 = abs(theta2-theta1)>np.pi
                        c2 = abs(phi2-phi1)>np.pi
                        c3 = abs(T2-T1)>np.pi
                        if c1 or c2 or c3:
                            ll_phase_theta.append(np.array(l_phase_theta))
                            ll_phase_phi.append(np.array(l_phase_phi))
                            ll_phase_T.append(np.array(l_phase_T))
                            l_phase_theta=[]
                            l_phase_phi=[]
                            l_phase_T = []

                    ll_phase_theta.append(np.array(l_phase_theta))
                    ll_phase_phi.append(np.array(l_phase_phi))
                    lll_theta.append(ll_phase_theta)
                    lll_phi.append(ll_phase_phi)
                    ll_phase_T.append(np.array(l_phase_T))
                    lll_T.append(ll_phase_T)

                    plt.subplot(221+K1)
                    for ll_phase_theta,ll_phase_phi in zip(lll_theta, lll_phi):
                        for l_phase_theta, l_phase_phi in zip(ll_phase_theta,
                                                                ll_phase_phi):
                            plt.plot(l_phase_theta[:-1], l_phase_phi[:-1],
                                     lw = 2, color = 'lightblue')
                            #plt.quiver(l_phase_theta[:-1], l_phase_phi[:-1],
                                        -(l_phase_theta[1:]-l_phase_theta[:-1]),
                                        -(l_phase_phi[1:]-l_phase_phi[:-1]),
                                        scale = 5)

                    plt.xlim(10**-2,2*np.pi-10**-2)
                    plt.ylim(10**-2,2*np.pi-10**-2)
                    plt.xlabel("Circadian phase")
                    plt.ylabel("Cell-cycle phase")
                    plt.title("K1 = " + str(K1) + " K2 = " + str(K2)+" K3 = " \
                                        + str(K3)+" gamma = " + str(gamma_0) )

                    plt.subplot(223+K1)
                    for ll_phase_theta,ll_phase_T in zip(lll_theta, lll_T):
                        for l_phase_theta, l_phase_T in zip(ll_phase_theta,
                                                                ll_phase_T):
                            plt.plot(l_phase_theta[:-1], l_phase_T[:-1], lw = 2,
                                                            color = 'lightblue')
                            #plt.quiver(l_phase_theta[:-1], l_phase_phi[:-1],
                                        -(l_phase_theta[1:]-l_phase_theta[:-1]),
                                        -(l_phase_phi[1:]-l_phase_phi[:-1]),
                                        scale = 5)


                    plt.xlim(10**-2,2*np.pi-10**-2)
                    plt.ylim(10**-2,2*np.pi-10**-2)
                    plt.xlabel("Circadian phase")
                    plt.ylabel("T phase")
                    plt.title("K1 = " + str(K1) + " K2 = " + str(K2)+" K3 = " \
                                        + str(K3)+" gamma = " + str(gamma_0) )


                plt.show()
                plt.close()
                '''
                try:
                    dic_dephasing[(K2,K3,gamma_0, gamma_1)] = \
                                    (l_res_K1[0][0]-l_res_K1[1][0])%(2*np.pi)
                except:
                    pass
    return dic_dephasing

l_arg = space_K2
n_cpu = 20


#22h entrainement
T_temp = 22
warnings.simplefilter("ignore")
original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGINT, original_sigint_handler)
pool = Pool(n_cpu)
try:
    results = pool.map(compute_dephasing_mp, l_arg)
except:
    print("BUG")
    pool.terminate()
else:
    print("Normal termination")
pool.close()
pool.join()

dic_dephasing_22 = { k: v for d in results for k, v in d.items() }

print('22h DONE')

#24h entrainement
T_temp = 24
warnings.simplefilter("ignore")
original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGINT, original_sigint_handler)
pool = Pool(n_cpu)
try:
    results = pool.map(compute_dephasing_mp, l_arg)
except:
    print("BUG")
    pool.terminate()
else:
    print("Normal termination")
pool.close()
pool.join()

dic_dephasing_24 = { k: v for d in results for k, v in d.items() }

print('24h DONE')

#24h entrainement
T_temp = 26
warnings.simplefilter("ignore")
original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGINT, original_sigint_handler)
pool = Pool(n_cpu)
try:
    results = pool.map(compute_dephasing_mp, l_arg)
except:
    print("BUG")
    pool.terminate()
else:
    print("Normal termination")
pool.close()
pool.join()

dic_dephasing_26 = { k: v for d in results for k, v in d.items() }

print('26h DONE')

for key_22, val_22 in dic_dephasing_22.items():
    if val_22/(2*np.pi)*24>10 and val_22/(2*np.pi)*24<12:

        for key_24, val_24 in dic_dephasing_24.items():
            if val_24/(2*np.pi)*24>7.5 and val_24/(2*np.pi)*24<9.5:

                for key_26, val_26 in dic_dephasing_26.items():
                    if val_26/(2*np.pi)*24>3.5 and val_26/(2*np.pi)*24<5.5:

                        print('22h', key_22, val_22)
                        print('24h', key_24, val_24)
                        print('26h', key_26, val_26)


"""
#plot the actual look of the traces for 100 different traces
for n in range(1):
    tspan, vect_Y = simulate(tf, K1=1, K2=0.01, K3 = 0.0575,
                                gamma_0 = 5.654866776461628,
                                gamma_1 = 5.654866776461628)
    tspan = tspan[-150:]
    vect_Y = vect_Y[-150:]

    l_phase_theta_flat =  vect_Y[:,0]%(2*np.pi)
    l_phase_phi_flat =  vect_Y[:,1]%(2*np.pi)
    l_phase_T_flat = vect_Y[:,2]%(2*np.pi)

    plt.plot(tspan,np.cos(l_phase_theta_flat), alpha = 1, color = 'red',
            label = 'theta div')
    #plt.plot(tspan,np.cos(l_phase_phi_flat), alpha = 0.1, color = 'blue',
            label = 'phi')

    tspan, vect_Y = simulate(tf, K1=0, K2=0.01, K3 = 0.0575, gamma_0 = 3.769,
                            gamma_1 = 3.769)
    tspan = tspan[-150:]
    vect_Y = vect_Y[-150:]

    l_phase_theta_flat =  vect_Y[:,0]%(2*np.pi)
    plt.plot(tspan,np.cos(l_phase_theta_flat), alpha = 1, color = 'orange',
            label = 'theta nodiv')
    plt.plot(tspan,np.cos(l_phase_T_flat), '--', alpha = 1, color = 'black',
            label = 'T° entrainment')
plt.legend()

#plt.plot(tspan, np.cos(2*np.pi/24*tspan), '--', label = 'entrainment')
plt.show()
plt.close()

"""
