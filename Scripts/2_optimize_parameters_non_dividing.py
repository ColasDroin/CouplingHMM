# -*- coding: utf-8 -*-
""" This script is used to optimize the parameters which are inferred from
non-dividing traces under the EM algorithm.
Update : Script deprecated since no parameter except the coupling is optimized
using EM. """
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import pickle
import numpy as np
import sys
import copy
import matplotlib
import os
matplotlib.use('Agg') #to run the script on a distant server
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

### Import internal modules
from Classes.LoadData import LoadData
from Classes.HMM_SemiCoupled import HMM_SemiCoupled
from Classes.PlotResults import PlotResults
import Classes.EM as EM

###Import internal functions
from Functions.signal_model import signal_model
from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.clean_waveform import clean_waveform

#access main directory
os.chdir('..')
""""""""""""""""""""" LOAD SHELL ARGUMENTS """""""""""""""""
try:
    nb_iter = int(sys.argv[1])
    nb_traces = int(sys.argv[2])
    size_block = int(sys.argv[3])
    cell = sys.argv[4]
    if sys.argv[5]=="None":
        temperature = None
    else:
        temperature = int(sys.argv[5])
except:
    print("No shell input given, default arguments used")
    nb_iter = 15
    nb_traces = 1000
    size_block = 100
    cell = 'NIH3T3'
    temperature = None

""""""""""""""""""""" LOAD INIT GUESS PARAMETERS """""""""""""""""
path = 'Parameters/Real/init_parameters_nodiv_'+str(temperature)+"_"+cell+".p"
with open(path, 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)

""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
display_parameters_from_file(path, show = False)

""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""
theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )
""""""""""""""""""""" LOAD DATA """""""""""""""""
if cell == 'NIH3T3':
    path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    dataClass=LoadData(path, nb_traces, temperature = temperature,
                        division = False)
else:
    path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
    dataClass=LoadData(path, nb_traces, temperature = temperature,
                        division = True) #all thz U2OS divide

(ll_area_tot_flat, ll_signal_tot_flat, ll_nan_circadian_factor_tot_flat,
    ll_obs_phi_tot_flat, T_theta, T_phi) = dataClass.load()
print(len(ll_signal_tot_flat), " traces kept")


""""""""""""""""""""" CREATE BLOCK OF TRACES """""""""""""""""
ll_area_tot = []
ll_signal_tot = []
first= True
#print(ll_area_tot_flat)
#print(ll_signal_tot_flat)

for index, (l_area, l_signal) in enumerate(zip(ll_area_tot_flat,
                                                ll_signal_tot_flat)):
    if index%size_block==0:
        if not first:
            ll_area_tot.append(ll_area)
            ll_signal_tot.append(ll_signal)

        else:
            first = False
        ll_area = [l_area]
        ll_signal = [l_signal]
    else:
        ll_area.append(l_area)
        ll_signal.append(l_signal)
#get remaining trace
ll_area_tot.append(ll_area)
ll_signal_tot.append(ll_signal)



""""""""""""""""""""" OPTIMIZATION """""""""""""""""
lP=0
l_lP=[]
l_idx_to_remove = []
for it in range(nb_iter):
    print("Iteration :", it)
    l_jP_phase = []
    l_jP_amplitude = []
    l_jP_background = []
    l_gamma = []
    l_gamma_0 = []
    l_logP = []
    ll_signal_hmm = []

    """ CLEAN TRACES AFTER FIRST ITERATION """
    if it==1:
        ll_signal_tot_flat = ll_signal_hmm_clean
        print("nb traces apres 1ere iteration : ", len(ll_signal_tot_flat))
        ll_signal_tot = []
        first= True
        for index, l_signal in enumerate( ll_signal_tot_flat):
            if index%size_block==0:
                if not first:
                    ll_signal_tot.append(ll_signal)
                else:
                    first = False
                ll_signal = [l_signal]
            else:
                ll_signal.append(l_signal)
        #get remaining trace
        ll_signal_tot.append(ll_signal)

    for ll_signal in ll_signal_tot:


        """ INITIALIZE AND RUN HMM """
        l_var = [theta_var_coupled, amplitude_var, background_var]
        hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian,
                            ll_val_phi = [], waveform = W ,  pi = pi,
                            crop = False )
        (l_gamma_0_temp, l_gamma_temp ,l_logP_temp,  ll_alpha,  ll_beta, l_E,
            ll_cnorm, ll_idx_phi_hmm_temp, ll_signal_hmm_temp,
            ll_nan_circadian_factor_temp) = hmm.run_em()

        #create ll_mat_TR
        ll_mat_TR = [np.array([theta_var_coupled.TR_no_coupling \
                        for i in gamma_temp]) for gamma_temp in l_gamma_temp ]


        """ PLOT TRACE EXAMPLE """
        zp = zip(enumerate(ll_signal_hmm_temp),l_gamma_temp, l_logP_temp)
        for (idx, signal), gamma, logP in zp:
            plt_result = PlotResults(gamma, l_var, signal_model, signal,
                                    waveform = W, logP = None,
                                    temperature = temperature, cell = cell)
            plt_result.plotEverythingEsperance(False, idx)
            if idx==0:
                break



        l_jP_phase_temp, l_jP_amplitude_temp,l_jP_background_temp \
            = EM.compute_jP_by_block(ll_alpha, l_E, ll_beta, ll_mat_TR,
                                    amplitude_var.TR, background_var.TR,
                                    N_theta, N_amplitude_theta,
                                    N_background_theta)
        l_jP_phase.extend(l_jP_phase_temp)
        l_jP_amplitude.extend(l_jP_amplitude_temp)
        l_jP_background.extend(l_jP_background_temp)
        l_gamma.extend(l_gamma_temp)
        l_logP.extend(l_logP_temp)
        l_gamma_0.extend(l_gamma_0_temp)
        ll_signal_hmm.extend(ll_signal_hmm_temp)

    t_l_jP = (l_jP_phase, l_jP_amplitude,l_jP_background)

    """ REMOVE BAD TRACES IF FIRST ITERATION"""
    if it==0:
        Plim = np.percentile(l_logP, 10)
        #Plim = -10000
        for idx_trace, P in enumerate(l_logP):
            if P<=Plim:
                l_idx_to_remove.append(idx_trace)
        for index in sorted(l_idx_to_remove, reverse=True):
            del t_l_jP[0][index]
            del t_l_jP[1][index]
            del t_l_jP[2][index]
            del l_gamma[index]
            del l_logP[index]
            del l_gamma_0[index]
            del ll_signal_hmm[index]
        ll_signal_hmm_clean = copy.deepcopy(ll_signal_hmm)



    """ PARAMETERS UPDATE """
    ll_idx_obs_phi = None
    [F_up, pi_up, std_theta_up, sigma_em_circadian_up, ll_coef,
    std_amplitude_theta_up, std_background_theta_up, mu_amplitude_theta_up,
    mu_background_theta_up, W_up] = EM.run_EM(l_gamma_0, l_gamma, t_l_jP,
                                              theta_var_coupled, ll_idx_obs_phi,
                                              F, ll_signal_hmm,  amplitude_var,
                                              background_var, W,
                                              ll_idx_coef = None )

    if np.mean(l_logP)-lP<10**-9:
        print("diff:", np.mean(l_logP)-lP)
        print(print("average lopP:", np.mean(l_logP)))
        break
    else:
        lP = np.mean(l_logP)
        print("average lopP:", lP)
        l_lP.append(lP)

    """ PLOT WAVEFORM """
    plt.plot(theta_var_coupled.domain, W_up)
    plt.xlabel('Circadian phase')
    plt.ylabel(r'$W(\theta)$')
    plt.ylim([-0.1,1.5])
    plt.show()
    plt.close()


    """ CHOOSE NEW PARAMETERS """

    cond_F = False
    cond_pi = False
    cond_std_theta = False
    cond_sigma_em_circadian = False
    cond_std_amplitude_theta = False
    cond_std_background_theta = False
    cond_mu_amplitude_theta = False
    cond_mu_background_theta = False
    cond_W = False

    """ PUT ITERATION CONDITIONS ON THE UPDATE """
    [cond_it_F, cond_it_pi, cond_it_std_theta, cond_it_sigma_em_circadian,
    cond_it_std_amplitude_theta, cond_it_std_background_theta,
    cond_it_mu_amplitude_theta, cond_it_mu_background_theta, cond_it_W] \
                    = [False,False,False,False,False,False,False,False,False]

    '''
    if it>=0:   cond_it_F = True
    if it>=18:   cond_it_pi = True
    if it>=25:   cond_it_std_theta = True
    if it>=18:   cond_it_sigma_em_circadian = True
    if it>=10:   cond_it_std_amplitude_theta = True
    if it>=14:   cond_it_std_background_theta = True
    if it>=10:   cond_it_mu_amplitude_theta = True
    if it>=14:   cond_it_mu_background_theta = True
    if it>=0:   cond_it_W = True
    '''

    if it>=0:   cond_it_F = True
    if it>=0:   cond_it_pi = True
    if it>=0:   cond_it_std_theta = True
    if it>=0:   cond_it_sigma_em_circadian = True
    if it>=0:   cond_it_std_amplitude_theta = True
    if it>=0:   cond_it_std_background_theta = True
    if it>=0:   cond_it_mu_amplitude_theta = True
    if it>=0:   cond_it_mu_background_theta = True
    if it>=0:   cond_it_W = True

    """ REINPUT NEW PARAMETERS """
    if cond_F and cond_it_F:
        F = F_up
    if cond_pi and cond_it_pi:
        pi = pi_up
    #block simga_theta to 0.2 to keep good inference
    if cond_std_theta and cond_it_std_theta and std_theta_up<0.2:
        std_theta = std_theta_up
    #block sigma_e to 0.1 to prevent divergence
    if cond_sigma_em_circadian and cond_it_sigma_em_circadian \
                                                and sigma_em_circadian_up>0.:
        sigma_em_circadian = sigma_em_circadian_up
    if cond_std_amplitude_theta and cond_it_std_amplitude_theta:
        std_amplitude_theta = std_amplitude_theta_up
    if cond_std_background_theta and cond_it_std_background_theta:
        std_background_theta = std_background_theta_up
    if cond_mu_amplitude_theta and cond_it_mu_amplitude_theta:
        mu_amplitude_theta = mu_amplitude_theta_up
    if cond_mu_background_theta and cond_it_mu_background_theta:
        mu_background_theta = mu_background_theta_up
    if cond_W and cond_it_W:
        W = W_up


    #redefine boundaries OU process if first iterations
    if it<10:
        l_boundaries_amplitude_theta = (mu_amplitude_theta-5*std_amplitude_theta,
                                    mu_amplitude_theta+5*std_amplitude_theta)
    if it<14:
        l_boundaries_background_theta = (mu_background_theta\
                                    -5*std_background_theta,
                                    mu_background_theta+5*std_background_theta)

    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
     gamma_background_theta, l_boundaries_background_theta,
    F]


    theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters)


plt.plot(l_lP)
plt.savefig("Parameters/Real/opt_parameters_nodiv_"+str(temperature)+"_"\
                                                                +cell+'.pdf')
#plt.show()
plt.close()

""""""""""""""""""""" RENORMALIZE WAVEFORM AND UPDATE PARAMETERS """""""""""""""
W = clean_waveform(W)

l_parameters = [dt, sigma_em_circadian, W, pi,
N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
gamma_amplitude_theta, l_boundaries_amplitude_theta,
N_background_theta, mu_background_theta, std_background_theta,
gamma_background_theta, l_boundaries_background_theta,
F]

""""""""""""""""""""" WRAP PARAMETERS """""""""""""""""
path = "Parameters/Real/opt_parameters_nodiv_"+str(temperature)+"_"+cell+".p"
pickle.dump( l_parameters, open( path, "wb" ) )
""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
display_parameters_from_file(path, show = True)
