# -*- coding: utf-8 -*-
""" This script is used to find the optimal smoothing parameter (it can be
extremely long to run if done on all the traces), as well as to compute the bias
which should be removed at each iteration. """
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import pickle
import numpy as np
import sys
import copy
import matplotlib
matplotlib.use('Agg') #to run the script on a distant server
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import os

sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

### Import internal modules
from Classes.LoadData import LoadData
from Classes.HMM_SemiCoupled import HMM_SemiCoupled
from Classes.PlotResults import PlotResults
import Classes.EM as EM

###Import internal functions
from Functions.create_hidden_variables import create_hidden_variables
from Functions.make_colormap import make_colormap
from Functions.signal_model import signal_model

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
    nb_iter = 30
    nb_traces = 1000
    size_block = 100
    cell = 'NIH3T3'
    temperature = None

""""""""""""""""""""" LOAD COLORMAP """""""""""""""""
c = mcolors.ColorConverter().to_rgb
bwr = make_colormap(  [c('blue'), c('white'), 0.48, c('white'),
                        0.52,c('white'),  c('red')])

""""""""""""""""""""" LOAD OPTIMIZED PARAMETERS """""""""""""""""

with open('Parameters/Real/opt_parameters_nodiv_'+str(temperature)+"_"\
                                                        +cell+'.p', 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)

""""""""""""""""""""" LOAD DATA """""""""""""""""
if cell == 'NIH3T3':
    path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
else:
    path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
dataClass=LoadData(path, nb_traces, temperature = temperature, division = True,
                    several_cell_cycles = False)
(ll_area_tot_flat_tot, ll_signal_tot_flat_tot,
    ll_nan_circadian_factor_tot_flat_tot, ll_obs_phi_tot_flat_tot,
    T_theta, T_phi) = dataClass.load()
print(len(ll_signal_tot_flat_tot), " traces kept")

""""""""""""""""""""" SPECIFY F OPTIMIZATION CONDITIONS """""""""""""""""

#makes algorithm go faster
only_F_and_pi = True
#we don't know the inital condiion when traces divide
pi = None
#we start with a random empty F
F = (np.random.rand( N_theta, N_phi)-0.5)*0.01

""""""""""""""""""""" CORRECT PARAMETERS ACCORDINGLY """""""""""""""""

l_parameters = [dt, sigma_em_circadian, W, pi,
N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
gamma_amplitude_theta, l_boundaries_amplitude_theta,
N_background_theta, mu_background_theta, std_background_theta,
gamma_background_theta, l_boundaries_background_theta,
F]

""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""
theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )

""""""""""""""""""""" CORRECT INFERENCE BIAS """""""""""""""""
try:
    F_no_coupling = pickle.load( open( "Parameters/Misc/F_no_coupling_"\
                                    +str(temperature)+"_"+cell+'.p', "rb" ) )
    for idx_theta, theta in enumerate(theta_var_coupled.domain):
        F_no_coupling[idx_theta, :] = np.mean(F_no_coupling[idx_theta, :])
except:
    print("F_no_coupling not found, no bias correction applied")
    F_no_coupling = None

""""""""""""""""""""" DIVIDE THE DATASET IN CHUNKS """""""""""""""""
nb_traces_tot = len(ll_signal_tot_flat_tot)
fraction_CV = 3
size_chunk = int(nb_traces_tot/fraction_CV)
chunked_ll_area_tot_flat_tot = [ll_area_tot_flat_tot[x:x+size_chunk] \
                       for x in range(0, len(ll_area_tot_flat_tot), size_chunk)]
chunked_ll_signal_tot_flat_tot = [ll_signal_tot_flat_tot[x:x+size_chunk] \
                    for x in range(0, len(ll_signal_tot_flat_tot), size_chunk)]
chunked_ll_nan_circadian_factor_tot_flat_tot = \
    [ll_nan_circadian_factor_tot_flat_tot[x:x+size_chunk] \
    for x in range(0, len(ll_nan_circadian_factor_tot_flat_tot), size_chunk)]
chunked_ll_obs_phi_tot_flat_tot = [ll_obs_phi_tot_flat_tot[x:x+size_chunk] \
                    for x in range(0, len(ll_obs_phi_tot_flat_tot), size_chunk)]

""""""""""""""""""" RUN EM OPTI ON A GIVEN SET OF CHUNKED TRAIN/TEST """""""""
l_idx_chunk = list(range(fraction_CV))
ll_lP_test = []
for idx_chunk in range(fraction_CV):
    print("chunk ", idx_chunk+1, " out of ", fraction_CV, ' being analyzed')
    current_l_idx_chunk = l_idx_chunk[:idx_chunk]+l_idx_chunk[idx_chunk+1:]

    ll_area_tot_flat_train,  ll_area_tot_flat_test\
        = ([x for i in current_l_idx_chunk \
            for x in chunked_ll_area_tot_flat_tot[i]],
          chunked_ll_area_tot_flat_tot[idx_chunk])

    ll_signal_tot_flat_train, ll_signal_tot_flat_test \
        = ([x for i in current_l_idx_chunk \
            for x in chunked_ll_signal_tot_flat_tot[i]],
          chunked_ll_signal_tot_flat_tot[idx_chunk])

    (ll_nan_circadian_factor_tot_flat_train,
    ll_nan_circadian_factor_tot_flat_test) = \
        ([x for i in current_l_idx_chunk\
            for x in chunked_ll_nan_circadian_factor_tot_flat_tot[i]],
        chunked_ll_nan_circadian_factor_tot_flat_tot[idx_chunk])

    ll_obs_phi_tot_flat_train, ll_obs_phi_tot_flat_test \
        = ([x for i in current_l_idx_chunk \
                for x in chunked_ll_obs_phi_tot_flat_tot[i]],
            chunked_ll_obs_phi_tot_flat_tot[idx_chunk])

    """"""""""""""""""""" CREATE BLOCK OF TRACES """""""""""""""""
    ll_area_tot = []
    ll_signal_tot = []
    ll_nan_circadian_factor_tot = []
    ll_obs_phi_tot = []
    first= True
    zp = enumerate(zip(ll_area_tot_flat_train, ll_signal_tot_flat_train,
                        ll_nan_circadian_factor_tot_flat_train,
                        ll_obs_phi_tot_flat_train))
    for index, (l_area, l_signal, l_nan_circadian_factor, l_obs_phi) in zp:
        if index%size_block==0:
            if not first:
                ll_area_tot.append(ll_area)
                ll_signal_tot.append(ll_signal)
                ll_nan_circadian_factor_tot.append(ll_nan_circadian_factor)
                ll_obs_phi_tot.append(ll_obs_phi)

            else:
                first = False
            ll_area = [l_area]
            ll_signal = [l_signal]
            ll_nan_circadian_factor = [l_nan_circadian_factor]
            ll_obs_phi = [l_obs_phi]
        else:
            ll_area.append(l_area)
            ll_signal.append(l_signal)
            ll_nan_circadian_factor.append(l_nan_circadian_factor)
            ll_obs_phi.append(l_obs_phi)
    #get remaining trace
    ll_area_tot.append(ll_area)
    ll_signal_tot.append(ll_signal)
    ll_nan_circadian_factor_tot.append(ll_nan_circadian_factor)
    ll_obs_phi_tot.append(ll_obs_phi)



    """"""""""""""""""""" OPTIMIZATION """""""""""""""""
    def buildObsPhiFromIndex(ll_index):
        ll_obs_phi = []
        for l_index in ll_index:
            l_obs_phi = []
            for index in l_index:
                if index==-1:
                    l_obs_phi.append(-1)
                else:
                    l_obs_phi.append(index/N_theta*2*np.pi)
            ll_obs_phi.append(l_obs_phi)
        return ll_obs_phi


    l_lP_test=[]
    lambda_space = np.logspace(-9,-4,10)
    for lambda_parameter in lambda_space:
        l_lP_train = []
        lP=0
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
            ll_idx_phi_hmm = []
            ll_nan_circadian_factor_hmm = []



            """ CLEAN TRACES AFTER FIRST ITERATION """
            if it==1:
                ll_signal_tot_flat = ll_signal_hmm_clean
                ll_idx_phi_tot_flat = ll_idx_phi_hmm_clean
                ll_nan_circadian_factor_tot_flat = \
                                              ll_nan_circadian_factor_hmm_clean
                print("nb traces after 1st iteration : ",
                     len(ll_signal_tot_flat))
                ll_signal_tot = []
                ll_idx_phi_tot = []
                ll_nan_circadian_factor_tot = []
                ll_obs_phi_tot = []
                first= True
                zp = zip(enumerate( ll_signal_tot_flat), ll_idx_phi_tot_flat,
                                    ll_nan_circadian_factor_tot_flat)
                for (index, l_signal), l_idx_phi, l_nan_circadian_factor in zp:
                    if index%size_block==0:
                        if not first:
                            ll_signal_tot.append(ll_signal)
                            ll_idx_phi_tot.append(ll_idx_phi)
                            ll_nan_circadian_factor_tot.append(
                                                        ll_nan_circadian_factor)
                            ll_obs_phi_tot.append( buildObsPhiFromIndex(
                                                                    ll_idx_phi))
                        else:
                            first = False
                        ll_signal = [l_signal]
                        ll_idx_phi = [l_idx_phi]
                        ll_nan_circadian_factor = [l_nan_circadian_factor]
                    else:
                        ll_signal.append(l_signal)
                        ll_idx_phi.append(l_idx_phi)
                        ll_nan_circadian_factor.append(l_nan_circadian_factor)

                #get remaining trace
                ll_signal_tot.append(ll_signal)
                ll_idx_phi_tot.append(ll_idx_phi)
                ll_nan_circadian_factor_tot.append(ll_nan_circadian_factor)
                ll_obs_phi_tot.append( buildObsPhiFromIndex(ll_idx_phi) )

            zp = zip(ll_signal_tot, ll_obs_phi_tot, ll_nan_circadian_factor_tot)
            for ll_signal, ll_obs_phi, ll_nan_circadian_factor in zp:

                """ INITIALIZE AND RUN HMM """
                l_var = [theta_var_coupled, amplitude_var, background_var]
                hmm = HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian,
                                      ll_val_phi = ll_obs_phi, waveform = W ,
                                      ll_nan_factor = ll_nan_circadian_factor,
                                       pi = pi, crop = True )
                (l_gamma_0_temp, l_gamma_temp ,l_logP_temp, ll_alpha, ll_beta,
                    l_E, ll_cnorm, ll_idx_phi_hmm_temp, ll_signal_hmm_temp,
                    ll_nan_circadian_factor_hmm_temp) = hmm.run_em()

                #crop and create ll_mat_TR
                ll_signal_hmm_cropped_temp = [[s for s, idx in zip(l_s, l_idx) \
                                              if idx>-1] for l_s, l_idx \
                                              in zip(ll_signal_hmm_temp,
                                                    ll_idx_phi_hmm_temp)  ]
                ll_idx_phi_hmm_cropped_temp =[[idx for idx in l_idx if idx>-1] \
                                            for l_idx in  ll_idx_phi_hmm_temp ]
                ll_mat_TR = [ np.array( [theta_var_coupled.TR[:,idx_phi,:] \
                                        for idx_phi in l_idx_obs_phi]) \
                                        for l_idx_obs_phi \
                                        in ll_idx_phi_hmm_cropped_temp ]


                """ PLOT TRACE EXAMPLE """
                zp2 = zip(enumerate(ll_signal_hmm_cropped_temp),l_gamma_temp,
                                    l_logP_temp)
                for (idx, signal), gamma, logP in zp2:
                    plt_result = PlotResults(gamma, l_var, signal_model,
                                            signal, waveform = W, logP = None,
                                            temperature = temperature,
                                            cell = cell)
                    plt_result.plotEverythingEsperance(False, idx)
                    if idx==0:
                        break

                l_jP_phase_temp, l_jP_amplitude_temp,l_jP_background_temp \
                    = EM.compute_jP_by_block(ll_alpha, l_E, ll_beta, ll_mat_TR,
                                            amplitude_var.TR, background_var.TR,
                                            N_theta, N_amplitude_theta,
                                            N_background_theta, only_F_and_pi)
                l_jP_phase.extend(l_jP_phase_temp)
                l_jP_amplitude.extend(l_jP_amplitude_temp)
                l_jP_background.extend(l_jP_background_temp)
                l_gamma.extend(l_gamma_temp)
                l_logP.extend(l_logP_temp)
                l_gamma_0.extend(l_gamma_0_temp)
                ll_signal_hmm.extend(ll_signal_hmm_temp)
                ll_idx_phi_hmm.extend(ll_idx_phi_hmm_temp)
                ll_nan_circadian_factor_hmm.extend(
                                               ll_nan_circadian_factor_hmm_temp)

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
                    del ll_idx_phi_hmm[index]
                    del ll_nan_circadian_factor_hmm[index]
                ll_signal_hmm_clean = copy.deepcopy(ll_signal_hmm)
                ll_idx_phi_hmm_clean = copy.deepcopy(ll_idx_phi_hmm)
                ll_nan_circadian_factor_hmm_clean = copy.deepcopy(
                                                    ll_nan_circadian_factor_hmm)


            """ PARAMETERS UPDATE """

            [F_up, pi_up, std_theta_up, sigma_em_circadian_up, ll_coef,
            std_amplitude_theta_up, std_background_theta_up,
            mu_amplitude_theta_up, mu_background_theta_up, W_up] \
                = EM.run_EM(l_gamma_0, l_gamma, t_l_jP, theta_var_coupled,
                            ll_idx_phi_hmm, F, ll_signal_hmm,  amplitude_var,
                            background_var, W, ll_idx_coef = F_no_coupling,
                            only_F_and_pi = only_F_and_pi,
                            lambd_parameter = lambda_parameter )

            if np.mean(l_logP)-lP<10**-9:
                print("diff:", np.mean(l_logP)-lP)
                print(print("average lopP:", np.mean(l_logP)))
                break
            else:
                lP = np.mean(l_logP)
                print("average lopP:", lP)
                l_lP_train.append(lP)

            """ CHOOSE NEW PARAMETERS """
            F = F_up
            pi_up = pi

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

            """ PLOT COUPLING FUNCTION
            plt.pcolormesh(theta_var_coupled.domain, theta_var_coupled.codomain,
                            F.T, cmap=bwr, vmin=-0.3, vmax=0.3)
            plt.xlim([0, 2*np.pi])
            plt.ylim([0, 2*np.pi])
            plt.colorbar()
            plt.xlabel("theta")
            plt.ylabel("phi")
            plt.show()
            plt.close()
            """


        """ RUN TESTING SET """
        l_var_1 = [theta_var_coupled, amplitude_var, background_var]
        hmm=HMM_SemiCoupled(l_var, ll_signal_tot_flat_test, sigma_em_circadian,
                        ll_val_phi = ll_obs_phi_tot_flat_test, waveform = W,
                        ll_nan_factor = ll_nan_circadian_factor_tot_flat_test,
                        pi = pi, crop = True )
        l_logP = hmm.runOpti()
        l_lP_test.append(np.mean(l_logP))


    #after analyzing a whole chunk, save the list of probability on the test set
    #for every lambda
    ll_lP_test.append(l_lP_test)

#compute the mean probability on each chunk for every lambda
mean_l_lP_test = np.sum( np.array(ll_lP_test), axis = 0 )/len(lambda_space)
plt.semilogx(lambda_space, mean_l_lP_test)
plt.xlabel("lambda")
plt.ylabel("logP")
plt.savefig('Results/Smoothing/CV_'+str(temperature)+"_"+cell+'.pdf')
plt.show()
plt.close()
