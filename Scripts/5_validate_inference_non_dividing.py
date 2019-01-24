# -*- coding: utf-8 -*-
""" This script is used to validate the inference process for the parameters
inferred from non-dividing traces.
Update : Script deprecated since no parameter except the coupling is optimized
using EM. """
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import pickle
import numpy as np
import sys
import copy
import matplotlib
matplotlib.use('Agg') #to run the script on a distant server
import matplotlib.pyplot as plt
from scipy import interpolate
import random
import os

sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

### Import internal modules
from Classes.LoadData import LoadData
from Classes.HMM_SemiCoupled import HMM_SemiCoupled
from Classes.PlotResults import PlotResults
import Classes.EM as EM
from Classes.HMMsim import HMMsim

###Import internal functions
from Functions.create_hidden_variables import create_hidden_variables
from Functions.make_colormap import make_colormap
from Functions.signal_model import signal_model
from Functions.display_parameters import display_parameters_from_file
from RawDataAnalysis.estimate_waveform import estimate_waveform_from_signal
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
    nb_iter = 50
    nb_traces = 500
    size_block = 100
    cell = 'NIH3T3'
    temperature = None

""""""""""""""""""""" LOAD OPTIMIZED PARAMETERS """""""""""""""""
path = 'Parameters/Real/opt_parameters_nodiv_'+str(temperature)+"_"+cell+'.p'
with open(path, 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)

W_init = copy.copy(W) #save to make a plot in the end
""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
display_parameters_from_file(path, show = True)

""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""
theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )
l_var = [theta_var_coupled, amplitude_var, background_var]


""""""""""""""""""""" COMPUTE NB_TRACE_MAX """""""""""""""""
if cell == 'NIH3T3':
    path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    dataClass=LoadData(path, nb_traces, temperature = temperature,
                        division = False)
else:
    path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
    dataClass=LoadData(path, nb_traces, temperature = temperature,
                        division = True) #all the U2OS divide

(ll_area_tot_flat, ll_signal_tot_flat, ll_nan_circadian_factor_tot_flat,
    ll_obs_phi_tot_flat, T_theta, T_phi) = dataClass.load()
nb_traces_max = len(ll_signal_tot_flat)
if nb_traces>nb_traces_max:
    print("CAUTION : too many traces, number of traces generated reduced to: ",
          nb_traces_max)
    nb_traces = nb_traces_max


""""""""""""""""""""" GENERATE UNCOUPLED TRACES """""""""""
sim = HMMsim(l_var, signal_model , sigma_em_circadian,  waveform = W ,
            dt=0.5, uniform = True )
ll_t_l_xi, ll_t_obs  =  sim.simulate_n_traces(nb_traces=nb_traces, tf=60)

""""""""""""""""""""" REORDER VARIABLES """""""""""""""""
ll_obs_circadian = []
ll_obs_nucleus = []
lll_xi_circadian = []
lll_xi_nucleus = []
for idx, (l_t_l_xi, l_t_obs) in enumerate(zip(ll_t_l_xi, ll_t_obs)):
    ll_xi_circadian = [ t_l_xi[0] for t_l_xi in l_t_l_xi   ]
    ll_xi_nucleus = [ t_l_xi[1] for t_l_xi in l_t_l_xi   ]
    l_obs_circadian = np.array(l_t_obs)[:,0]
    l_obs_nucleus = np.array(l_t_obs)[:,1]
    ll_obs_circadian.append(l_obs_circadian)
    ll_obs_nucleus.append(l_obs_nucleus)
    lll_xi_circadian.append(ll_xi_circadian)
    lll_xi_nucleus.append(ll_xi_nucleus)

""""""""""""""""""""" COMPUTE ll_peak and possibly ll_idx_cell_cycle_start """""
ll_peak = []
ll_idx_cell_cycle_start=[]
#phase 0 is taken as reference for peak
for ll_xi_circadian in lll_xi_circadian:
    l_peak = []
    zp = zip(ll_xi_circadian[:-1], ll_xi_circadian[1:])
    for l_xi_circadian_1, l_xi_circadian_2 in zp:
        if l_xi_circadian_2[0]-l_xi_circadian_1[0]<-np.pi \
                                                    and not 1 in l_peak[-10:]:
            l_peak.append(1)
        else:
            l_peak.append(0)
    l_peak.append(0)
    ll_peak.append(l_peak)
if cell=='U2OS':
    for ll_xi_nucleus in lll_xi_nucleus:
        l_idx = []
        zp = zip(enumerate(ll_xi_nucleus[:-1]), ll_xi_nucleus[1:])
        for (idx, l_xi_nucleus_1), l_xi_nucleus_2 in zp:
            if l_xi_nucleus_2[0]-l_xi_nucleus_1[0] < -np.pi:
                l_idx.append(idx)
        ll_idx_cell_cycle_start.append(l_idx)

""""""""""""""""""""" PLOT TRACE EXAMPLES """""""""""""""""
zp = zip(enumerate(lll_xi_nucleus), lll_xi_circadian, ll_obs_circadian)
for (idx_trace, ll_xi_nucleus), ll_xi_circadian, l_obs_circadian in zp:
    tspan = np.linspace(0, len(l_obs_circadian)/2, len(l_obs_circadian),
                        endpoint=False  )
    ll_xi_nucleus = np.array(ll_xi_nucleus)
    ll_xi_circadian = np.array(ll_xi_circadian)
    plt.plot(tspan, l_obs_circadian, '-')
    plt.plot(tspan, ll_xi_circadian[:,0]/(2*np.pi), '--')
    plt.plot(tspan, np.exp(ll_xi_circadian[:,1]), '--')
    plt.plot(tspan, ll_xi_circadian[:,2], '--')
    plt.plot(tspan, ll_xi_nucleus[:,0]/(2*np.pi), '--')
    plt.plot(tspan, ll_peak[idx_trace])
    try:
        for v in ll_idx_cell_cycle_start[idx_trace]:
            plt.axvline(v/2)
    except:
        pass
    plt.show()
    plt.savefig("Results/Validation/Trace_no_div_"+str(idx_trace)+'_'+cell\
                                                +'_'+str(temperature)+'.pdf')
    plt.close()
    if idx_trace>10:
        break



""""""""""""""""""""" DEFINE PARAMETERS FOR INFERENCE """""""""""""""""
N_theta = 48
sigma_em_circadian = sigma_em_circadian
W = estimate_waveform_from_signal(ll_signal = ll_obs_circadian,
                            ll_peak = ll_peak,
                            domain_theta = theta_var_coupled.domain,
                            ll_idx_cell_cycle_start = ll_idx_cell_cycle_start)
pi = None
std_theta = std_theta

N_amplitude_theta = 30
mu_amplitude_theta = mu_amplitude_theta
std_amplitude_theta = std_amplitude_theta
gamma_amplitude_theta = gamma_amplitude_theta #GAMMA ASSUMED FIXED
l_boundaries_amplitude_theta = (mu_amplitude_theta-5*std_amplitude_theta,
                                mu_amplitude_theta+5*std_amplitude_theta)

N_background_theta = 30
mu_background_theta = mu_background_theta
std_background_theta = std_background_theta
gamma_background_theta = gamma_background_theta  #GAMMA ASSUMED FIXED
l_boundaries_background_theta = (mu_background_theta-5*std_background_theta,
                                mu_background_theta+5*std_background_theta)

""""""""""""""""""""" RECREATE HIDDEN VARIABLES FOR INFERENCE """""""""""""""""
l_parameters = [dt, sigma_em_circadian, W, pi,
N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
gamma_amplitude_theta, l_boundaries_amplitude_theta,
N_background_theta, mu_background_theta, std_background_theta,
gamma_background_theta, l_boundaries_background_theta,
F]
theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )


""""""""""""""""""""" WRAP INTIAL CONDITION PARAMETERS """""""""""""""""
path = "Parameters/Silico/init_parameters_nodiv_"+str(temperature)+"_"+cell+'.p'
pickle.dump( l_parameters, open(path , "wb" ) )

""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
display_parameters_from_file(path, show = True)

""""""""""""""""""""" DEFINE BLOCKS OF TRACES """""""""""""""""
ll_signal_tot_flat = ll_obs_circadian
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


""""""""""""""""""""" OPTIMIZATION """""""""""""""""
lP=-10000
l_lP=[]
l_idx_to_remove = []
for it in range(0, nb_iter):
    print("Iteration :", it)
    l_jP_phase = []
    l_jP_amplitude = []
    l_jP_background = []
    l_gamma = []
    l_gamma_0 = []
    l_logP = []


    for ll_signal in ll_signal_tot :

        """ INITIALIZE AND RUN HMM """
        l_var = [theta_var_coupled, amplitude_var, background_var]
        hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian,
                            ll_val_phi = [], waveform = W , pi = pi,
                            crop = False )
        (l_gamma_0_temp, l_gamma_temp ,l_logP_temp, ll_alpha, ll_beta, l_E,
            ll_cnorm, ll_idx_phi_hmm_temp, ll_signal_hmm_temp,
            ll_nan_circadian_factor_hmm_temp) = hmm.run_em()

        ll_mat_TR = [np.array( [theta_var_coupled.TR_no_coupling \
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

    t_l_jP = (l_jP_phase, l_jP_amplitude,l_jP_background)



    """ PARAMETERS UPDATE """
    ll_idx_obs_phi = None
    [F_up, pi_up, std_theta_up, sigma_em_circadian_up, ll_coef,
    std_amplitude_theta_up, std_background_theta_up, mu_amplitude_theta_up,
    mu_background_theta_up, W_up] = EM.run_EM(l_gamma_0, l_gamma, t_l_jP,
                                              theta_var_coupled, ll_idx_obs_phi,
                                              F, ll_signal_tot_flat,
                                              amplitude_var,  background_var,
                                              W, ll_idx_coef = None)

    if np.mean(l_logP)-lP<10**-9:
        print("diff:", np.mean(l_logP)-lP)
        print(print("average lopP:", np.mean(l_logP)))
        break
    else:
        lP = np.mean(l_logP)
        print("average lopP:", lP)
        l_lP.append(lP)


    """ CHOOSE NEW PARAMETERS """

    cond_F = False
    cond_pi = True
    cond_std_theta = False
    cond_sigma_em_circadian = False
    cond_std_amplitude_theta = False
    cond_std_background_theta = False
    cond_mu_amplitude_theta = False
    cond_mu_background_theta = False
    cond_W = True


    """ PUT ITERATION CONDITIONS ON THE UPDATE """
    [cond_it_F, cond_it_pi, cond_it_std_theta, cond_it_sigma_em_circadian,
    cond_it_std_amplitude_theta, cond_it_std_background_theta,
    cond_it_mu_amplitude_theta, cond_it_mu_background_theta, cond_it_W] \
                       = [False,False,False,False,False,False,False,False,False]

    """
    if it>=0:   cond_it_F = True
    if it>=0:   cond_it_pi = True
    if it>=22:   cond_it_std_theta = True
    if it>=5:   cond_it_sigma_em_circadian = True
    if it>=10:   cond_it_std_amplitude_theta = True
    if it>=14:   cond_it_std_background_theta = True
    if it>=10:   cond_it_mu_amplitude_theta = True
    if it>=14:   cond_it_mu_background_theta = True
    if it>=0:   cond_it_W = True
    """

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
    if cond_std_theta and cond_it_std_theta:
        std_theta = std_theta_up
    if cond_sigma_em_circadian and cond_it_sigma_em_circadian:
         #block sigma_e to 0.1 to prevent divergence
        if sigma_em_circadian_up>0.1:
            sigma_em_circadian = sigma_em_circadian_up
        else:
            sigma_em_circadian = 0.1
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
        """ PLOT WAVEFORM """
        plt.plot(theta_var_coupled.domain, W_up)
        plt.xlabel('Circadian phase')
        plt.ylabel(r'$W(\theta)$')
        plt.ylim([-0.1,1.5])
        plt.show()
        plt.close()



    #redefine boundaries OU process if first iterations
    if it<10:
        l_boundaries_amplitude_theta =(mu_amplitude_theta-5*std_amplitude_theta,
                                       mu_amplitude_theta+5*std_amplitude_theta)
    if it<14:
        l_boundaries_background_theta = (mu_background_theta\
                                        -5*std_background_theta,
                                        mu_background_theta\
                                        +5*std_background_theta)


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
plt.savefig("Parameters/Silico/opt_parameters_nodiv_"+str(temperature)+"_"\
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
pickle.dump( l_parameters, open( "Parameters/Silico/opt_parameters_nodiv_"\
                                    +str(temperature)+"_"+cell+'.p', "wb" ) )

""""""""""""""""""""" COMPARE TRUE AND INFERRED WAVEFORMS """""""""""""""""
plt.plot(theta_var_coupled.domain, W, label = 'Inferred')
plt.plot(theta_var_coupled.domain, W_init, label = 'Theo')
plt.xlabel('Circadian phase')
plt.ylabel(r'$W(\theta)$')
plt.legend()
plt.ylim([-0.1,1.5])
plt.savefig('Results/Validation/W_inferred_'+str(temperature)+"_"+cell+'.pdf')
plt.show()
plt.close()
