# -*- coding: utf-8 -*-
""" This script is used to optimize the parameters which are inferred from
dividing traces under the EM algorithm. In practice, it is only the coupling
function as well with the initial condition. """
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
from Functions.signal_model import signal_model
from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters
from Functions.display_parameters import display_parameters_from_file
from Functions.make_colormap import make_colormap

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
    nb_iter = 10
    nb_traces = 3000
    size_block = 100
    cell = 'NIH3T3'
    temperature = None


""""""""""""""""""""" LOAD COLORMAP """""""""""""""""
c = mcolors.ColorConverter().to_rgb
bwr = make_colormap(  [c('blue'), c('white'), 0.48, c('white'),
                        0.52,c('white'),  c('red')])

""""""""""""""""""""" LOAD OPTIMIZED PARAMETERS """""""""""""""""
path = 'Parameters/Real/opt_parameters_nodiv_'+str(temperature)+"_"+cell+'.p'
with open(path , 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)


""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
display_parameters_from_file(path, show = True)

""""""""""""""""""""" LOAD DATA """""""""""""""""
if cell == 'NIH3T3':
    path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
else:
    path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"

dataClass=LoadData(path, nb_traces, temperature = temperature, division = True,
                    several_cell_cycles = True, remove_odd_traces = True)
(ll_area_tot_flat, ll_signal_tot_flat, ll_nan_circadian_factor_tot_flat,
    ll_obs_phi_tot_flat, T_theta, T_phi) = dataClass.load()
print(len(ll_signal_tot_flat), " traces kept")

""""""""""""""""""""" SPECIFY F OPTIMIZATION CONDITIONS """""""""""""""""

#makes algorithm go faster
only_F_and_pi = True
#we don't know the inital condiion when traces divide
pi = None
#we start with a random empty F
F = (np.random.rand( N_theta, N_phi)-0.5)*0.01
#regularization
lambda_parameter = 2*10e-6
lambda_2_parameter = 0.005


""""""""""""""""""""" CORRECT INFERENCE BIAS """""""""""""""""
try:
    F_no_coupling = pickle.load( open( "Parameters/Misc/F_no_coupling_"\
                                +str(temperature)+"_"+cell+'.p', "rb" ) )
    for idx_theta in range(N_theta):
        F_no_coupling[idx_theta, :] = np.mean(F_no_coupling[idx_theta, :])

except:
    print("F_no_coupling not found, no bias correction applied")
    F_no_coupling = None


""""""""""""""""""""" CORRECT PARAMETERS ACCORDINGLY """""""""""""""""

l_parameters = [dt, sigma_em_circadian, W, pi,
N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
gamma_amplitude_theta, l_boundaries_amplitude_theta,
N_background_theta, mu_background_theta, std_background_theta,
gamma_background_theta, l_boundaries_background_theta,
F]

""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
display_parameters(l_parameters, show = True)

""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""
theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )

""""""""""""""""""""" CREATE BLOCK OF TRACES """""""""""""""""
ll_area_tot = []
ll_signal_tot = []
ll_nan_circadian_factor_tot = []
ll_obs_phi_tot = []
first= True
zp = enumerate(zip(ll_area_tot_flat, ll_signal_tot_flat,
                    ll_nan_circadian_factor_tot_flat, ll_obs_phi_tot_flat))
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
    ll_idx_phi_hmm = []
    ll_nan_circadian_factor_hmm = []



    """ CLEAN TRACES AFTER FIRST ITERATION """
    if it==1:
        ll_signal_tot_flat = ll_signal_hmm_clean
        ll_idx_phi_tot_flat = ll_idx_phi_hmm_clean
        ll_nan_circadian_factor_tot_flat = ll_nan_circadian_factor_hmm_clean
        print("nb traces apres 1ere iteration : ", len(ll_signal_tot_flat))
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
                    ll_nan_circadian_factor_tot.append(ll_nan_circadian_factor)
                    ll_obs_phi_tot.append( buildObsPhiFromIndex(ll_idx_phi) )
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
        hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian,
                            ll_val_phi = ll_obs_phi, waveform = W ,
                            ll_nan_factor = ll_nan_circadian_factor,
                            pi = pi, crop = True )
        (l_gamma_0_temp, l_gamma_temp ,l_logP_temp,  ll_alpha,  ll_beta, l_E,
            ll_cnorm, ll_idx_phi_hmm_temp, ll_signal_hmm_temp,
            ll_nan_circadian_factor_hmm_temp) = hmm.run_em()

        #crop and create ll_mat_TR
        ll_signal_hmm_cropped_temp = [[s for s, idx in zip(l_s, l_idx) \
                                                                if idx>-1] \
                                    for l_s, l_idx in  zip(ll_signal_hmm_temp,
                                    ll_idx_phi_hmm_temp)
                                    ]

        ll_idx_phi_hmm_cropped_temp = [[idx for idx in l_idx if idx>-1] \
                                        for l_idx in  ll_idx_phi_hmm_temp  ]

        ll_mat_TR = [np.array( [theta_var_coupled.TR[:,idx_phi,:] \
                    for idx_phi in l_idx_obs_phi]) for l_idx_obs_phi \
                    in ll_idx_phi_hmm_cropped_temp ]


        """ PLOT TRACE EXAMPLE """
        zp2 = zip(enumerate(ll_signal_hmm_cropped_temp),l_gamma_temp,
                            l_logP_temp)
        for (idx, signal), gamma, logP in zp2:
            plt_result = PlotResults(gamma, l_var, signal_model, signal,
                                     waveform = W, logP = None,
                                     temperature = temperature, cell = cell)
            plt_result.plotEverythingEsperance(False, idx)
            if idx==10:
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
        ll_nan_circadian_factor_hmm.extend(ll_nan_circadian_factor_hmm_temp)

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
        ll_nan_circadian_factor_hmm_clean \
                                    = copy.deepcopy(ll_nan_circadian_factor_hmm)


    """ PARAMETERS UPDATE """

    [F_up, pi_up, std_theta_up, sigma_em_circadian_up, ll_coef,
    std_amplitude_theta_up, std_background_theta_up, mu_amplitude_theta_up,
    mu_background_theta_up, W_up] = EM.run_EM(l_gamma_0, l_gamma, t_l_jP,
                                        theta_var_coupled,  ll_idx_phi_hmm,
                                        F, ll_signal_hmm,  amplitude_var,
                                        background_var, W,
                                        ll_idx_coef = F_no_coupling,
                                        only_F_and_pi = only_F_and_pi,
                                        lambd_parameter = lambda_parameter,
                                        lambd_2_parameter = lambda_2_parameter)

    if np.mean(l_logP)-lP<10**-9:
        print("diff:", np.mean(l_logP)-lP)
        print(print("average lopP:", np.mean(l_logP)))
        break
    else:
        lP = np.mean(l_logP)
        print("average lopP:", lP)
        l_lP.append(lP)

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

    """ PLOT COUPLING FUNCTION """
    plt.pcolormesh(theta_var_coupled.domain, theta_var_coupled.codomain,
                    F.T, cmap=bwr, vmin=-0.3, vmax=0.3)
    plt.xlim([0, 2*np.pi])
    plt.ylim([0, 2*np.pi])
    plt.colorbar()
    plt.xlabel("theta")
    plt.ylabel("phi")
    plt.show()
    plt.close()


plt.plot(l_lP)
plt.savefig("Parameters/Real/opt_parameters_div_"+str(temperature)+"_"\
                                                                 +cell+'.pdf')
#plt.show()
plt.close()


""""""""""""""""""""" WRAP PARAMETERS """""""""""""""""
pickle.dump( l_parameters, open( "Parameters/Real/opt_parameters_div_"\
                                        +str(temperature)+"_"+cell+".p", "wb"))

""""""""""""""""""""" PLOT FINAL COUPLING """""""""""""""""
plt.imshow(F.T, cmap=bwr, vmin=-0.3, vmax=0.3, interpolation='nearest',
                                origin='lower', extent=[0, 2*np.pi,0, 2*np.pi])
plt.colorbar()
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\phi$')
plt.title('Coupling Function')
plt.savefig("Results/PhaseSpace/Coupling_"+str(temperature)+"_"+cell+'.pdf')
plt.close()
