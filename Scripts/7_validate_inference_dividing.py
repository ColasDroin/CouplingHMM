# -*- coding: utf-8 -*-
""" This script is used to validate the optimization process for the parameters
obtained from dividing traces. """
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
from matplotlib import colors as mcolors
from mpl_toolkits import axes_grid1
import seaborn as sn
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
from PlotStochasticSpeedSpace import PlotStochasticSpeedSpace

###Import internal functions
from Functions.signal_model import signal_model
from Functions.create_hidden_variables import create_hidden_variables
from Functions.make_colormap import make_colormap
from Functions.create_coupling import build_coupling_array_from_2D_gaussian
from Functions.signal_model import signal_model
from Functions.display_parameters import display_parameters

#access main directory
os.chdir('..')

#nice plotting style
sn.set_style("whitegrid", {'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})
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

    record_no_coupling = sys.argv[6]
    if record_no_coupling=="True":
        record_no_coupling = True
    else:
        record_no_coupling = False

    infer_real_coupling = sys.argv[7]
    if infer_real_coupling=="True":
        infer_real_coupling = True
    else:
        infer_real_coupling = False


except:
    print("No shell input given, default arguments used")
    nb_iter = 30
    nb_traces = 50000
    size_block = 50
    cell = 'NIH3T3'
    temperature = None
    record_no_coupling = False
    infer_real_coupling = True


if record_no_coupling:
    print("record_no_coupling is True, "\
                                   +"therefore only one iteration will be made")
    nb_iter = 1
    print("record_no_coupling is True, "\
                               +"therefore infer_real_coupling is set to False")
    infer_real_coupling = False

path = 'Parameters/Real/opt_parameters_nodiv_'+str(temperature)+"_"+cell+'.p'
if infer_real_coupling:
    path = 'Parameters/Real/opt_parameters_div_'+str(temperature)+"_"+cell+'.p'

""""""""""""""""""""" LOAD OPTIMIZED PARAMETERS """""""""""""""""

with open(path, 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)

""""""""""""""""""""" LOAD COLORMAP """""""""""""""""
c = mcolors.ColorConverter().to_rgb
bwr = make_colormap(  [c('blue'), c('white'), 0.48, c('white'),
                       0.52,c('white'),  c('red')])

""""""""""""""""""""" DISPLAY PARAMETERS """""""""""""""""
display_parameters(l_parameters, show = True)

if not infer_real_coupling:
    """"""""""""""""""""" CREATE COUPLING """""""""""""""""
    if record_no_coupling:
        F = np.zeros((N_theta, N_phi))
    else:
        t_coor_1 = (1.5,4.5)
        mat_var_1 = np.array( [[0.5, 0 ], [0, 0.5]]   )
        t_coor_2 = (5,2)
        mat_var_2 = np.array( [[0.5, 0 ], [0, 0.5]]   )
        l_amp_1 = [-1,1]

        F = build_coupling_array_from_2D_gaussian(
                l_t_coor = [t_coor_1, t_coor_2],
                l_mat_var = [mat_var_1, mat_var_2],
                domainx = np.linspace(0,2*np.pi,N_theta, endpoint = False),
                domainy = np.linspace(0,2*np.pi,N_phi, endpoint = False),
                l_amp = l_amp_1)

    """"""""""""""""""""" UPDATE PARAMETERS """""""""""
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
    gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta,
    gamma_background_theta, l_boundaries_background_theta,
    F]



""""""""""""""""""""" PLOT THEORETICAL COUPLING """""""""""
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

plt.figure(figsize=(5*1.2,5*1.2))
ax = plt.gca()
im = plt.imshow(F.T, cmap=bwr, vmin=-0.3, vmax=0.3, origin='lower',
                interpolation = "spline16" , extent=[0,1,0,1])
add_colorbar(im, label = r'Acceleration ($rad.h^{-1}$)')
plt.xlabel(r'Circadian phase $\theta$')
plt.ylabel(r'Cell-cycle phase $\phi$')
plt.tight_layout()
plt.savefig("Results/Validation/Coupling_theoretical_"+str(temperature)+"_"\
            +cell+'_'+str(infer_real_coupling)+".pdf")
plt.show()
plt.close()

""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""
theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )
l_var = [theta_var_coupled, amplitude_var, background_var]

""""""""""""""""""""" COMPUTE NB_TRACES_MAX """""""""""""""""
if cell == 'NIH3T3':
    path =  "Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
else:
    path = "Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
dataClass=LoadData(path, nb_traces, temperature = temperature,
                    division = True, several_cell_cycles = True)
(ll_area_tot_flat, ll_signal_tot_flat, ll_nan_circadian_factor_tot_flat,
    ll_obs_phi_tot_flat, T_theta, T_phi) = dataClass.load()
#nb_traces_max = len(ll_signal_tot_flat)
nb_traces_max = 600
if nb_traces>nb_traces_max:
    print("CAUTION : too many traces, number of traces generated reduced to: ",
            nb_traces_max)
    nb_traces = nb_traces_max

""""""""""""""""""""" GENERATE COUPLED TRACES """""""""""
sim = HMMsim(  l_var, signal_model , sigma_em_circadian,  waveform = W ,
            dt=0.5, uniform = True )
ll_t_l_xi, ll_t_obs  =  sim.simulate_n_traces(nb_traces=nb_traces, tf = 70)

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
    #plt.show()
    plt.savefig("Results/Validation/Trace_div_"+str(idx_trace)+'_'+cell+'_'\
                 +str(temperature)+'.pdf')
    plt.close()
    if idx_trace>10:
        break


""""""""""""""""""""" PLOT GENERATED COUPLING """""""""""""""""
plt_space_theo = PlotStochasticSpeedSpace((lll_xi_circadian, lll_xi_nucleus),
                                           l_var, dt, w_phi, cell, temperature,
                                           cmap = bwr)
space_theta_th, space_phi_th, space_count_th \
                                = plt_space_theo.plotPhaseSpace(save_plot=False)


""""""""""""""""""""" DEFINE PARAMETERS FOR INFERENCE """""""""""""""""

pi = None
F = (np.random.rand( N_theta, N_phi)-0.5)*0.01


l_parameters = [dt, sigma_em_circadian, W, pi,
N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
gamma_amplitude_theta, l_boundaries_amplitude_theta,
N_background_theta, mu_background_theta, std_background_theta,
gamma_background_theta, l_boundaries_background_theta,
F]

only_F_and_pi = True
lambda_parameter = 2*10e-6
lambda_2_parameter = 0.005

""""""""""""""""""""" CORRECT INFERENCE BIAS """""""""""""""""
if not record_no_coupling:
    try:
        F_no_coupling = pickle.load( open( "Parameters/Misc/F_no_coupling_"\
                                    +str(temperature)+"_"+cell+'.p', "rb" ) )
        for idx_theta, theta in enumerate(theta_var_coupled.domain):
            F_no_coupling[idx_theta, :] \
                                = [np.mean(F_no_coupling[idx_theta, :])]*N_phi
    except:
        print("F_no_coupling not found, no bias correction applied")
        F_no_coupling = None
else:
    F_no_coupling = None


""""""""""""""""""""" RECREATE HIDDEN VARIABLES FOR INFERENCE """""""""""""""""
theta_var_coupled, amplitude_var, background_var \
                        = create_hidden_variables(l_parameters = l_parameters )

""""""""""""""""""""" CREATE ll_idx_obs_phi and ll_val_phi"""""""""""""""""
ll_val_phi =[np.array(ll_xi)[:,0] for  ll_xi in lll_xi_nucleus]
ll_idx_obs_phi = []
for l_obs in ll_val_phi:
    l_idx_obs_phi = []
    for obs in l_obs:
        l_idx_obs_phi.append( int(round(obs/(2*np.pi) \
                                * len(theta_var_coupled.codomain)))\
                                %len(theta_var_coupled.codomain))
    ll_idx_obs_phi.append(l_idx_obs_phi)

""""""""""""""""""""" DEFINE BLOCKS OF TRACES """""""""""""""""
ll_area_tot_flat = ll_obs_nucleus
ll_signal_tot_flat = ll_obs_circadian
ll_obs_phi_tot_flat = ll_val_phi

ll_area_tot = []
ll_signal_tot = []
ll_obs_phi_tot = []
first= True
zp = enumerate(zip(ll_area_tot_flat, ll_signal_tot_flat, ll_obs_phi_tot_flat))
for index, (l_area, l_signal,  l_obs_phi) in zp:
    if index%size_block==0:
        if not first:
            ll_area_tot.append(ll_area)
            ll_signal_tot.append(ll_signal)
            ll_obs_phi_tot.append(ll_obs_phi)

        else:
            first = False
        ll_area = [l_area]
        ll_signal = [l_signal]
        ll_obs_phi = [l_obs_phi]
    else:
        ll_area.append(l_area)
        ll_signal.append(l_signal)
        ll_obs_phi.append(l_obs_phi)
#get remaining trace
ll_area_tot.append(ll_area)
ll_signal_tot.append(ll_signal)
ll_obs_phi_tot.append(ll_obs_phi)


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


    for ll_signal, ll_obs_phi in zip(ll_signal_tot, ll_obs_phi_tot) :

        """ INITIALIZE AND RUN HMM """
        l_var = [theta_var_coupled, amplitude_var, background_var]
        hmm=HMM_SemiCoupled(l_var, ll_signal, sigma_em_circadian,
                            ll_val_phi = ll_obs_phi, waveform = W ,
                            pi = pi, crop = True )
        (l_gamma_0_temp, l_gamma_temp ,l_logP_temp,  ll_alpha,  ll_beta, l_E,
            ll_cnorm, ll_idx_phi_hmm_temp, ll_signal_hmm_temp,
            ll_nan_circadian_factor_hmm_temp) = hmm.run_em()

        #crop and create ll_mat_TR
        ll_signal_hmm_cropped_temp = [[s for s, idx in zip(l_s, l_idx) \
                            if idx>-1] for l_s, l_idx \
                            in  zip(ll_signal_hmm_temp ,ll_idx_phi_hmm_temp)]

        ll_idx_phi_hmm_cropped_temp = [[idx for idx in l_idx if idx>-1] \
                                        for l_idx in  ll_idx_phi_hmm_temp  ]

        ll_mat_TR = [np.array( [theta_var_coupled.TR[:,idx_phi,:] \
                     for idx_phi in l_idx_obs_phi]) for l_idx_obs_phi \
                     in ll_idx_phi_hmm_cropped_temp ]


        """ PLOT TRACE EXAMPLE """
        zp = zip(enumerate(ll_signal_hmm_cropped_temp),l_gamma_temp,
                            l_logP_temp)
        for (idx, signal), gamma, logP in zp:
            plt_result = PlotResults(gamma, l_var, signal_model, signal,
                                     waveform = W, logP = None,
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


    t_l_jP = (l_jP_phase, l_jP_amplitude,l_jP_background)


    """ PARAMETERS UPDATE """

    [F_up, pi_up, std_theta_up, sigma_em_circadian_up, ll_coef,
    std_amplitude_theta_up, std_background_theta_up, mu_amplitude_theta_up,
    mu_background_theta_up, W_up] = EM.run_EM(l_gamma_0, l_gamma, t_l_jP,
                                        theta_var_coupled,
                                        ll_idx_obs_phi, F, ll_signal_tot,
                                        amplitude_var,  background_var, W,
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

""""""""""""""""""""" SAVE RESULTS """""""""""""""""

if record_no_coupling:
    pickle.dump( F, open( "Parameters/Misc/F_no_coupling_"+str(temperature)\
                                                        +"_"+cell+'.p', "wb" ) )
    for idx_theta in range(N_theta):
        F[idx_theta, :] = np.mean(F[idx_theta, :])
    plt.pcolormesh(theta_var_coupled.domain, theta_var_coupled.codomain,
                                        F.T, cmap='bwr', vmin=-0.3, vmax=0.3)
    plt.xlim([0, 2*np.pi])
    plt.ylim([0, 2*np.pi])
    plt.colorbar()
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\phi$')
    plt.savefig("Parameters/Misc/F_no_coupling_"+str(temperature)\
                                                            +"_"+cell+'.pdf')
    plt.show()
    plt.close()
else:
    #record log-likelihood evolution
    plt.plot(l_lP)
    plt.savefig("Parameters/Silico/opt_parameters_div_"+str(temperature)\
                                +"_"+cell+'_'+str(infer_real_coupling)+'.pdf')
    plt.show()
    plt.close()
    """"""""""""""""""""" WRAP PARAMETERS """""""""""""""""
    pickle.dump( l_parameters, open( "Parameters/Silico/opt_parameters_div_"\
        +str(temperature)+"_"+cell+'_'+str(infer_real_coupling)+'.p', "wb" ) )

    """ SAVE FINAL COUPLING TO COMPARE WITH THEORETICAL ONE """
    plt.figure(figsize=(5*1.2,5*1.2))
    ax = plt.gca()
    im = plt.imshow(F.T, cmap=bwr, vmin=-0.3, vmax=0.3, origin='lower',
                    interpolation = "spline16" , extent=[0,1,0,1])
    add_colorbar(im, label = r'Acceleration ($rad.h^{-1}$)')
    plt.xlabel(r'Circadian phase $\theta$')
    plt.ylabel(r'Cell-cycle phase $\phi$')
    plt.tight_layout()
    plt.savefig("Results/Validation/F_inferred_"+str(temperature)+"_"\
                +cell+'_'+str(infer_real_coupling)+'.pdf')
    plt.show()
    plt.close()
