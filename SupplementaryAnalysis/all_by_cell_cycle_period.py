# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
# Import external modules
import numpy as np
import scipy.stats as st
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')  # to run the script on a distant server
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.animation as animation
import pandas as pd
import sys
import os
import pickle
import scipy
import matplotlib.colors as mcolors
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from scipy.fftpack import fft
import seaborn as sn

# Import internal modules
sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))

from Classes.LoadData import LoadData
from Classes.PlotResults import PlotResults
from Classes.HMM_SemiCoupled import HMM_SemiCoupled
import Classes.EM as EM
from Classes.DetSim import DetSim
from Classes.HMMsim import HMMsim
from Classes.PlotStochasticSpeedSpace import PlotStochasticSpeedSpace

from Functions.create_hidden_variables import create_hidden_variables
from Functions.display_parameters import display_parameters_from_file
from Functions.signal_model import signal_model
from Functions.plot_phase_space_density import plot_phase_space_density
from Functions.make_colormap import make_colormap


# colormap
c = mcolors.ColorConverter().to_rgb
bwr = make_colormap([c('blue'), c('white'), 0.48, c(
    'white'), 0.52, c('white'), c('red')])

np.set_printoptions(threshold=np.nan)

#nice plotting style
sn.set_style("whitegrid",
             {'xtick.direction': 'out',
              'xtick.major.size': 6.0,
              'xtick.minor.size': 3.0,
              'ytick.color': '.15',
              'ytick.direction': 'out',
              'ytick.major.size': 6.0,
              'ytick.minor.size': 3.0})

""""""""""""""""""""" FUNCTIONS """""""""""""""""""""
def all_by_cell_cycle_period(cell='NIH3T3', nb_traces=500, size_block=100):
    """
    Compute and plot various results (circadian speed, phase space, attractor
    etc.), for evoving cell-cycle speed.

    Parameters
    ----------
    cell : string
        Cell condition.
    nb_traces : integer
        Number of traces from which the inference is made.
    size_block : integer
        Size of the traces chunks (to save memory).
    """
    temperature = None  # get set to none after loading the parameters
    l_T_phi = np.arange(12, 49, 10)
    #l_T_phi = [22]
    ##################### LOAD OPTIMIZED PARAMETERS ##################
    path = '../Parameters/Real/opt_parameters_div_' + \
        str(temperature) + "_" + cell + '.p'
    with open(path, 'rb') as f:
            l_parameters = [dt, sigma_em_circadian, W, pi,
            N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
            N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
            N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
            gamma_amplitude_theta, l_boundaries_amplitude_theta,
            N_background_theta, mu_background_theta, std_background_theta,
            gamma_background_theta, l_boundaries_background_theta,
            F] = pickle.load(f)
    temperature = None

    ##################### DISPLAY PARAMETERS ##################
    display_parameters_from_file(path, show=True)

    for T_phi in l_T_phi:
        period_phi = T_phi
        w_phi = 2 * np.pi / T_phi
        l_parameters = [dt, sigma_em_circadian, W, pi, N_theta, std_theta,
            period_theta, l_boundaries_theta, w_theta, N_phi, std_phi,
            period_phi, l_boundaries_phi, w_phi, N_amplitude_theta,
            mu_amplitude_theta, std_amplitude_theta, gamma_amplitude_theta,
            l_boundaries_amplitude_theta, N_background_theta,
            mu_background_theta, std_background_theta, gamma_background_theta,
            l_boundaries_background_theta, F]
        print('### T-CELL-CYCLE = ' + str(T_phi) + '###')
        ##################### LOAD DATA ##################
        if cell == 'NIH3T3':
            path = "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
        else:
            path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"

        dataClass = LoadData(
            path,
            nb_traces,
            temperature=temperature,
            division=True,
            several_cell_cycles=False,
            remove_odd_traces=True)
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, std_T_theta, T_phi_obs, std_T_phi) = \
                          dataClass.load(period_phi=T_phi, load_annotation=True)
        ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v > 0] \
                                                          for l_peak in ll_peak]
        print(len(ll_signal), " traces kept")

        ##################### CREATE HIDDEN VARIABLES ##################
        theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters=l_parameters)
        l_var = [theta_var_coupled, amplitude_var, background_var]

        ##################### CREATE AND RUN HMM ##################
        hmm = HMM_SemiCoupled(
            l_var,
            ll_signal,
            sigma_em_circadian,
            ll_obs_phi,
            waveform=W,
            ll_nan_factor=ll_nan_circadian_factor,
            pi=pi,
            crop=True)
        l_gamma_div, l_logP_div = hmm.run(project=False)

        ##################### REMOVE BAD TRACES #####################
        try:
            Plim = np.percentile(l_logP_div, 10)
            idx_to_keep = [i for i, logP in enumerate(
                l_logP_div) if logP > Plim]
            l_gamma_div = [l_gamma_div[i] for i in idx_to_keep]
            ll_signal = [ll_signal[i] for i in idx_to_keep]
            ll_area = [ll_area[i] for i in idx_to_keep]
            l_logP_div = [l_logP_div[i] for i in idx_to_keep]
            ll_obs_phi = [ll_obs_phi[i] for i in idx_to_keep]
            ll_idx_cell_cycle_start = [
                ll_idx_cell_cycle_start[i] for i in idx_to_keep]
            ll_idx_peak = [ll_idx_peak[i] for i in idx_to_keep]
            print("Kept traces with div: ", len(idx_to_keep))

            ##################### CROP SIGNALS FOR PLOTTING ##################
            l_first = [[it for it, obj in enumerate(
                l_obs_phi) if obj != -1][0] for l_obs_phi in ll_obs_phi]
            l_last = [[len(l_obs_phi) -
                       it -
                       1 for it, obj in enumerate(l_obs_phi[::-
                                                            1]) if obj != -
                       1][0] for l_obs_phi in ll_obs_phi]
            ll_signal = [l_signal[first:last + 1] for l_signal,
                         first, last in zip(ll_signal, l_first, l_last)]
            ll_area = [l_area[first:last + 1]
                       for l_area, first, last in zip(ll_area, l_first, l_last)]
            ll_obs_phi = [l_obs_phi[first:last + 1] for l_obs_phi,
                          first, last in zip(ll_obs_phi, l_first, l_last)]
            ll_idx_cell_cycle_start = [[v for v in l_idx_cell_cycle_start \
                                                    if v >= first and v <= last]
                                       for l_idx_cell_cycle_start, first, last \
                                       in zip(ll_idx_cell_cycle_start, l_first,
                                                                        l_last)]
            ll_idx_peak = [[v for v in l_idx_peak if v >= first and v <= last]
                           for l_idx_peak, first, last in zip(ll_idx_peak,
                                                                l_first,
                                                                l_last)]

            ##################### CREATE ll_idx_obs_phi and ll_val_phi#########
            ll_idx_obs_phi = []
            for l_obs in ll_obs_phi:
                l_idx_obs_phi = []
                for obs in l_obs:
                    l_idx_obs_phi.append(int(round(obs /(2 *np.pi) * \
                                        len(theta_var_coupled.codomain))) % \
                                                len(theta_var_coupled.codomain))
                ll_idx_obs_phi.append(l_idx_obs_phi)


            # ##################### PLOT FITS ##################
            # for (idx, signal), gamma, logP, area, l_obs_phi,
            #    l_idx_cell_cycle_start,
            #     l_idx_peak in zip(enumerate(ll_signal),l_gamma_div,l_logP_div,
            #     ll_area, ll_obs_phi, ll_idx_cell_cycle_start, ll_idx_peak):
            #     plt_result = PlotResults(gamma, l_var, signal_model, signal,
            #                 waveform = W, logP = logP,
            #                 temperature = temperature,
            #                 cell = cell)
            #     E_model, E_theta, E_A, E_B = \
            #                    plt_result.plotEverythingEsperance( False, idx)


            ##################### COMPUTE CIRCADIAN SPEED ##################
            dic_circadian_speed = {}
            for idx_theta in range(N_theta):
                dic_circadian_speed[idx_theta] = []
            for (idx, signal), gamma, logP, area, l_obs_phi in zip(
                    enumerate(ll_signal), l_gamma_div, l_logP_div, ll_area,
                              ll_obs_phi):
                plt_result = PlotResults(
                    gamma,
                    l_var,
                    signal_model,
                    signal,
                    waveform=W,
                    logP=logP,
                    temperature=temperature,
                    cell=cell)
                l_E_model, l_E_theta, l_E_A, l_E_B = \
                                plt_result.plotEverythingEsperance(False, idx)
                for theta_1_norm, theta_2_norm in zip(
                        l_E_theta[:-1], l_E_theta[1:]):
                    theta_1 = theta_1_norm * 2 * np.pi
                    theta_2 = theta_2_norm * 2 * np.pi
                    if theta_2 - theta_1 > -np.pi and theta_2 - theta_1 < np.pi:
                        speed = (theta_2 - theta_1) / 0.5
                    elif theta_2 - theta_1 < -np.pi:
                        speed = (theta_2 + 2 * np.pi - theta_1) / 0.5
                    else:
                        speed = (theta_2 - theta_1 - 2 * np.pi) / 0.5
                    idx_theta = int(round(
                        theta_1 / (2 * np.pi) * \
                                            len(theta_var_coupled.codomain))) \
                                            % len(theta_var_coupled.codomain)
                    dic_circadian_speed[idx_theta].append(speed)
            ##################### PLOT SPEED VS PHASE ##################
            l_mean_speed = []
            l_std_speed = []
            for idx_theta in range(N_theta):
                l_mean_speed.append(np.mean(dic_circadian_speed[idx_theta]))
                l_std_speed.append(np.std(dic_circadian_speed[idx_theta]))

            plt.clf()
            # plt.figure(figsize=(5,10))
            plt.errorbar(theta_var_coupled.codomain, l_mean_speed,
                         yerr=l_std_speed, fmt='o', label='data')
            plt.ylim([0, 0.6])
            plt.xlabel(r'Circadian phase')
            plt.ylabel(r'Circadian speed')
            plt.legend()
            # plt.title(r'Data')
            # plt.show()
            plt.savefig(
                '../Results/AllByCellCycle/CircadianSpeed_data_' +
                str(T_phi) +
                '.pdf')
            plt.close()

            ##################### GET DATA ATTRACTOR ##################
            M_at = plot_phase_space_density(
                l_var, l_gamma_div, ll_idx_obs_phi, F_superimpose=F, save=False)

        except BaseException:
            pass

        ##################### GET DETERMINISTIC ATTRACTOR AND REPELLER #########
        detSim = DetSim(l_parameters, cell, temperature)

        # get attractor
        ll_phase_theta, ll_phase_phi = detSim.plot_trajectory(
            ti=5000, tf=5100, rand=True, save=False, K=1, T_phi=T_phi)
        # get repeller
        ll_phase_theta_rep, ll_phase_phi_rep = detSim.plot_trajectory(
            ti=5000, tf=-5100, rand=True, save=False, K=1, T_phi=T_phi)

        ##################### PLOT COUPLING ##################
        plt.figure(figsize=(10, 10))
        plt.imshow(
            F.T,
            cmap=bwr,
            vmin=-0.3,
            vmax=0.3,
            interpolation='spline16',
            origin='lower',
            extent=[
                0,
                1,
                0,
                1])
        # plt.colorbar()
        plt.xlabel(r'Circadian phase $\theta$')
        plt.ylabel(r'Cell-cycle phase $\phi$')
        # plt.show()
        # plt.close()

        ##################### PLOT VECTORFIELD ##################
        X = np.linspace(0, 2 * np.pi, F.shape[0], endpoint=False)
        Y = np.linspace(0, 2 * np.pi, F.shape[1], endpoint=False)

        U = 2 * np.pi / period_theta + F.T
        V = np.empty((F.shape[0], F.shape[1]))
        V.fill(2 * np.pi / T_phi)

        # sample to reduce the number of arrows
        l_idx_x = [x for x in range(0, F.shape[0], 4)]
        l_idx_y = [x for x in range(0, F.shape[1], 4)]
        X = X[l_idx_x]
        Y = Y[l_idx_y]
        U = [u[l_idx_y] for u in U[l_idx_x]]
        V = [v[l_idx_y] for v in V[l_idx_x]]
        C = [c[l_idx_y] for c in F.T[l_idx_x]]

        ##################### PLOT DETERMINISTIC ATTRACTOR AND REPELLER ########

        # plot vectorfield
        plt.quiver(np.array(X) / (2 * np.pi), np.array(Y) /
                   (2 * np.pi), U, V, C, alpha=.5, cmap='cool')
        plt.quiver(np.array(X) / (2 * np.pi), np.array(Y) / (2 * np.pi), U,
                   V, edgecolor='k', facecolor='None', linewidth=.1)

        # plot attractor
        for l_phase_theta, l_phase_phi in zip(ll_phase_theta, ll_phase_phi):
            plt.plot(np.array(l_phase_theta) / (2 * np.pi),
                     np.array(l_phase_phi) / (2 * np.pi),
                     lw=2,
                     color='green',
                     alpha=0.1,
                     label='Attractor')

        # plot repeller
        for l_phase_theta_rep, l_phase_phi_rep in zip(
                ll_phase_theta_rep, ll_phase_phi_rep):
            plt.plot(np.array(l_phase_theta_rep) / (2 * np.pi),
                     np.array(l_phase_phi_rep) / (2 * np.pi),
                     lw=2,
                     color='red',
                     alpha=0.1,
                     label='Repeller')

        print(np.array(l_phase_theta) / (2 * np.pi))
        print(np.array(l_phase_phi) / (2 * np.pi))
        # plt.show()
        # plt.close()

        ##################### PLOT STOCHASTIC ATTRACTOR ##################
        sim = HMMsim(l_var, signal_model, sigma_em_circadian,
                     waveform=W, dt=0.5, uniform=True, T_phi=T_phi)
        ll_t_l_xi, ll_t_obs = sim.simulate_n_traces(nb_traces=100, tf=1000)

        ### CROP BEGINNING OF THE TRACES ###
        ll_t_l_xi = [l_t_l_xi[-700:] for l_t_l_xi in ll_t_l_xi]
        ll_t_obs = [l_t_obs[-700:] for l_t_obs in ll_t_obs]
        ##################### REORDER VARIABLES ##################
        ll_obs_circadian = []
        ll_obs_nucleus = []
        lll_xi_circadian = []
        lll_xi_nucleus = []
        for idx, (l_t_l_xi, l_t_obs) in enumerate(zip(ll_t_l_xi, ll_t_obs)):
            ll_xi_circadian = [t_l_xi[0] for t_l_xi in l_t_l_xi]
            ll_xi_nucleus = [t_l_xi[1] for t_l_xi in l_t_l_xi]
            l_obs_circadian = np.array(l_t_obs)[:, 0]
            l_obs_nucleus = np.array(l_t_obs)[:, 1]
            ll_obs_circadian.append(l_obs_circadian)
            ll_obs_nucleus.append(l_obs_nucleus)
            lll_xi_circadian.append(ll_xi_circadian)
            lll_xi_nucleus.append(ll_xi_nucleus)

        omega_phi = 2 * np.pi / T_phi
        sim = PlotStochasticSpeedSpace(
            (lll_xi_circadian,
             lll_xi_nucleus),
            l_var,
            dt,
            omega_phi,
            cell,
            temperature,
            cmap=None)
        _, _, M_sim = sim.getPhaseSpace()
        M_sim_theta = np.array([l / np.sum(l) for l in M_sim]).T

        l_theta = []
        l_phi = []
        for idx_phi, phi in enumerate(theta_var_coupled.domain):
            #l_theta.append( np.angle(np.sum(np.multiply(M_sim_theta[idx_phi],
                                            #np.exp(1j*np.array(
                                            #theta_var_coupled.domain)))))\
                                            #%(2*np.pi) )
            l_theta.append(
                theta_var_coupled.domain[np.argmax(M_sim_theta[idx_phi])])
            l_phi.append(phi)

        abs_d_data_x = np.abs(np.diff(l_theta))
        mask_x = np.hstack(
            [abs_d_data_x > abs_d_data_x.mean() + 3 * abs_d_data_x.std(),
                                                                    [False]])
        masked_l_theta = np.array(
            [x if not m else np.nan for x, m in zip(l_theta, mask_x)])

        abs_d_data_x = np.abs(np.diff(l_phi))
        mask_x = np.hstack(
            [abs_d_data_x > abs_d_data_x.mean() + 3 * abs_d_data_x.std(),
                                                                    [False]])
        masked_l_phi = np.array(
            [x if not m else np.nan for x, m in zip(l_phi, mask_x)])

        l_theta = masked_l_theta
        l_phi = masked_l_phi

        #plt.plot(np.array(l_theta)/(2*np.pi),
        #         np.array(l_phi)/(2*np.pi),
        #          lw = 2, color = 'grey', alpha = 1.,
        #           label = 'E[Stochastic simulation]')
        # plt.show()
        # plt.close()
        #print("stochastic attractor done")
        try:

            ##################### PLOT ATTRACTOR OF THE DATA ##################

            l_theta = []
            l_phi = []
            for idx_phi, phi in enumerate(theta_var_coupled.domain):
                l_theta.append(
                    np.angle(
                        np.sum(
                            np.multiply(
                                M_at.T[idx_phi],
                                np.exp(1j *np.array(
                                        theta_var_coupled.domain))))) %
                    (2 *np.pi))
                l_phi.append(phi)

            abs_d_data_x = np.abs(np.diff(l_theta))
            mask_x = np.hstack(
                [abs_d_data_x > abs_d_data_x.mean() + 3 * abs_d_data_x.std(),
                                                                       [False]])
            masked_l_theta = np.array(
                [x if not m else np.nan for x, m in zip(l_theta, mask_x)])

            abs_d_data_x = np.abs(np.diff(l_phi))
            mask_x = np.hstack(
                [abs_d_data_x > abs_d_data_x.mean() + 3 * abs_d_data_x.std(),
                                                                       [False]])
            masked_l_phi = np.array(
                [x if not m else np.nan for x, m in zip(l_phi, mask_x)])

            l_theta = masked_l_theta
            l_phi = masked_l_phi
            plt.plot(np.array(l_theta) / (2 * np.pi), np.array(l_phi) /
                     (2 * np.pi), lw=2, color='black', alpha=1, label='E[Data]')
        except BaseException:
            print('attractor of the data not plotted !')

        # plt.legend()
        plt.savefig('../Results/AllByCellCycle/all_' + str(T_phi) + '.pdf')
        plt.show()
        plt.close()

        ##################### PLOT LOCATION ON TONGUES (SET APPROPRIATE
        #PARAMETERS DEPENDING ON WHAT WAS DONE IN THE SIM...) ##################
        ll_arnold = pickle.load(
            open(
                "../Results/DetSilico/arnold_" +
                cell +
                '_' +
                str(temperature) +
                ".p",
                "rb"))
        speed_space = np.linspace(
            2 * np.pi / (2.5 * 24), 2 * np.pi / (24 / 3), 800)
        #speed_space = np.linspace(2*np.pi/(3*24), 2*np.pi/(24/3), 400)
        period_space = list(reversed(2 * np.pi / speed_space))
        #coupling_space = np.linspace(0.,3., 200)
        coupling_space = np.linspace(0., 2., 150)

        plt.pcolormesh(period_space[:-1], coupling_space, ll_arnold,
                       cmap='binary', vmin=0, vmax=3, shading='gouraud')
        plt.scatter([T_phi], [1], color='red')
        # plt.colorbar()
        plt.xlim([period_space[0], period_space[-2]])
        plt.ylim([coupling_space[0], coupling_space[-1]])
        plt.xlabel(r"$T_{\phi}:T_{\theta}$")
        plt.ylabel("K")
        locs, labels = plt.xticks()
        plt.xticks([12, 2 / 3 * 24, 24, 24 * 11 / 8, 48],
                   ['2:1', '3:2', '1:1', '8:11', '1:2'])
        plt.savefig('../Results/AllByCellCycle/tongue_' + str(T_phi) + '.pdf')
        plt.show()
        plt.close()

        ##################### FFT ON VERY LONG DETERMINISTIC TRACE #############
        waveform_temp = W + W[0]
        waveform_func = interp1d(np.linspace(
            0, 2 * np.pi, len(waveform_temp), endpoint=True), waveform_temp)

        tspan, vect_Y = detSim.simulate(
            tf=10000, full_simulation=False, rand=True)
        l_signal = waveform_func(vect_Y[1000:, 0] % (2 * np.pi))

        Fs = 2.0  # sampling rate
        n = len(l_signal)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq = frq[range(int(n / 2))]  # one side frequency range
        Y = np.fft.fft(l_signal) / n  # fft computing and normalization
        Y = Y[range(int(n / 2))]

        plt.plot(frq, abs(Y))
        plt.xlim(1 / 40, 1 / 10)
        plt.ylim(0, 0.5)
        plt.xticks([1/40, 1/36, 1/32, 1/28, 1/24, 1/20, 1/16, 1/12, 1/10],
                   ['40','36','32','28','24','20','16','12','10'])
        # plt.grid()
        plt.xlabel('Circadian period (h)')
        plt.savefig('../Results/AllByCellCycle/det_fft_' + str(T_phi) + '.pdf')
        plt.show()
        plt.close()

        ##################### PLOT SPEED VS PHASE ##################
        dic_circadian_speed = {}
        for idx_theta in range(N_theta):
            dic_circadian_speed[idx_theta] = []

        for theta_1, theta_2 in zip(vect_Y[:-1, 0], vect_Y[1:, 0]):
            if theta_2 - theta_1 > -np.pi and theta_2 - theta_1 < np.pi:
                speed = (theta_2 - theta_1) / 0.5
            elif theta_2 - theta_1 < -np.pi:
                speed = (theta_2 + 2 * np.pi - theta_1) / 0.5
            else:
                speed = (theta_2 - theta_1 - 2 * np.pi) / 0.5
            idx_theta = int(round(theta_1 / (2 * np.pi) *
                                  len(theta_var_coupled.codomain))) \
                                  % len(theta_var_coupled.codomain)
            dic_circadian_speed[idx_theta].append(speed)

        l_mean_speed = []
        l_std_speed = []
        for idx_theta in range(N_theta):
            l_mean_speed.append(np.mean(dic_circadian_speed[idx_theta]))
            l_std_speed.append(np.std(dic_circadian_speed[idx_theta]))

        plt.errorbar(theta_var_coupled.codomain, l_mean_speed,
                     yerr=l_std_speed, fmt='o', label='ODE sim')
        plt.ylim([0, 0.6])
        plt.xlabel(r'Circadian phase')
        plt.ylabel(r'Circadian speed')
        plt.legend()
        #plt.title("Deterministic simulation")
        plt.savefig(
            '../Results/AllByCellCycle/CircadianSpeed_det_' +
            str(T_phi) +
            '.pdf')
        plt.close()

        #plt.plot(tspan[:80], l_signal[-80:])
        # plt.show()
        # plt.close()

        ##################### FFT ON VERY LONG STOCHASTIC TRACE ################
        sim = HMMsim(l_var, signal_model, std_em=0., waveform=W,
                     dt=0.5, uniform=True, T_phi=T_phi)
        l_t_l_xi, l_t_obs = sim.simulate(tf=20000)
        l_obs_circadian = np.array(l_t_obs)[1000:, 0]
        l_phase_theta = [t_l_xi[0][0] for t_l_xi in l_t_l_xi]
        # print(l_phase_theta)
        ##################### PLOT SPEED VS PHASE ##################
        dic_circadian_speed = {}
        for idx_theta in range(N_theta):
            dic_circadian_speed[idx_theta] = []

        for theta_1, theta_2 in zip(l_phase_theta[:-1], l_phase_theta[1:]):
            if theta_2 - theta_1 > -np.pi and theta_2 - theta_1 < np.pi:
                speed = (theta_2 - theta_1) / 0.5
            elif theta_2 - theta_1 < -np.pi:
                speed = (theta_2 + 2 * np.pi - theta_1) / 0.5
            else:
                speed = (theta_2 - theta_1 - 2 * np.pi) / 0.5
            idx_theta = int(round(theta_1 / (2 * np.pi) *
                                  len(theta_var_coupled.codomain))) \
                                  % len(theta_var_coupled.codomain)
            dic_circadian_speed[idx_theta].append(speed)

        l_mean_speed = []
        l_std_speed = []
        for idx_theta in range(N_theta):
            l_mean_speed.append(np.mean(dic_circadian_speed[idx_theta]))
            l_std_speed.append(np.std(dic_circadian_speed[idx_theta]))

        plt.errorbar(theta_var_coupled.codomain, l_mean_speed,
                     yerr=l_std_speed, fmt='o', label='MC sim')
        plt.ylim([0, 0.6])
        plt.xlabel(r'Circadian phase')
        plt.ylabel(r'Circadian speed')
        plt.legend()
        #plt.title("Stochastic simulation")
        plt.savefig(
            '../Results/AllByCellCycle/CircadianSpeed_stoch_' +
            str(T_phi) +
            '.pdf')
        plt.close()

        Fs = 2.0  # sampling rate
        n = len(l_obs_circadian)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq = frq[range(int(n / 2))]  # one side frequency range
        Y = np.fft.fft(l_obs_circadian) / n  # fft computing and normalization
        Y = Y[range(int(n / 2))]

        plt.plot(frq, abs(Y))
        plt.xlim(1 / 40, 1 / 10)
        plt.ylim(0, 0.5)
        plt.xticks([1/40, 1/36, 1/32, 1/28, 1/24, 1/20, 1/16, 1/12, 1/10],
                   ['40','36','32','28','24','20','16','12','10'])
        plt.xlabel('Circadian period (h)')
        # plt.grid()
        plt.savefig(
            '../Results/AllByCellCycle/stoch_fft_' +
            str(T_phi) +
            '.pdf')
        plt.show()
        plt.close()


def fourier_by_cell_cycle(cell='NIH3T3'):
    """
    Compute and plot the fourier transform of a very long simulated trace
     for evoving cell-cycle speed.

    Parameters
    ----------
    cell : string
        Cell condition.
    """
    temperature = None  # get set to none after loading the parameters
    #l_T_phi = np.arange(5,72,0.1)
    #l_T_phi = [22]
    l_T_phi = [48, 36, 24, 16, 12]
    ##################### LOAD OPTIMIZED PARAMETERS ##################
    path = '../Parameters/Real/opt_parameters_div_' + \
        str(temperature) + "_" + cell + '.p'
    with open(path, 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)
    temperature = None

    ##################### DISPLAY PARAMETERS ##################
    display_parameters_from_file(path, show=True)

    fig, arr_ax = plt.subplots(
        len(l_T_phi), 2, sharex=False, sharey=False, figsize=(10, 10))

    for idx_T, T_phi in enumerate(l_T_phi):
        period_phi = T_phi
        w_phi = 2 * np.pi / T_phi
        l_parameters = [dt, sigma_em_circadian, W, pi, N_theta, std_theta,
            period_theta, l_boundaries_theta, w_theta, N_phi, std_phi,
            period_phi, l_boundaries_phi, w_phi, N_amplitude_theta,
            mu_amplitude_theta, std_amplitude_theta, gamma_amplitude_theta,
            l_boundaries_amplitude_theta, N_background_theta,
            mu_background_theta, std_background_theta, gamma_background_theta,
            l_boundaries_background_theta, F]

        ##################### CREATE HIDDEN VARIABLES ##################
        theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters=l_parameters)
        l_var = [theta_var_coupled, amplitude_var, background_var]

        ##################### FFT ON VERY LONG DETERMINISTIC TRACE #############
        detSim = DetSim(l_parameters, cell, temperature)
        waveform_temp = W + W[0]
        waveform_func = interp1d(np.linspace(
            0, 2 * np.pi, len(waveform_temp), endpoint=True), waveform_temp)

        tspan, vect_Y = detSim.simulate(
            tf=20000, full_simulation=False, rand=True)
        l_signal = waveform_func(vect_Y[1000:, 0] % (2 * np.pi))

        Fs = 2.0  # sampling rate
        n = len(l_signal)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq_det = frq[range(int(n / 2))]  # one side frequency range
        Y = np.fft.fft(l_signal) / n  # fft computing and normalization
        Y_det = Y[range(int(n / 2))]
        arr_ax[idx_T, 0].set_ylabel(r'$T_\phi=$' + str(T_phi))
        arr_ax[idx_T, 0].axvline(1 / 24, color='green', alpha=0.8, ls='--')
        arr_ax[idx_T, 0].axvline(1 / T_phi, color='black', alpha=0.8, ls='--')
        #arr_ax[idx_T, 0].axvline(1/T_phi/2, color = 'grey')
        #arr_ax[idx_T, 0].axvline(1/T_phi*2, color = 'grey')
        #arr_ax[idx_T, 0].axvline(1/T_phi/3, color = 'grey')
        #arr_ax[idx_T, 0].axvline(1/T_phi*3, color = 'grey')

        arr_ax[idx_T, 0].plot(frq_det, abs(Y_det))
        arr_ax[idx_T, 0].set_xlim(1 / 50, 1 / 10)
        arr_ax[idx_T, 0].set_ylim(-0.01, 0.3)
        arr_ax[idx_T, 0].set_xticks(
            [1 / 48, 1 / 36, 1 / 28, 1 / 24, 1 / 20, 1 / 16, 1 / 12, 1 / 10])
        arr_ax[idx_T, 0].set_xticklabels(
            ['48', '36', '28', '24', '20', '16', '12', '10'])
        if idx_T == len(l_T_phi) - 1:
            arr_ax[idx_T, 0].set_xlabel('Circadian period (h)')
        if idx_T == 0:
            arr_ax[idx_T, 0].set_title('FFT on deterministic simulations')

        # grid for determinstic heatmap
        try:
            grid_det[idx_T, :] = abs(Y_det)
        except BaseException:
            grid_det = np.zeros((len(l_T_phi), len(abs(Y_det))))
            grid_det[idx_T, :] = abs(Y_det)
        ##################### FFT ON VERY LONG STOCHASTIC TRACE ################
        sim = HMMsim(l_var, signal_model, std_em=0., waveform=W,
                     dt=0.5, uniform=True, T_phi=T_phi)
        l_t_l_xi, l_t_obs = sim.simulate(tf=20000)
        l_obs_circadian = np.array(l_t_obs)[1000:, 0]
        l_phase_theta = [t_l_xi[0][0] for t_l_xi in l_t_l_xi]

        Fs = 2.0  # sampling rate
        n = len(l_obs_circadian)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq_stoc = frq[range(int(n / 2))]  # one side frequency range
        Y = np.fft.fft(l_obs_circadian) / n  # fft computing and normalization
        Y_stoc = Y[range(int(n / 2))]

        arr_ax[idx_T, 1].axvline(1 / 24, color='green', alpha=0.8, ls='--')
        arr_ax[idx_T, 1].axvline(1 / T_phi, color='black', alpha=0.8, ls='--')
        #arr_ax[idx_T, 1].axvline(1/T_phi/2, color = 'grey')
        #arr_ax[idx_T, 1].axvline(1/T_phi*2, color = 'grey')
        #arr_ax[idx_T, 1].axvline(1/T_phi/3, color = 'grey')
        #arr_ax[idx_T, 1].axvline(1/T_phi*3, color = 'grey')

        arr_ax[idx_T, 1].plot(frq_stoc, abs(Y_stoc))

        arr_ax[idx_T, 1].set_xlim(1 / 50, 1 / 10)
        arr_ax[idx_T, 1].set_ylim(-0.01, 0.3)
        arr_ax[idx_T, 1].set_xticks(
            [1 / 48, 1 / 36, 1 / 28, 1 / 24, 1 / 20, 1 / 16, 1 / 12, 1 / 10])
        arr_ax[idx_T, 1].set_xticklabels(
            ['48', '36', '28', '24', '20', '16', '12', '10'])
        arr_ax[idx_T, 1].set_yticklabels([])

        if idx_T == len(l_T_phi) - 1:
            arr_ax[idx_T, 1].set_xlabel('Circadian period (h)')
        if idx_T == 0:
            arr_ax[idx_T, 1].set_title('FFT on stochastic simulations')

        # grid for stochastic heatmap
        try:
            grid_sto[idx_T, :] = abs(Y_stoc)
        except BaseException:
            grid_sto = np.zeros((len(l_T_phi), len(abs(Y_stoc))))
            grid_sto[idx_T, :] = abs(Y_stoc)

        ##################### POSITION ON ARNOLD TONGUE ##################
        ###
        #ll_arnold =  pickle.load(open( "../Results/DetSilico/arnold_"+cell\
        #                +'_'+str(temperature)+".p", "rb" ) )
        var_test = pickle.load(open ("../Results/DetSilico/arnold_bis2_"+cell+\
                    '_'+str(temperature)+".p", "rb" ) )
        speed_space = np.linspace(2*np.pi/(2.5*24), 2*np.pi/(24/3), 800)
        period_space = list(reversed(2*np.pi/speed_space))
        coupling_space = np.linspace(0.,2., 150)


        # arr_ax[idx_T, 0].pcolormesh(period_space[:-1], coupling_space,
        #                             ll_arnold, cmap = 'binary', vmin = 0,
        #                             vmax = 3, shading='gouraud')
        # arr_ax[idx_T, 0].scatter([T_phi], [1], color = 'red')
        # arr_ax[idx_T, 0].set_xlim([period_space[0], period_space[-2]])
        # arr_ax[idx_T, 0].set_ylim([coupling_space[0], coupling_space[-1]])
        # arr_ax[idx_T, 0].set_xlabel(r"$T_{\phi}:T_{\theta}$")
        # arr_ax[idx_T, 0].set_ylabel("K")
        # arr_ax[idx_T, 0].set_xticks([12,2/3*24, 24, 24*4/3, 48])
        # arr_ax[idx_T, 0].set_xticklabels(['2:1', '3:2','1:1', '3:4', '1:2'])
        # if idx_T==0:
        #     arr_ax[idx_T, 0].set_title('Location on Arnold tongues')


        arr_ax[idx_T, 0].pcolormesh(period_space, coupling_space,
                                    var_test[:,::-1], cmap = 'binary', vmin = 0,
                                    vmax = 10**-8, shading='gouraud')
        arr_ax[idx_T, 0].scatter([T_phi], [1], color = 'red')
        arr_ax[idx_T, 0].set_xlim([period_space[0], period_space[-1]])
        arr_ax[idx_T, 0].set_xlabel(r"$T_{\phi}:T_{\theta}$")
        arr_ax[idx_T, 0].set_ylabel("K")
        arr_ax[idx_T, 0].set_xticks([12,2/3*24, 24, 24*4/3, 48])
        arr_ax[idx_T, 0].set_xticklabels(['2:1', '3:2','1:1', '3:4', '1:2'])
        if idx_T==0:
            arr_ax[idx_T, 0].set_title('Location on Arnold tongues')
        ###
    plt.tight_layout()
    plt.savefig('../Results/AllByCellCycle/fft.pdf')
    plt.close()




    # create heatmaps
    plt.figure(figsize=(5, 5))
    plt.pcolormesh(
        frq_det,
        1 /
        np.array(l_T_phi),
        grid_det,
        cmap='coolwarm',
        vmin=0,
        vmax=10**-
        1)  # , shading='gouraud')
    plt.colorbar()
    plt.xlim([1 / 50, 1 / 10])
    plt.ylim([1 / 50, 1 / 10])
    plt.xlabel(r"$T_{\theta}$")
    plt.ylabel(r'$T_{\phi}$')
    locs, labels = plt.xticks()
    plt.xticks([1 / 48, 1 / 40, 1 / 32, 1 / 24, 1 / 20, 1 / 16, 1 /
                12, 1 / 10], ['48', '40', '32', '24', '20', '16', '12', '10'])
    plt.yticks([1 / 48, 1 / 40, 1 / 32, 1 / 24, 1 / 20, 1 / 16, 1 /
                12, 1 / 10], ['48', '40', '32', '24', '20', '16', '12', '10'])
    plt.savefig('../Results/AllByCellCycle/fft_detheat.pdf')
    plt.show()
    plt.close()

    # create heatmaps
    plt.figure(figsize=(5, 5))
    plt.pcolormesh(
        frq_stoc,
        1 /
        np.array(l_T_phi),
        grid_sto,
        cmap='coolwarm',
        vmin=0,
        vmax=10**-
        1)  # , shading='gouraud')
    plt.colorbar()
    plt.xlim([1 / 50, 1 / 10])
    plt.ylim([1 / 50, 1 / 10])
    plt.xlabel(r"$T_{\theta}$")
    plt.ylabel(r'$T_{\phi}$')
    locs, labels = plt.xticks()
    plt.xticks([1 / 48, 1 / 40, 1 / 32, 1 / 24, 1 / 20, 1 / 16, 1 /
                12, 1 / 10], ['48', '40', '32', '24', '20', '16', '12', '10'])
    plt.yticks([1 / 48, 1 / 40, 1 / 32, 1 / 24, 1 / 20, 1 / 16, 1 /
                12, 1 / 10], ['48', '40', '32', '24', '20', '16', '12', '10'])
    plt.savefig('../Results/AllByCellCycle/fft_stoheat.pdf')
    plt.show()
    plt.close()

    dic_save = {'frq_det': frq_det, 'grid_det': grid_det,
                'l_T_phi': l_T_phi, 'frq_stoc': frq_stoc, 'grid_sto': grid_sto}
    path = '../Results/AllByCellCycle/save_fft.p'
    pickle.dump(dic_save, open(path, "wb"))


def anim_fourier(cell='NIH3T3'):
    """
    Compute and animate the fourier transform of a very long simulated trace
     for evoving cell-cycle speed.

    Parameters
    ----------
    cell : string
        Cell condition.
    """
    temperature = None  # get set to none after loading the parameters
    l_T_phi = [float("{0:.2f}".format(x))
               for x in np.arange(24 / 3, 2.5 * 24, 0.05)]
    #l_T_phi = [22]
    ##################### LOAD OPTIMIZED PARAMETERS ##################
    path = '../Parameters/Real/opt_parameters_div_' + \
        str(temperature) + "_" + cell + '.p'
    with open(path, 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)
    temperature = None

    ##################### DISPLAY PARAMETERS ##################
    display_parameters_from_file(path, show=False)

    ##################### INITIAL PLOT ##################
    fig, arr_ax = plt.subplots(
        1, 3, sharex=False, sharey=False, figsize=(20, 10))

    ll_arnold = pickle.load(
        open(
            "../Results/DetSilico/arnold_" +
            cell +
            '_' +
            str(temperature) +
            ".p",
            "rb"))
    var_test = pickle.load(
        open(
            "../Results/DetSilico/arnold_bis2_" +
            cell +
            '_' +
            str(temperature) +
            ".p",
            "rb"))
    speed_space = np.linspace(2 * np.pi / (2.5 * 24),
                              2 * np.pi / (24 / 3), 800)
    period_space = list(reversed(2 * np.pi / speed_space))
    coupling_space = np.linspace(0., 2., 150)

    arr_ax[0].pcolormesh(period_space, coupling_space, var_test[:, ::-1],
                         cmap='binary', vmin=0, vmax=10**-8, shading='gouraud')
    arr_ax[0].set_xlim([period_space[0], period_space[-1]])

    #arr_ax[0].pcolormesh(period_space[:-1], coupling_space, ll_arnold,
    #cmap = 'binary', vmin = 0, vmax = 3, shading='gouraud')
    arr_ax[1].axvline(1 / 24, color='green')
    arr_ax[2].axvline(1 / 24, color='green')
    point_arnold = arr_ax[0].scatter([], [], color='red')
    line_harm_det, = arr_ax[1].plot([], [], lw=1, color='grey')
    line_harm_stoc, = arr_ax[2].plot([], [], lw=1, color='grey')
    line_det, = arr_ax[1].plot([], [], lw=2)
    line_stoc, = arr_ax[2].plot([], [], lw=2)

    ##################### INITIAL DATA ##################

    def init():
        arr_ax[2].set_xlim(1 / 48, 1 / 10)
        arr_ax[2].set_ylim(-0.01, 0.3)
        arr_ax[1].set_xlim(1 / 48, 1 / 10)
        arr_ax[1].set_ylim(-0.01, 0.3)
        arr_ax[0].set_xlim([period_space[0], period_space[-2]])
        arr_ax[0].set_ylim([coupling_space[0], coupling_space[-1]])

        arr_ax[1].grid(True)
        arr_ax[2].grid(True)

        arr_ax[0].set_xlabel(r"$T_{\phi}:T_{\theta}$")
        arr_ax[0].set_ylabel("K")
        arr_ax[0].set_xticks([12, 2/3 * 24, 24, 24 * 4/3, 48])
        arr_ax[0].set_xticklabels(['2:1', '3:2', '1:1', '3:4', '1:2'])
        arr_ax[0].set_title('Location on Arnold tongues')

        arr_ax[1].set_xticks(
            [1/40, 1/36, 1/32, 1/28, 1/24, 1/20, 1/16, 1/12, 1/10])
        arr_ax[1].set_xticklabels(
            ['40', '36', '32', '28', '24', '20', '16', '12', '10'])
        arr_ax[1].set_xlabel('Circadian period (h)')
        arr_ax[1].set_title('FFT on deterministic simulations')

        arr_ax[2].set_xticks(
            [1/40, 1/36, 1/32, 1/28, 1/24, 1/20, 1/16, 1/12, 1/10])
        arr_ax[2].set_xticklabels(
            ['40', '36', '32', '28', '24', '20', '16', '12', '10'])
        arr_ax[2].set_yticklabels([])
        arr_ax[2].set_xlabel('Circadian period (h)')
        arr_ax[2].set_title('FFT on stochastic simulations')

        line_det.set_data([], [])
        line_stoc.set_data([], [])
        line_harm_det.set_data([], [])
        line_harm_stoc.set_data([], [])
        point_arnold.set_offsets([])
        return line_det, line_stoc, point_arnold

    def run(data):
        # update the data
        [x_p, y_p, x_s, y_s, x_d, y_d] = data
        arr_ax[1].set_ylabel(r'$T_\phi=$' + str(x_p))
        # arr_ax[1].figure.canvas.draw()
        line_harm_det.set_data([1 / x_p, 1 / x_p], [-10, 10])
        line_harm_stoc.set_data([1 / x_p, 1 / x_p], [-10, 10])
        line_det.set_data(x_d, y_d)
        line_stoc.set_data(x_s, y_s)
        point_arnold.set_offsets([x_p, y_p])
        return line_det, line_stoc, point_arnold

    ##################### COMPUTE DATA ##################
    l_data = []
    for idx_T, T_phi in enumerate(l_T_phi):

        period_phi = T_phi
        w_phi = 2 * np.pi / T_phi
        l_parameters = [dt, sigma_em_circadian, W, pi, N_theta, std_theta,
            period_theta, l_boundaries_theta, w_theta, N_phi, std_phi,
            period_phi, l_boundaries_phi, w_phi, N_amplitude_theta,
            mu_amplitude_theta, std_amplitude_theta, gamma_amplitude_theta,
            l_boundaries_amplitude_theta, N_background_theta,
            mu_background_theta, std_background_theta, gamma_background_theta,
            l_boundaries_background_theta, F]

        ##################### CREATE HIDDEN VARIABLES ##################
        theta_var_coupled, amplitude_var, background_var = \
                        create_hidden_variables(l_parameters=l_parameters)
        l_var = [theta_var_coupled, amplitude_var, background_var]

        ##################### FFT ON VERY LONG DETERMINISTIC TRACE ############
        detSim = DetSim(l_parameters, cell, temperature)
        waveform_temp = W + W[0]
        waveform_func = interp1d(np.linspace(
            0, 2 * np.pi, len(waveform_temp), endpoint=True), waveform_temp)

        tspan, vect_Y = detSim.simulate(
            tf=20000, full_simulation=False, rand=True)
        l_signal = waveform_func(vect_Y[1000:, 0] % (2 * np.pi))

        Fs = 2.0  # sampling rate
        n = len(l_signal)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq_det = frq[range(int(n / 2))]  # one side frequency range
        Y = np.fft.fft(l_signal) / n  # fft computing and normalization
        Y_det = Y[range(int(n / 2))]

        ##################### FFT ON VERY LONG STOCHASTIC TRACE ################
        sim = HMMsim(l_var, signal_model, std_em=0., waveform=W,
                     dt=0.5, uniform=True, T_phi=T_phi)
        l_t_l_xi, l_t_obs = sim.simulate(tf=20000)
        l_obs_circadian = np.array(l_t_obs)[1000:, 0]

        Fs = 2.0  # sampling rate
        n = len(l_obs_circadian)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq_stoc = frq[range(int(n / 2))]  # one side frequency range
        Y = np.fft.fft(l_obs_circadian) / n  # fft computing and normalization
        Y_stoc = Y[range(int(n / 2))]

        # frq_stoc=np.linspace(0,0.4,100)
        #Y_stoc = [T_phi*10**-2/3]*100
        # frq_det=np.linspace(0,0.4,100)
        #Y_det = [T_phi*10**-2/3]*100
        l_data.append([T_phi, 1, frq_stoc, np.abs(
            Y_stoc), frq_det, np.abs(Y_det)])
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.2, hspace=None)
    ani = animation.FuncAnimation(
        fig, run, l_data, blit=True, interval=100, repeat=True, init_func=init)
    ani.save('../Results/AllByCellCycle/fft.mp4',
             fps=30, extra_args=['-vcodec', 'libx264'])
    #ani.save('../Results/AllByCellCycle/fft.gif', writer='imagemagick', fps=30)
    # plt.show()


def anim_phase_space(cell='NIH3T3', nb_traces=500, size_block=100):
    """
    Compute the phase-space density with the corresponding attractor for
    evolving cell-cycle speeds and animate it.

    Parameters
    ----------
    cell : string
        Cell condition.
    nb_traces : integer
        Number of traces from which the inference is made.
    size_block : integer
        Size of the traces chunks (to save memory).

    """
    def mask(l_theta, l_phi):
        abs_d_data_x = np.abs(np.diff(l_theta))
        mask_x = np.hstack(
            [abs_d_data_x > abs_d_data_x.mean()+3*abs_d_data_x.std(), [False]])
        masked_l_theta = np.array(
            [x if not m else np.nan for x, m in zip(l_theta, mask_x)])

        abs_d_data_x = np.abs(np.diff(l_phi))
        mask_x = np.hstack(
            [abs_d_data_x > abs_d_data_x.mean()+3*abs_d_data_x.std(), [False]])
        masked_l_phi = np.array(
            [x if not m else np.nan for x, m in zip(l_phi, mask_x)])

        l_theta = masked_l_theta
        l_phi = masked_l_phi

        return l_theta, l_phi

    temperature = None  # get set to none after loading the parameters
    l_T_phi = np.arange(12, 15, 0.5)
    #l_T_phi = [22]
    ##################### LOAD OPTIMIZED PARAMETERS ##################
    path = '../Parameters/Real/opt_parameters_div_' + \
        str(temperature) + "_" + cell + '.p'
    with open(path, 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)
    temperature = None

    ##################### DISPLAY PARAMETERS ##################
    display_parameters_from_file(path, show=False)

    l_data = []
    for T_phi in l_T_phi:
        period_phi = T_phi
        w_phi = 2 * np.pi / T_phi
        l_parameters = [dt, sigma_em_circadian, W, pi, N_theta, std_theta,
            period_theta, l_boundaries_theta, w_theta, N_phi, std_phi,
            period_phi, l_boundaries_phi, w_phi, N_amplitude_theta,
            mu_amplitude_theta, std_amplitude_theta, gamma_amplitude_theta,
            l_boundaries_amplitude_theta, N_background_theta,
            mu_background_theta, std_background_theta, gamma_background_theta,
            l_boundaries_background_theta, F]
        print('### T-CELL-CYCLE = ' + str(T_phi) + '###')
        ##################### LOAD DATA ##################
        if cell == 'NIH3T3':
            path = "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
        else:
            path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"

        dataClass = LoadData(
            path,
            nb_traces,
            temperature=temperature,
            division=True,
            several_cell_cycles=False,
            remove_odd_traces=True)
        (ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, ll_peak,
        ll_idx_cell_cycle_start, T_theta, std_T_theta, T_phi_obs, std_T_phi) \
            = dataClass.load(period_phi=T_phi, load_annotation=True)
        ll_idx_peak = [[idx for idx, v in enumerate(l_peak) if v > 0] \
                                                        for l_peak in ll_peak]
        print(len(ll_signal), " traces kept")

        ##################### CREATE HIDDEN VARIABLES ##################
        theta_var_coupled, amplitude_var, background_var = \
                            create_hidden_variables(l_parameters=l_parameters)
        l_var = [theta_var_coupled, amplitude_var, background_var]

        ##################### CREATE AND RUN HMM ##################
        hmm = HMM_SemiCoupled(
            l_var,
            ll_signal,
            sigma_em_circadian,
            ll_obs_phi,
            waveform=W,
            ll_nan_factor=ll_nan_circadian_factor,
            pi=pi,
            crop=True)
        l_gamma_div, l_logP_div = hmm.run(project=False)

        ##################### REMOVE BAD TRACES #####################
        Plim = np.percentile(l_logP_div, 10)
        idx_to_keep = [i for i, logP in enumerate(l_logP_div) if logP > Plim]
        l_gamma_div = [l_gamma_div[i] for i in idx_to_keep]
        ll_signal = [ll_signal[i] for i in idx_to_keep]
        ll_area = [ll_area[i] for i in idx_to_keep]
        l_logP_div = [l_logP_div[i] for i in idx_to_keep]
        ll_obs_phi = [ll_obs_phi[i] for i in idx_to_keep]
        ll_idx_cell_cycle_start = [
            ll_idx_cell_cycle_start[i] for i in idx_to_keep]
        ll_idx_peak = [ll_idx_peak[i] for i in idx_to_keep]
        print("Kept traces with div: ", len(idx_to_keep))

        ##################### CROP SIGNALS FOR PLOTTING ##################
        l_first = [[it for it, obj in enumerate(
            l_obs_phi) if obj != -1][0] for l_obs_phi in ll_obs_phi]
        l_last = [[len(l_obs_phi) -
                   it -
                   1 for it, obj in enumerate(l_obs_phi[::-
                                                        1]) if obj != -
                   1][0] for l_obs_phi in ll_obs_phi]
        ll_signal = [l_signal[first:last + 1]
                     for l_signal, first, last in zip(ll_signal,l_first,l_last)]
        ll_area = [l_area[first:last + 1]
                   for l_area, first, last in zip(ll_area, l_first, l_last)]
        ll_obs_phi = [l_obs_phi[first:last + 1] for l_obs_phi,
                      first, last in zip(ll_obs_phi, l_first, l_last)]
        ll_idx_cell_cycle_start = [[v for v in l_idx_cell_cycle_start \
                                                if v >= first and v <= last]
                                   for l_idx_cell_cycle_start, first, last \
                                   in zip(ll_idx_cell_cycle_start,
                                          l_first, l_last)]
        ll_idx_peak = [[v for v in l_idx_peak if v >= first and v <= last]
                       for l_idx_peak, first, last in zip(ll_idx_peak, l_first,
                                                          l_last)]

        ##################### CREATE ll_idx_obs_phi and ll_val_phi##############
        ll_idx_obs_phi = []
        for l_obs in ll_obs_phi:
            l_idx_obs_phi = []
            for obs in l_obs:
                l_idx_obs_phi.append(int(round(obs / (2 *np.pi) *
                                              len(theta_var_coupled.codomain)))%
                                     len(theta_var_coupled.codomain))
            ll_idx_obs_phi.append(l_idx_obs_phi)

        ##################### GET DETERMINISTIC ATTRACTOR AND REPELLER #########
        detSim = DetSim(l_parameters, cell, temperature)

        # get attractor
        ll_phase_theta, ll_phase_phi = detSim.plot_trajectory(
            ti=5000, tf=5100, rand=True, save=False, K=1, T_phi=T_phi)
        # get repeller
        ll_phase_theta_rep, ll_phase_phi_rep = detSim.plot_trajectory(
            ti=5000, tf=-5100, rand=True, save=False, K=1, T_phi=T_phi)

        l_phase_theta_att = [
           theta for l_phase_theta in ll_phase_theta for theta in l_phase_theta]

        l_phase_phi_att = [
            phi for l_phase_phi in ll_phase_phi for phi in l_phase_phi]
        l_phase_theta_att, l_phase_phi_att = mask(
            l_phase_theta_att, l_phase_phi_att)

        l_phase_theta_rep = [theta for l_phase_theta in ll_phase_theta_rep \
                                                    for theta in l_phase_theta]
        l_phase_phi_rep = [
            phi for l_phase_phi in ll_phase_phi_rep for phi in l_phase_phi]
        l_phase_theta_rep, l_phase_phi_rep = mask(l_phase_theta_rep,
                                                                l_phase_phi_rep)

        ##################### GET DATA ATTRACTOR ##################
        M_at = plot_phase_space_density(
            l_var, l_gamma_div, ll_idx_obs_phi, F_superimpose=F, save=False)
        ##################### PLOT VECTORFIELD ##################
        X = np.linspace(0, 2 * np.pi, F.shape[0], endpoint=False)
        Y = np.linspace(0, 2 * np.pi, F.shape[1], endpoint=False)

        U = 2 * np.pi / T_theta + F.T
        V = np.empty((F.shape[0], F.shape[1]))
        V.fill(2 * np.pi / T_phi)

        # sample to reduce the number of arrows
        l_idx_x = [x for x in range(0, F.shape[0], 4)]
        l_idx_y = [x for x in range(0, F.shape[1], 4)]
        X = X[l_idx_x]
        Y = Y[l_idx_y]
        U = [u[l_idx_y] for u in U[l_idx_x]]
        V = [v[l_idx_y] for v in V[l_idx_x]]
        C = [c[l_idx_y] for c in F.T[l_idx_x]]

        ##################### GET STOCHASTIC ATTRACTOR ##################
        sim = HMMsim(l_var, signal_model, sigma_em_circadian,
                     waveform=W, dt=0.5, uniform=True, T_phi=T_phi)
        ll_t_l_xi, ll_t_obs = sim.simulate_n_traces(nb_traces=10, tf=1000)

        ### CROP BEGINNING OF THE TRACES ###
        ll_t_l_xi = [l_t_l_xi[-700:] for l_t_l_xi in ll_t_l_xi]
        ll_t_obs = [l_t_obs[-700:] for l_t_obs in ll_t_obs]
        ##################### REORDER VARIABLES ##################
        ll_obs_circadian = []
        ll_obs_nucleus = []
        lll_xi_circadian = []
        lll_xi_nucleus = []
        for idx, (l_t_l_xi, l_t_obs) in enumerate(zip(ll_t_l_xi, ll_t_obs)):
            ll_xi_circadian = [t_l_xi[0] for t_l_xi in l_t_l_xi]
            ll_xi_nucleus = [t_l_xi[1] for t_l_xi in l_t_l_xi]
            l_obs_circadian = np.array(l_t_obs)[:, 0]
            l_obs_nucleus = np.array(l_t_obs)[:, 1]
            ll_obs_circadian.append(l_obs_circadian)
            ll_obs_nucleus.append(l_obs_nucleus)
            lll_xi_circadian.append(ll_xi_circadian)
            lll_xi_nucleus.append(ll_xi_nucleus)

        omega_phi = 2 * np.pi / T_phi
        sim = PlotStochasticSpeedSpace(
            (lll_xi_circadian,
             lll_xi_nucleus),
            l_var,
            dt,
            omega_phi,
            cell,
            temperature,
            cmap=None)
        _, _, M_sim = sim.getPhaseSpace()
        M_sim_theta = np.array([l / np.sum(l) for l in M_sim]).T

        l_theta = []
        l_phi = []
        for idx_phi, phi in enumerate(theta_var_coupled.domain):
            #l_theta.append( np.angle(np.sum(np.multiply(M_sim_theta[idx_phi],
            #                np.exp(1j*np.array(theta_var_coupled.domain)))))\
            #                %(2*np.pi) )
            l_theta.append(
                theta_var_coupled.domain[np.argmax(M_sim_theta[idx_phi])])
            l_phi.append(phi)

        l_theta_stoc, l_phi_stoc = mask(l_theta, l_phi)
        ##################### PLOT ATTRACTOR OF THE DATA ##################

        l_theta = []
        l_phi = []
        for idx_phi, phi in enumerate(theta_var_coupled.domain):
            l_theta.append(np.angle(np.sum(np.multiply(M_at.T[idx_phi], np.exp(
                1j * np.array(theta_var_coupled.domain))))) % (2 * np.pi))
            l_phi.append(phi)

        l_theta_data, l_phi_data = mask(l_theta, l_phi)

        l_data.append([T_phi,
                       (U,
                        V,
                        C),
                       (l_phase_theta_att,
                        l_phase_phi_att),
                       (l_phase_theta_rep,
                           l_phase_phi_rep),
                       (l_theta_stoc,
                           l_phi_stoc),
                       (l_theta_data,
                           l_phi_data)])
    ##################### MAKE ANIMATION ##################
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(10, 10))

    #ax.imshow(F.T, cmap=bwr, vmin=-0.3, vmax=0.3, interpolation='spline16',
                #origin='lower', extent=[0, 1,0, 1])
    ax.pcolormesh(np.linspace(0, 1, N_theta), np.linspace(
        0, 1, N_theta), F.T, cmap=bwr, vmin=-0.3, vmax=0.3)

    quiver1 = ax.quiver(np.array(X) / (2 * np.pi), np.array(Y) /
                        (2 * np.pi), [], [], [], alpha=.5, cmap='cool')
    quiver2 = ax.quiver(np.array(X) / (2 * np.pi), np.array(Y) / (2 * np.pi),
                        [], [], edgecolor='k', facecolor='None', linewidth=.1)

    # , label = 'Attractor')
    line_att, = ax.plot([], [], lw=2, color='green', alpha=0.1)
    # , label = 'Repeller')
    line_rep, = ax.plot([], [], lw=2, color='red', alpha=0.1)
    # ., label = 'E[Stochastic simulation]')
    l_stoc_att, = ax.plot([], [], lw=2, color='grey', alpha=1)
    l_data_att, = ax.plot([], [], lw=2, color='black',
                          alpha=1)  # ., label = 'E[Data]')

    def init():
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        # ax.grid(True)

        ax.set_xlabel(r'Circadian phase $\theta$')
        ax.set_ylabel(r'Cell-cycle phase $\phi$')

        line_att.set_data([], [])
        line_rep.set_data([], [])
        l_stoc_att.set_data([], [])
        l_data_att.set_data([], [])
        quiver1.set_UVC([], [], [])
        quiver2.set_UVC([], [], [])
        return line_att, line_rep, l_stoc_att, l_data_att, quiver1, quiver2

    def run(data):

        # update the data
        [T_phi, (U, V, C), (l_phase_theta_att, l_phase_phi_att),
        (l_phase_theta_rep,l_phase_phi_rep), (l_theta_stoc, l_phi_stoc),
        (l_theta_data, l_phi_data)] = data
        ax.set_title(r'$T_\phi=$' + str(T_phi))
        # arr_ax[1].figure.canvas.draw()

        line_att.set_data(l_phase_theta_att, l_phase_phi_att)
        line_rep.set_data(l_phase_theta_rep, l_phase_phi_rep)
        l_stoc_att.set_data(l_theta_stoc, l_phi_stoc)
        l_data_att.set_data(l_theta_data, l_phi_data)
        quiver1.set_UVC(U, V, C)
        quiver2.set_UVC(U, V)
        return line_att, line_rep, l_stoc_att, l_data_att, quiver1, quiver2

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    ani = animation.FuncAnimation(
        fig, run, l_data, blit=False, interval=1, repeat=True, init_func=init)
    ani.save('../Results/AllByCellCycle/phase_space.mp4',
             fps=30, extra_args=['-vcodec', 'libx264'])

    # plt.show()


def refine_heatmap(path='../Results/AllByCellCycle/save_fft_save.p'):
    """
    Refine and plot the Fourier transform heatmap previously computed and stored
    in a Pickle.

    Parameters
    ----------
    path : string
        Path of the Fourier transform heatmap.

    """
    dic_save = pickle.load(open(path, "rb"))
    frq_det = dic_save['frq_det']
    grid_det = dic_save['grid_det']
    l_T_phi = dic_save['l_T_phi']
    frq_stoc = dic_save['frq_stoc']
    grid_sto = dic_save['grid_sto']

    # create heatmaps
    plt.figure(figsize=(6, 5))

    T_phi = np.array(l_T_phi)
    T_det = 1 / frq_det
    T_det[0] = T_det[1]

    plt.pcolormesh(T_phi, T_det, grid_det.T, cmap='coolwarm',
                   vmin=0, vmax=10**-1)  # , shading='gouraud')
    plt.axvline(18.7, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.axvline(27.5, color='white', ls='--', alpha=0.5, lw=0.5)
    #plt.axvspan(18.7, 27.5, alpha=0.1, color='white')
    plt.text(21.5, 37, '1:1', color='white', size=12)

    #plt.axvspan(11.6, 12, alpha=0.1, color='white')
    plt.axvline(11.6, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.axvline(12, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.text(11, 37, '2:1', color='white', size=12)

    #plt.axvspan(36.7, 38., alpha=0.1, color='white')
    plt.axvline(36.7, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.axvline(38, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.text(36, 37, '2:3', color='white', size=12)

    #plt.axvspan(46.5, 52.5, alpha=0.1, color='white')
    plt.axvline(46.5, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.axvline(52.5, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.text(48, 37, '1:2', color='white', size=12)

    #plt.axvspan(7.9, 8.2, alpha=0.1, color='white')
    plt.axvline(7.9, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.axvline(8.2, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.text(6.5, 37, '3:1', color='white', size=12)

    plt.colorbar()
    plt.xlim([6, 60])
    plt.ylim([6, 40])
    plt.ylabel(r"$T_{\theta}$")
    plt.xlabel(r'$T_{\phi}$')
    locs, labels = plt.xticks()
    plt.xticks(np.arange(10, 62, 5), [str(x) for x in np.arange(10, 62, 5)])
    plt.yticks(np.arange(10, 42, 5), [str(x) for x in np.arange(10, 42, 5)])
    plt.title("FFT on deterministic system")
    plt.savefig('../Results/AllByCellCycle/fft_detheat.png', dpi=600)
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 5))

    T_phi = np.array(l_T_phi)
    T_sto = 1 / frq_stoc
    T_sto[0] = T_sto[1]

    plt.pcolormesh(T_phi, T_sto, grid_sto.T, cmap='coolwarm',
                   vmin=0, vmax=10**-1)  # , shading='gouraud')
    plt.axvline(18.7, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.axvline(27.5, color='white', ls='--', alpha=0.5, lw=0.5)
    #plt.axvspan(18.7, 27.5, alpha=0.1, color='white')
    plt.text(21.5, 37, '1:1', color='white', size=12)

    #plt.axvspan(11.6, 12, alpha=0.1, color='white')
    plt.axvline(11.6, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.axvline(12, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.text(11, 37, '2:1', color='white', size=12)

    #plt.axvspan(36.7, 38., alpha=0.1, color='white')
    plt.axvline(36.7, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.axvline(38, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.text(36, 37, '2:3', color='white', size=12)

    #plt.axvspan(46.5, 52.5, alpha=0.1, color='white')
    plt.axvline(46.5, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.axvline(52.5, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.text(48, 37, '1:2', color='white', size=12)

    #plt.axvspan(7.9, 8.2, alpha=0.1, color='white')
    plt.axvline(7.9, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.axvline(8.2, color='white', ls='--', alpha=0.5, lw=0.5)
    plt.text(6.5, 37, '3:1', color='white', size=12)

    plt.colorbar()
    plt.xlim([6, 60])
    plt.ylim([6, 40])
    plt.ylabel(r"$T_{\theta}$")
    plt.xlabel(r'$T_{\phi}$')
    locs, labels = plt.xticks()
    plt.xticks(np.arange(10, 62, 5), [str(x) for x in np.arange(10, 62, 5)])
    plt.yticks(np.arange(10, 42, 5), [str(x) for x in np.arange(10, 42, 5)])
    plt.title("FFT on stochastic system")
    plt.savefig('../Results/AllByCellCycle/fft_stoheat.png', dpi=600)
    plt.show()
    plt.close()


""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    all_by_cell_cycle_period(cell = 'NIH3T3', nb_traces = 200)
    #fourier_by_cell_cycle(cell = 'NIH3T3')
    #anim_fourier(cell = 'NIH3T3')
    #anim_phase_space(cell = 'NIH3T3', nb_traces = 5000000)
    #refine_heatmap()
