# -*- coding: utf-8 -*-
### Module import
import numpy as np
from StateVar import StateVar
import matplotlib.pyplot as plt
from scipy import io
from scipy.interpolate import interp1d
import pickle
import copy
from numba import jit
from scipy.optimize import minimize
import scipy.stats as st
import matplotlib.pyplot as plt

#fix dt
dt = 0.5

""" This file is not a class, but all computations linked to the EM algorithm
are done in here. It is not a class since the computations are optimized using
Numba, and the class handling is not perfect with this modules.
All the functions using a decorator '&' are optimized with Numba, and thus the
code they contain is not Pythonic. """

@jit
def compute_sigma_transition_theta(t, jPt, idx_trace, theta_domain,
                                   F_previous_iter, w_theta,
                                   ll_idx_obs_phi_trace = None):
    """
    Update sigma_theta.

    Parameters
    ----------
    t : float
        Current timepoint.
    jPt : ndarray
        Joint probability for the current timepoint.
    idx_trace : int
        Trace index.
    theta_domain : array
        Circadian phase domain.
    F_previous_iter : ndarray
        F estimated at previous iteration.
    w_theta : float
        Circadian speed.
    ll_idx_obs_phi_trace : list of list
        Cell-cycle indexes for several traces.

    Returns
    -------
    The current estimate of sigma_theta.
    """
    var = 0
    for idx_theta_i, theta_i in enumerate(theta_domain):
        if ll_idx_obs_phi_trace is not None:
            theta_dest = theta_i + (w_theta+F_previous_iter[idx_theta_i,
                                    ll_idx_obs_phi_trace[t]])*dt
        else:
            theta_dest =theta_i + w_theta*dt

        z_mean = np.exp(1j*(theta_dest))
        z_experimental = 0
        norm = np.sum(jPt[idx_theta_i])
        if norm==0:
            continue
        jPt_i_norm = jPt[idx_theta_i]/norm
        for idx_theta_k, theta_k in enumerate(theta_domain):
            z_experimental += np.exp(1j*theta_k)* jPt_i_norm[idx_theta_k]
            if np.isnan(z_experimental):
                pass
            var+= (min( abs( theta_dest - theta_k ),
                        abs( theta_dest - (theta_k+2*np.pi) ) ,
                        abs( theta_dest - (theta_k-2*np.pi) ) )**2) \
                        *jPt[idx_theta_i, idx_theta_k]


        z_experimental_angle = np.angle(z_experimental)
        z_experimental_abs = np.abs(z_experimental)
        #var+= 2 * np.log(z_mean / np.exp(1j*z_experimental_angle)\
                        #* z_experimental_abs) * norm #what it should be
        #var+= 2 * np.log(z_mean / np.exp(1j*z_experimental_angle)) \
                        #* z_experimental_abs * norm #what works
        #var+= 2 * np.log(z_mean/z_experimental)*norm #what was done before

    return var

@jit
def compute_sigma_transition_OU(jPt, domain, mu, gamma):
    """
    Update sigma for OU processes.

    Parameters
    ----------
    jPt : ndarray
        Joint probability for the current timepoint.
    domain : array
        OU state domain.
    mu : float
        Current estimate for the mean of the process.
    gamma : float
        Current estimate for the regression constant of the process.

    Returns
    -------
    The current estimate of sigma for the OU process.
    """
    var = 0
    for idx_A_i, A_i in enumerate(domain):
        for idx_A_k, A_k in enumerate(domain):
            var+= (A_k-(mu+(A_i-mu)*np.exp(-gamma*dt)))**2*jPt[idx_A_i, idx_A_k]
    fact = (2*gamma)/(1-np.exp(-2*gamma*dt))
    return var * fact

@jit
def compute_mu_OU(jPt, domain, gamma):
    """
    Update sigma for OU processes.

    Parameters
    ----------
    jPt : ndarray
        Joint probability for the current timepoint.
    domain : array
        OU state domain.
    gamma : float
        Current estimate for the regression constant of the process.

    Returns
    -------
    The current estimate of the mean for the OU process.
    """
    mu = 0
    #l_mu = []
    for idx_A_i, A_i in enumerate(domain):
        for idx_A_k, A_k in enumerate(domain):
            mu+=  (A_i*np.exp(-gamma*dt)-A_k) * jPt[idx_A_i, idx_A_k]
    mu = mu/(np.exp(-gamma*dt)-1)
    #print("mu", mu)
    return mu

@jit
def compute_sigma_emission(mat_diff_obs, l_signal, domain_theta, domain_A,
                           domain_B, W):
    """
    Update the observed noise parameter.

    Parameters
    ----------
    mat_diff_obs : ndarray
        Difference between model prediction and observation.
    l_signal : array
        Vector of observations.
    domain_theta : array
        Circadian phase domain.
    domain_A : array
        Amplitude process domain.
    domain_B : array
        Background process domain.
    W : list
        Waveform.

    Returns
    -------
    The current estimate of the observation noise.
    """
    if W is None:
        for idx_theta in range(len(domain_theta)):
            for idx_A in range(len(domain_A)):
                for idx_B in range(len(domain_B)):
                        loc = ((1+np.cos(domain_theta[idx_theta]))/2)**1.6 \
                                * np.exp(domain_A[idx_A]) + domain_B[idx_B]
                        for t in range(len(l_signal)):
                            mat_diff_obs[t, idx_theta, idx_A, idx_B ]=\
                                                        (l_signal[t] - loc)**2
    else:
        for idx_theta in range(len(domain_theta)):
            for idx_A in range(len(domain_A)):
                for idx_B in range(len(domain_B)):
                        loc = W[idx_theta] * np.exp(domain_A[idx_A]) \
                                + domain_B[idx_B]
                        for t in range(len(l_signal)):
                            mat_diff_obs[t, idx_theta, idx_A, idx_B ]=\
                                                        (l_signal[t] -  loc)**2
    return mat_diff_obs

@jit
def compute_waveform(W_num, W_denum, l_signal, domain_theta, domain_A,
                    domain_B, gamma):
    """
    Update the waveform.

    Parameters
    ----------
    W_num : list
        Temporary variable (waveform numerator in the update formula)
    W_denum : list
        Temporary variable (waveform denominator in the update formula)
    l_signal : list
        List of observations.
    domain_theta : array
        Circadian phase domain.
    domain_A : array
        Amplitude process domain.
    domain_B : array
        Background process domain.
    gamma : float
        Regression constant.

    Returns
    -------
    The current estimate of the waveform.
    """
    for idx_theta, theta in enumerate(domain_theta):
        num = 0
        denum = 0
        for t, s in enumerate(l_signal):
            for idx_B, B in enumerate(domain_B):
                for idx_A, A in enumerate(domain_A):
                    num+=(s-B)*gamma[t, idx_theta, idx_A, idx_B]
                    denum+=np.exp(A) * gamma[t, idx_theta, idx_A, idx_B]
        W_num[idx_theta] += num
        W_denum[idx_theta] += denum
    return W_num, W_denum


@jit(nopython=True)
def compute_all_jP( l_alpha_TR_ph_A, l_alpha_TR_ph_B,  l_alpha_TR_A_B,
                    l_mat_TR_theta, l_beta_E, background_TR, amplitude_TR,
                    N_theta, N_amplitude_theta, N_background_theta):
    """
    Compute joint probability for the phase, the amplitude and the background.

    Parameters
    ----------
    l_alpha_TR_ph_A : list of array
        alpha vectors marginalized on the phase and the amplitude.
    l_alpha_TR_ph_B : list of array
        alpha vectors marginalized on the phase and the background.
    l_alpha_TR_A_B : list of array
        alpha vectors marginalized on the amplitude and the background.
    l_mat_TR_theta : list of array
        List of transitions matrix for theta (depending on phi)
    l_beta_E : list of array
        Beta vector multiplied by the observation probability.
    background_TR : array
        Transition matrix for the background.
    amplitude_TR : array
        Transition matrix for the amplitude.
    N_theta : int
        Number of circadian states
    N_amplitude_theta : int
        Number of amplitude states
    N_background_theta : int
        Number of background states

    Returns
    -------
    The current joint probabilities for the different coestimated variables.
    """
    jP_phase = np.zeros((len(l_alpha_TR_A_B), N_theta, N_theta) )
    jP_amplitude = np.zeros((len(l_alpha_TR_A_B), N_amplitude_theta,
                            N_amplitude_theta))
    jP_background = np.zeros((len(l_alpha_TR_A_B), N_background_theta,
                            N_background_theta))


    for t in range(len(l_alpha_TR_A_B)):
        for idx_A in range(N_amplitude_theta):

            for idx_B in range(N_background_theta):

                for idx_theta_i in range(N_theta):
                    for idx_theta_j in range(N_theta):
                        jP_phase[t, idx_theta_i, idx_theta_j] += \
                            l_alpha_TR_A_B[t, idx_theta_i, idx_A, idx_B] \
                            * l_mat_TR_theta[t, idx_theta_i, idx_theta_j] \
                            * l_beta_E[t, idx_theta_j, idx_A, idx_B]


            for idx_theta in range(N_theta):

                for idx_B_i in range(N_background_theta):
                    for idx_B_j in range(N_background_theta):
                        jP_background[t, idx_B_i, idx_B_j] += \
                        l_alpha_TR_ph_A[t, idx_theta, idx_A, idx_B_i] \
                        * background_TR[idx_B_i, idx_B_j] \
                        * l_beta_E[t, idx_theta, idx_A, idx_B_j]


        for idx_B in range(N_background_theta):
            for idx_theta in range(N_theta):
                for idx_A_i in range(N_amplitude_theta):
                    for idx_A_j in range(N_amplitude_theta):
                        jP_amplitude[t, idx_A_i, idx_A_j] += \
                        l_alpha_TR_ph_B[t, idx_theta, idx_A_i, idx_B] \
                        * amplitude_TR[idx_A_i, idx_A_j] \
                        * l_beta_E[t, idx_theta, idx_A_j, idx_B]

    return jP_phase, jP_amplitude, jP_background


@jit(nopython=True)
def compute_jP_phase(l_alpha_TR_A_B, l_mat_TR_theta, l_beta_E, N_theta,
                     N_amplitude_theta, N_background_theta):
    """
    Compute joint probability for the phase only.

    Parameters
    ----------
    l_alpha_TR_A_B : list of array
        alpha vectors marginalized on the amplitude and the background.
    l_mat_TR_theta : list of array
        List of transitions matrix for theta (depending on phi)
    l_beta_E : list of array
        Beta vector multiplied by the observation probability.
    N_theta : int
        Number of circadian states
    N_amplitude_theta : int
        Number of amplitude states
    N_background_theta : int
        Number of background states

    Returns
    -------
    The current joint probabilities for the phase.
    """

    jP_phase = np.zeros((len(l_alpha_TR_A_B), N_theta, N_theta) )
    for t in range(len(l_alpha_TR_A_B)):
        for idx_A in range(N_amplitude_theta):
            for idx_B in range(N_background_theta):
                for idx_theta_i in range(N_theta):
                    for idx_theta_j in range(N_theta):
                        jP_phase[t, idx_theta_i, idx_theta_j] += \
                            l_alpha_TR_A_B[t, idx_theta_i, idx_A, idx_B] \
                            * l_mat_TR_theta[t, idx_theta_i, idx_theta_j] \
                            * l_beta_E[t, idx_theta_j, idx_A, idx_B]

    return jP_phase



def compute_jP_by_block(ll_alpha, l_E, ll_beta, ll_mat_TR, amplitude_TR,
                        background_TR, N_theta, N_amplitude_theta,
                        N_background_theta, only_F_and_pi = False):
    """
    Compute joint probability by block of traces to go faster.

    Parameters
    ----------
    ll_alpha : list of list array
        alpha vectors.
    l_E : list
        list of emissions.
    ll_beta : list of list array
        beta vectors.
    ll_mat_TR : list of list of array
        Transitions matrices.
    background_TR : array
        Transition matrix for the background.
    amplitude_TR : array
        Transition matrix for the amplitude.
    N_theta : int
        Number of circadian states
    N_amplitude_theta : int
        Number of amplitude states
    N_background_theta : int
        Number of background states

    Returns
    -------
    List of joint probabilities for the phase, amplitude and background,
    computed by blocks.
    """
    l_jP_phase = []
    l_jP_amplitude = []
    l_jP_background = []

    print("Compute Joint probability")
    #then compute joint probability

    l_alpha_TR_A_B = None
    l_alpha_TR_ph_A = None
    l_alpha_TR_ph_B = None

    for idx_trace in range(len(ll_alpha)):

        l_alpha_TR_A_B = np.tensordot(np.tensordot(ll_alpha[idx_trace][:-1],
                                                   amplitude_TR, axes = (2,0)),
                                     background_TR, axes = (2,0))
        #print(l_alpha_TR_A_B.shape)
        l_beta_E = l_E[idx_trace][1:]*ll_beta[idx_trace][1:]
        if not only_F_and_pi:

            l_alpha_TR_ph = np.array([np.tensordot(ll_alpha[idx_trace][:-1][t],
                                                   ll_mat_TR[idx_trace][:-1][t],
                                                   axes = (0,0)) \
                                for t in range(len(ll_alpha[idx_trace][:-1])) ])
            #print(l_alpha_TR_ph.shape)
            l_alpha_TR_ph_A = np.tensordot(l_alpha_TR_ph, amplitude_TR,
                                           axes = (1,0))
            #print(l_alpha_TR_ph_A.shape)
            l_alpha_TR_ph_A = np.swapaxes(l_alpha_TR_ph_A , 2, 1)
            l_alpha_TR_ph_A = np.swapaxes(l_alpha_TR_ph_A , 3, 2)
            #print(l_alpha_TR_ph_A.shape)

            l_alpha_TR_ph_B = np.tensordot(l_alpha_TR_ph, background_TR,
                                           axes = (2,0))
            #print(l_alpha_TR_ph_B.shape)
            l_alpha_TR_ph_B = np.swapaxes(l_alpha_TR_ph_B , 2, 1)
            #print(l_alpha_TR_ph_B.shape)

            jP_phase, jP_amplitude, jP_background = \
            compute_all_jP(l_alpha_TR_ph_A, l_alpha_TR_ph_B,
                           l_alpha_TR_A_B, ll_mat_TR[idx_trace][:-1], l_beta_E,
                           background_TR, amplitude_TR, N_theta,
                           N_amplitude_theta, N_background_theta)
            jP_amplitude = \
                    np.array([jP_t/np.sum(jP_t) for jP_t in jP_amplitude])
            jP_background = \
                    np.array([jP_t/np.sum(jP_t) for jP_t in jP_background])
        else:
            jP_phase = compute_jP_phase(l_alpha_TR_A_B,
                                        ll_mat_TR[idx_trace][:-1],
                                        l_beta_E, N_theta, N_amplitude_theta,
                                        N_background_theta)
            jP_amplitude = None
            jP_background = None

        jP_phase = np.array([jP_t/np.sum(jP_t) for jP_t in jP_phase])

        l_jP_phase.append(jP_phase)
        l_jP_amplitude.append(jP_amplitude)
        l_jP_background.append(jP_background)

    return l_jP_phase, l_jP_amplitude, l_jP_background




def run_EM(l_gamma_0, l_gamma, t_l_jP, theta_var_coupled, ll_idx_obs_phi,
           F_previous_iter, ll_signal, amplitude_var, background_var,
           W_previous_iter = None, ll_idx_coef = None, only_F_and_pi = False,
           lambd_parameter = 10e-6, lambd_2_parameter = 0):
    """
    Main EM function to update all the parameters.

    Parameters
    ----------
    l_gamma_0 : list of array
        list of inferred intial condition for each process.
    l_gamma : list of array
        list of probabilities of being in a given state.
    t_l_jP : tuple of list of arrays
        tuple of list of joint probabilities for each process.
    theta_var_coupled : StateVarSemiCoupled
        circadian phase object.
    ll_idx_obs_phi : list of list
        cell cycle indexes for all traces.
    F_previous_iter : ndarray
        F estimated at previous iteration.
    ll_signal : list of list
        list of observations for each trace.
    amplitude_var : StateVar
        amplitude object.
     background_var : StateVar
        background object.
     W_previous_iter : list
        waveform estimated at the previous iteration
     ll_idx_coef = list of lists
        deprecated...
     only_F_and_pi : bool
        To update only the coupling function and the initial condition.
     lambd_parameter : float
        regularization parameter (to smooth)
     lambd_2_parameter : float
        regularization parameter (to prevent divergence of the coupling)
    Returns
    -------
    List of updated parameters.
    """

    #crop traces if nucleus signal is present
    if ll_idx_obs_phi is not None:
        ll_signal = [[s for s, idx in zip(l_s, l_idx) if idx>-1] \
                    for l_s, l_idx in zip(ll_signal ,ll_idx_obs_phi)]
        ll_idx_obs_phi = [ [idx for idx in l_idx if idx>-1] \
                            for l_idx in  ll_idx_obs_phi ]

    F, pi, sigma_theta, sigma_e, ll_idx_coef, sigma_A, sigma_B, mu_A, mu_B, W =\
            (None, None, None, None, ll_idx_coef, None, None, None, None, None)
    N_theta = theta_var_coupled.nb_substates
    N_phi = len(theta_var_coupled.codomain)
    N_amplitude_theta = amplitude_var.nb_substates
    N_background_theta = background_var.nb_substates
    w_theta = theta_var_coupled.l_parameters_f_trans[0]
    l_jP_phase = t_l_jP[0]
    l_jP_amplitude = t_l_jP[1]
    l_jP_background = t_l_jP[2]

    """ IF EM WITH FULL F """

    F = np.zeros((N_theta, N_phi))
    if ll_idx_obs_phi is not None:
        T_tot = np.sum([len(l_idx_obs_phi) for l_idx_obs_phi in ll_idx_obs_phi])
        F_num=np.zeros((N_theta, N_phi), dtype ='complex128')
        F_denum=np.zeros((N_theta, N_phi))
        np.set_printoptions(threshold=np.nan)

        #regularization parameters
        lambd = lambd_parameter*T_tot

        #lambd2 = 0
        #lambd2 = lambd_2_parameter / (8* 22)*T_tot
        lambd2 = T_tot/22 * 0.5 *lambd_2_parameter / ( 2*0.15**2 )

        M = np.zeros((N_theta*N_phi, N_theta*N_phi))

        print("Compute expected phases for the coupling function")
        for idx_trace, jP in enumerate(l_jP_phase):
            for t, jPt in enumerate(jP):
                #print(np.sum(jPt))
                #print(np.sum(jPt, axis = 1))
                F_num[:,ll_idx_obs_phi[idx_trace][t]] +=\
                 np.einsum(jPt, [0,1],
                 np.exp(1j*theta_var_coupled.domain),[1],[0])
                F_denum[:,ll_idx_obs_phi[idx_trace][t]] += np.sum(jPt,axis=1)

        #print(F_denum)

        print("Compute coupling function itself")
        for idx_theta, theta in enumerate(theta_var_coupled.domain):
            for idx_phi, phi in enumerate(theta_var_coupled.codomain):
                esp_theta_k_prob = np.angle(F_num[idx_theta, idx_phi]\
                                    /F_denum[idx_theta, idx_phi])%(2*np.pi)
                if ll_idx_coef is None:
                    esp_theta_k_model = (theta+w_theta*dt)%(2*np.pi)
                else:
                    esp_theta_k_model = (theta+(ll_idx_coef[idx_theta, idx_phi]\
                                                        +w_theta)*dt)%(2*np.pi)
                if esp_theta_k_prob - esp_theta_k_model > np.pi:
                    esp_theta_k_model += 2*np.pi
                if esp_theta_k_model - esp_theta_k_prob > np.pi:
                    esp_theta_k_prob += 2*np.pi
                #if idx_theta==0 and idx_phi==0:
                #print(esp_theta_k_prob)
                    #print(esp_theta_k_model)
                F[idx_theta, idx_phi] =  esp_theta_k_prob - esp_theta_k_model
                F[idx_theta, idx_phi] = F[idx_theta, idx_phi]/dt
                #remove nan values
                if np.isnan(F[idx_theta, idx_phi]):
                    F[idx_theta, idx_phi] = 0
                #regularization

                #fill matrix of coefficients
                F[idx_theta, idx_phi] = F[idx_theta, idx_phi] \
                                            * F_denum[idx_theta, idx_phi] * dt
                M[np.ravel_multi_index((idx_theta, idx_phi), (N_theta, N_phi)),
                  np.ravel_multi_index((idx_theta, idx_phi), (N_theta, N_phi))]\
                   =  F_denum[idx_theta, idx_phi]*dt + 2*0.15**2*lambd2+10**-15


                count = 0
                for idx_theta_2 in range(-1,2):
                    for idx_phi_2 in range(-1,2):
                        dist = 1#np.sqrt(idx_theta_2**2+idx_phi_2**2)

                        idx_tmp_1 = np.ravel_multi_index((idx_theta, idx_phi),
                                                            (N_theta, N_phi))
                        idx_tmp_2 = np.ravel_multi_index((
                                                (idx_theta+idx_theta_2)%N_theta,
                                                (idx_phi+idx_phi_2)%N_phi ),
                                                (N_theta, N_phi))
                        if M[ idx_tmp_1, idx_tmp_2  ]==0:
                            #print(dist)
                            M[ idx_tmp_1, idx_tmp_2  ] \
                                -= lambd/dist
                            count +=1//dist
                M[idx_tmp_1,
                 np.ravel_multi_index((idx_theta, idx_phi), (N_theta, N_phi)) ]\
                    += count*lambd

                '''
                M[ idx_tmp_1,
                np.ravel_multi_index(((idx_theta+1)%N_theta, (idx_phi)%N_phi ),
                (N_theta, N_phi))  ] -= lambd

                M[ idx_tmp_1,
                np.ravel_multi_index(((idx_theta-1)%N_theta, (idx_phi)%N_phi ),
                (N_theta, N_phi))  ] -= lambd

                M[ idx_tmp_1,
                np.ravel_multi_index(((idx_theta)%N_theta, (idx_phi+1)%N_phi ),
                (N_theta, N_phi))  ] -= lambd

                M[ idx_tmp_1,
                np.ravel_multi_index(((idx_theta)%N_theta, (idx_phi-1)%N_phi ),
                (N_theta, N_phi))  ] -= lambd

                M[ idx_tmp_1,
                np.ravel_multi_index((idx_theta, idx_phi),
                (N_theta, N_phi))  ] +=4*lambd
                '''

        F_flat = np.linalg.solve(M, F.flatten())
        F = np.reshape(F_flat, (N_theta, N_phi))


    #compute initial distribution
    #print(l_gamma_init[0])
    l_gamma_init = np.array(l_gamma_0)
    pi = np.sum(l_gamma_init, axis = 0)
    pi = pi/np.sum(pi)
    l_gamma_init = []

    #print(pi)

    if not only_F_and_pi:
        #compute sigma emission
        norm = np.sum([np.sum(gamma) for gamma in l_gamma])
        sigma_e = 0
        if len(l_gamma)>0:
            for gamma, l_signal in zip(l_gamma, ll_signal):
                mat_diff_obs = np.empty((len(l_signal), N_theta,
                                        N_amplitude_theta, N_background_theta))
                mat_diff_obs = compute_sigma_emission(mat_diff_obs, l_signal,
                                                      theta_var_coupled.domain,
                                                      amplitude_var.domain,
                                                      background_var.domain,
                                                      W_previous_iter)
                sigma_e += np.sum(np.multiply(mat_diff_obs, gamma))
            sigma_e = np.sqrt(sigma_e/norm)
            theta_tmp = theta_var_coupled.l_parameters_f_trans[1]
            if sigma_e<0.1 * theta_tmp or np.isnan(sigma_e):
                sigma_e = 0.1 *theta_tmp
        print("sigma_e", sigma_e)

        #compute waveform:
        W_num  = np.zeros((N_theta))
        W_denum  = np.zeros((N_theta))
        if len(l_gamma)>0:
            for gamma, l_signal in zip(l_gamma, ll_signal):
                W_num, W_denum = compute_waveform(W_num, W_denum, l_signal,
                                                  theta_var_coupled.domain,
                                                  amplitude_var.domain,
                                                  background_var.domain, gamma)
        W = W_num/W_denum



        #compute sigma transition theta
        var_theta = 0
        w_theta = theta_var_coupled.l_parameters_f_trans[0]
        l_var_theta = []
        if ll_idx_obs_phi is not None:
            for idx_trace, jP in enumerate(l_jP_phase):
                for t, jPt in enumerate(jP):
                    var = compute_sigma_transition_theta(t, jPt, idx_trace,
                                                      theta_var_coupled.domain,
                                                      F_previous_iter, w_theta,
                                                      ll_idx_obs_phi[idx_trace])
                    l_var_theta .append(var)
        else:
            for idx_trace, jP in enumerate(l_jP_phase):
                for t, jPt in enumerate(jP):
                    var = compute_sigma_transition_theta(t, jPt, idx_trace,
                                                    theta_var_coupled.domain,
                                                    F_previous_iter, w_theta,
                                                    None)
                    l_var_theta .append(var)
        #print(l_var_theta)
        var_theta = np.sum(l_var_theta)/len(l_var_theta)
        #print(norm)
        #print(np.sum([len(gamma) for gamma in l_gamma]))
        sigma_theta = np.real(np.sqrt(var_theta)/(dt**0.5))
        print("sigma_theta", sigma_theta)



        #compute sigma transition amplitude
        var_amplitude = 0
        l_var_amplitude = []
        for idx_trace, jP in enumerate(l_jP_amplitude):
            #plt.imshow(np.log(np.sum(jP, axis=0)))
            #plt.show()
            for t, jPt in enumerate(jP):

                var = compute_sigma_transition_OU(jPt,amplitude_var.domain,
                                        amplitude_var.l_parameters_f_trans[0],
                                        amplitude_var.l_parameters_f_trans[1])
                l_var_amplitude.append(var)
        var_amplitude = np.sum(l_var_amplitude)/len(l_var_amplitude)
        sigma_A = np.sqrt(var_amplitude)
        print("sigma_A", sigma_A)

        #compute sigma transition background
        var_background = 0
        l_var_background = []
        for idx_trace, jP in enumerate(l_jP_background):
            for t, jPt in enumerate(jP):
                var = compute_sigma_transition_OU( jPt, background_var.domain,
                                        background_var.l_parameters_f_trans[0],
                                        background_var.l_parameters_f_trans[1])
                l_var_background.append(var)
        var_background = np.sum(l_var_background)/len(l_var_background)
        sigma_B = np.sqrt(var_background)
        print("sigma_B", sigma_B)

        #compute mu amplitude
        l_mu_amplitude = []
        for idx_trace, jP in enumerate(l_jP_amplitude):
            for t, jPt in enumerate(jP):

                mu = compute_mu_OU(jPt, amplitude_var.domain,
                                    amplitude_var.l_parameters_f_trans[1])
                l_mu_amplitude.append(mu )
        mu_A = np.sum(l_mu_amplitude)/len(l_mu_amplitude)
        print("mu_A", mu_A)

        #compute mu background
        l_mu_background = []
        for idx_trace, jP in enumerate(l_jP_background):
            for t, jPt in enumerate(jP):

                mu = compute_mu_OU(jPt, background_var.domain,
                                  background_var.l_parameters_f_trans[1])
                l_mu_background.append(mu)
        mu_B = np.sum(l_mu_background)/len(l_mu_background)

        print("mu_B", mu_B)

    return F, pi, sigma_theta, sigma_e, ll_idx_coef, sigma_A, sigma_B,\
            mu_A, mu_B, W
