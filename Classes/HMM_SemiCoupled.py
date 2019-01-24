# -*- coding: utf-8 -*-
### Module import
import numpy as np
from StateVar import StateVar
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numba import jit
from numpy.linalg import inv

""" This file is a class, but some computation are done with Numba in order to
go faster. All the functions which are outside of the main class and are
decorated with '&' are meant to be used with Numba. """


def compute_fast_normpdf(x, loc, scale):
    """
    Fast way to compute probability from a Gaussian.

    Parameters
    ----------
    x : float
        Point on which the probability must be estimated.
    loc : float
        Gaussian mean.
    scale : flat
        Gaussian std.

    Returns
    -------
    Probability of x.
    """
    return np.exp(-0.5 * ((x - loc)/scale)**2) / ((2*np.pi)**0.5 * scale)


def product(*args):
    """
    Return cartesian product of the input arguments
    """
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield prod


@jit
def fill_E( E, l_size, l_stateVar, l_obs, domain_t, domain_A, domain_B, std_em,
            waveform):
    """
    Fill the emission matrix, for all possible states combinations.

    Parameters
    ----------
    E : array
        Emission matrix (initially empty)
    l_size : list
        Sizes of the domain of the variables.
    l_stateVar : list of StateVar
        List of variable objects.
    l_obs : list
        List of observations.
    domain_t : array
        Time domain.
    domain_A : array
        Amplitude domain.
    domain_B : array
        Background domain.
    std_em : float
        Observation noise.
    waveform : list
        Waveform.

    Returns
    -------
    Filled emission matrix.
    """
    exponent = 1.6
    if waveform is None:
        for idx_theta in range(l_size[1]):
            fact =  (2*np.pi)**0.5 * std_em
            for idx_A in range(l_size[2]):
                for idx_B in range(l_size[3]):
                    loc = ((1+np.cos(domain_t[ idx_theta  ]))/2)**exponent \
                            * np.exp(domain_A[idx_A]) + domain_B[idx_B]
                    for t in range(l_size[0]):
                        E[ t, idx_theta, idx_A, idx_B   ] = np.exp(-0.5 \
                        * ((l_obs[t] -  loc)/ std_em )**2) / fact
    else:
        for idx_theta in range(l_size[1]):
            for idx_A in range(l_size[2]):
                for idx_B in range(l_size[3]):
                    loc = waveform[  idx_theta] * np.exp(domain_A[idx_A]) \
                            + domain_B[idx_B]
                    for t in range(l_size[0]):
                        E[ t, idx_theta, idx_A, idx_B ] = np.exp(-0.5 \
                                        * ((l_obs[t] -  loc)/ std_em   )**2) \
                                        / ( (2*np.pi)**0.5 * std_em)
    return E


###
class HMM_SemiCoupled:
    """
    This class is used to run the HMM on a given set of traces,
    with a chosen set of variables.
    """
    def __init__(self, l_stateVar, ll_obs, std_em, ll_val_phi = [],
                 waveform = None, ll_nan_factor = [], pi =None, crop = False ):
        """
        Constructor of HMM_SemiCoupled.

        Parameters
        ----------
        l_stateVar : list
            List of hidden variables.
        ll_obs : list
            List of list of observations.
        std_em : float
            Observation noise.
        ll_val_phi : list of list
            Cell-cycle phase for each time and each trace.
        waveform : list
            Waveform
        ll_nan_factor : list of list
            Observations to ignore.
        pi : ndarray
            Initial condition.
        crop : bool
            Crop traces before the first peak and after the last one.
        """
        self.l_stateVar = l_stateVar
        self.ll_obs = ll_obs
        self.std_em = std_em
        self.waveform = waveform
        self.ll_nan_factor = ll_nan_factor
        if len(ll_val_phi)>0:
            self.ll_obs_phi = ll_val_phi
        else:
            self.ll_obs_phi = [ [-1]*len(l_signal) for l_signal in self.ll_obs ]
        self.ll_idx_obs_phi = self.computeListIndexPhi()
        self.pi = pi
        self.crop = crop

    def buildEmissionMatrix(self, index_obs):
        """
        Build the emission matrix

        Parameters
        ----------
        index_obs : int
            Index of the current trace.

        Returns
        -------
        A filled emission matrix for the current trace.
        """
        #Get the time/observation dimension from the trace
        l_size=[len(self.ll_obs[index_obs])]
        #Add the size of each stateVar
        l_size.extend([stateVar.nb_substates for stateVar in self.l_stateVar])

        #create the emission matrix, with as much dimensions as variables
        #(plus the time)
        E = np.empty(l_size)
        E = fill_E( E, l_size,  self.l_stateVar, self.ll_obs[index_obs],
                    self.l_stateVar[0].domain, self.l_stateVar[1].domain,
                    self.l_stateVar[2].domain,  self.std_em, self.waveform)

        if len(self.ll_nan_factor)>0:
            for idx, val in enumerate(self.ll_nan_factor[index_obs]):
                if val and idx<E.shape[0]:
                    E[idx] = np.ones(E[idx].shape)

        return E


    def buildInitialDistribution(self):
        """
        Build initial distribution for each hidden variable.

        Returns
        -------
        A filled alpha matrix for time zero.
        """
        l_size = [stateVar.nb_substates for stateVar in self.l_stateVar]
        alpha = np.zeros(l_size)
        alpha.fill(1.)
        l_distrib = [var.pi for var in self.l_stateVar]

        alpha = l_distrib[0]
        for distr in l_distrib[1:]:
            alpha = np.einsum(alpha, list(range(len(alpha.shape))), distr,
                    [len(alpha.shape)], list(range(len(alpha.shape)+1)))

        return alpha

    def computeListIndexPhi(self):
        """
        Convert cell-cycle phase observation to cell-cycle idx.

        Returns
        -------
        A list of list of cell-cycle indexes.
        """
        ll_idx_obs_phi = []
        for l_obs in self.ll_obs_phi:
            l_idx_obs_phi = []
            for obs in l_obs:
                if obs!=-1:
                    l_idx_obs_phi.append( int(round(obs/(2*np.pi) \
                    *  len(self.l_stateVar[0].codomain )))\
                    %len(self.l_stateVar[0].codomain ))
                else:
                    l_idx_obs_phi.append(-1)
            ll_idx_obs_phi.append(l_idx_obs_phi)
        #print(ll_idx_obs_phi)
        return ll_idx_obs_phi



    def doForward(self, E, index_obs):
        """
        Run the foward pass of the forward-backward algorithm.

        Parameters
        ----------
        E : ndarray
            Emission matrix.
        index_obs : interger
            Index of the current trace.

        Returns
        -------
        The alpha ndarray, and the corresponding normalization factors.
        """

        #print("Forward algorithm started")
        l_size = [stateVar.nb_substates for stateVar in self.l_stateVar]
        l_alpha=np.empty([E.shape[0]+1]+l_size, dtype=np.float32)
        #l_alpha=[]
        l_cnorm=[]

        if self.pi is not None:
            alpha = self.pi
        else:
            alpha = self.buildInitialDistribution()
        l_alpha[0] = alpha
        for t in range(E.shape[0]):

            for i, var in enumerate(self.l_stateVar):

                if i==0:
                    #print(self.ll_idx_obs_phi[index_obs][t])
                    if t-1>=0:
                        #print(t-1, len(self.ll_idx_obs_phi[index_obs]))
                        if self.ll_idx_obs_phi[index_obs][t-1]!=-1:
                            alpha = np.tensordot(alpha, var.TR[:,
                                        self.ll_idx_obs_phi[index_obs][t-1],
                                        :   ],
                                    axes=(0,0))
                        else:
                            #plt.matshow(var.TR_no_coupling)
                            #plt.show()
                            alpha = np.tensordot(alpha, var.TR_no_coupling,
                                                axes = (0,0))
                    else:
                        alpha = np.tensordot(alpha, var.TR_no_coupling,
                                                axes = (0,0))
                else:
                    alpha = np.tensordot(alpha,var.TR, axes=(0,0))
                #current variable always become the first dimension after
                #new product


            alpha=np.multiply(alpha,E[t])

            cnorm = np.sum(alpha)
            alpha = alpha / cnorm
            l_alpha[t+1] = alpha
            l_cnorm.append(cnorm)
        return l_alpha, l_cnorm


    def doBackward(self, E, index_obs, l_cnorm):
        """
        Run the backward pass of the forward-backward algorithm.

        Parameters
        ----------
        E : ndarray
            Emission matrix.
        index_obs : interger
            Index of the current trace.
        l_cnorm : list
            List of normalization factors.

        Returns
        -------
        The beta ndarray.
        """
        #print("Backward algorithm started")
        l_size = [stateVar.nb_substates for stateVar in self.l_stateVar]
        l_beta=np.empty([E.shape[0]+1]+l_size, dtype=np.float32)
        #start with initially 1 distribution
        beta = np.zeros(l_size);
        beta.fill(1.)
        l_beta[E.shape[0]] = beta
        #iterate over all times


        for t in range(E.shape[0]-1,-1,-1):
            beta = np.multiply(E[t], beta)
            #beta = np.swapaxes(beta, 0, 1)
            for i, var in enumerate(self.l_stateVar):

                if i==0:
                    if t-1>=0:
                        if self.ll_idx_obs_phi[index_obs][t-1]!=-1:
                            beta = np.tensordot(beta, var.TR[:,
                                    self.ll_idx_obs_phi[index_obs][t-1] , :  ],
                                    axes=(0,1))
                        else:
                            beta = np.tensordot(beta, var.TR_no_coupling ,
                                    axes=(0,1))
                    else:
                        beta = np.tensordot(beta, var.TR_no_coupling ,
                                    axes=(0,1))
                else:
                    beta = np.tensordot(beta, var.TR, axes=(0,1))
                #same than for forward

            #normalize

            beta = beta / l_cnorm[t]
            l_beta[t] = beta
        return l_beta


    def get_trace_limit(self, l_obs_phi):
        """
        Get the limit indexes of the circadian trace according to the cropping
        of the cell-cycle.

        Parameters
        ----------
        l_obs_phi : list
            List of cell-cycle observations.

        Returns
        -------
        The limit indexes of the current trace.
        """
        first_to_keep = 0
        last_to_keep = 0
        for idx, val in enumerate(l_obs_phi[:-1]):
            if val ==-1:
                first_to_keep = idx+1
            if val > -1 :
                last_to_keep = idx
            if val>-1 and l_obs_phi[idx+1] == -1:
                break
            last_to_keep+=1
        #print(first_to_keep, last_to_keep)
        return first_to_keep, last_to_keep

    def crop_results(self, gamma, l_alpha, l_beta, l_cnorm, E, l_obs_phi):
        """
        Crop inferred matrices before and after first and last divisions.

        Parameters
        ----------
        gamma : ndarray
            matrix of probability of being in a given state given all the
                observations.
        l_alpha : list
            list of alpha matrices (for each timepoint)
        l_beta : list
            list of beta matrices (for each timepoint)
        l_cnorm : list
            List of normalization factors.
        E : ndarray
            Emission matrix.
        l_obs_phi : list
            List of cell-cycle observations.

        Returns
        -------
        The cropped matrices.
        """
        gamma_0 =  gamma[0]
        #remove artificial initial condition
        gamma = gamma[1:]
        l_alpha = l_alpha[1:]
        l_beta = l_beta[1:]

        if self.crop:
            first_to_keep, last_to_keep = self.get_trace_limit(l_obs_phi)
            gamma = gamma[first_to_keep:last_to_keep+1]
            l_alpha = l_alpha[first_to_keep:last_to_keep+1]
            l_beta = l_beta[first_to_keep:last_to_keep+1]
            l_cnorm = l_cnorm[first_to_keep:last_to_keep+1]
            E = E[first_to_keep:last_to_keep+1]

        return  gamma_0, gamma, l_alpha, l_beta, E, l_cnorm

    def forwardBackward(self, index_obs, backward=True, em = False,
                        normalized = False):
        """
        Run the complete forwardbackward for a given trace.

        Parameters
        ----------
        index_obs : int
            Index of the current trace.
        backward : bool
            If false, skip the backward pass (useful for debug).
        em : bool
            If true, ignore the annotated zones of the signal.
        normalized: bool
            Normalize or not the log probability of the signal.

        Returns
        -------
        The gamma matrices, with the corresponding trace probabilities.
        """

        #Build emission matrix for the current trace
        E=self.buildEmissionMatrix(index_obs)
        l_alpha, l_cnorm = self.doForward(E, index_obs)
        if not backward:
            return self.logP(l_cnorm, index_obs, normalized)
        else:
            l_beta = self.doBackward(E, index_obs, l_cnorm)
            gamma = np.multiply(l_alpha, l_beta)
            gamma_0, gamma, l_alpha, l_beta, E, l_cnorm = \
            self.crop_results(gamma, l_alpha, l_beta, l_cnorm, E,
                                self.ll_obs_phi[index_obs])
            if np.isnan(np.min(gamma)):
                gamma = None
        if em:
            if len( self.ll_nan_factor)>0:
                return  gamma_0, gamma, E, l_alpha, l_beta, l_cnorm, \
                self.logP(l_cnorm, index_obs),  self.ll_idx_obs_phi[index_obs], \
                self.ll_obs[index_obs], self.ll_nan_factor[index_obs]
            else:
                return  gamma_0, gamma, E, l_alpha, l_beta, l_cnorm, \
                self.logP(l_cnorm, index_obs),  self.ll_idx_obs_phi[index_obs],\
                 self.ll_obs[index_obs], None
        else:
            return gamma, self.logP(l_cnorm, index_obs, normalized = True)


    """ DEPRECATED
    def forwardBackwardPrior(self):
        l_gamma=[]
        for index_obs in range(len(self.ll_obs)):
            E=self.buildEmissionMatrix(index_obs)
            l_alpha, l_cnorm = self.doForward(E, index_obs)
            l_beta = self.doBackward(E, index_obs, l_cnorm)
            gamma = np.multiply(l_alpha, l_beta)
            gamma = np.sum(gamma, axis=(2,3))
            l_gamma.append(gamma)
        return l_gamma
    """

    def logPCheck(self, l_cnorm, l_beta):
        """
        Check that the data probability is the same according to the forward and
        the backward (useful for debug).

        Parameters
        ----------
        l_cnorm : list
            List of normalization factors.
        l_beta : list
            list of beta matrices (for each timepoint)
        """
        #forward
        logP_f = np.sum(np.log(l_cnorm))
        #backward
        logP_b = np.sum(np.log(l_cnorm)) + np.log(np.sum(l_beta[0]))\
                +np.log(1/np.size(l_beta[0]))
        print(logP_f, logP_b)

    def logP(self, l_cnorm, index_obs, normalized = False):
        """
        Compute the log-probability of a given trace.

        Parameters
        ----------
        l_cnorm : list
            List of normalization factors.
        index_obs : int
            Index of the current trace.
        normalized: bool
            Normalize or not the log probability of the signal.

        Returns
        -------
        The log-probability of the trace.
        """
        first_to_keep, last_to_keep = \
                                self.get_trace_limit(self.ll_obs_phi[index_obs])
        if len(self.ll_nan_factor)>0:
            if self.crop:
                dm = self.ll_nan_factor[index_obs][first_to_keep:last_to_keep+1]
                for i, val in enumerate(dm):
                    if val:
                        l_cnorm[i] = np.nan
            else:
                for i, val in enumerate(self.ll_nan_factor[index_obs]):
                    if val:
                        l_cnorm[i] = np.nan


            l_cnorm = [x for x in l_cnorm if not np.isnan(x)]
        if normalized:
            return np.sum(np.log(l_cnorm))/len(l_cnorm)
        else:
            return np.sum(np.log(l_cnorm))

    def run_fb(self, project = False):
        """
        Run the complete forwardbackward for several traces.

        Parameters
        ----------
        project : bool
            Marginalize the hidden variables probability distributions to keep
            only the phase (to save memory).

        Returns
        -------
        The gamma matrices, with the corresponding trace probabilities, for all
        traces.
        """
        l_gamma=[]
        l_logP=[]
        for index_obs in range(len(self.ll_obs)):
            if index_obs%50==0:
                print("Current trace : ", index_obs)
            gamma, logP = self.forwardBackward(index_obs)
            if gamma is not None:
                if project:
                    gamma = np.sum(gamma, axis=(2,3))

                l_gamma.append(gamma)
                l_logP.append(logP)
            else:
                print("Not conserved trace since nan in gamma")
                l_gamma.append(None)
                l_logP.append(-100000)
        print("Done")

        return l_gamma, l_logP


    def run(self, project = False):
        """
        Run the complete forwardbackward for several traces.

        Parameters
        ----------
        project : bool
            Marginalize the hidden variables probability distributions to keep
            only the phase (to save memory).

        Returns
        -------
        The gamma matrices, with the corresponding trace probabilities, for all
        traces.
        """
        return self.run_fb(project)

    def run_em(self):
        """
        Run the complete forwardbackward for several traces, and return all
        matrices (greedy, but needed for EM).

        Returns
        -------
        All the matrices inferred during the FB algorithm.
        """
        l_gamma_0 = []
        l_gamma=[]
        l_logP=[]
        ll_alpha = []
        ll_beta = []
        l_E=[]
        ll_cnorm = []
        ll_idx_phi = []
        ll_signal = []
        ll_nan_circadian_factor = []
        for index_obs in range(len(self.ll_obs)):
            #print("Current trace : ", index_obs)
            (gamma_0, gamma, E, l_alpha, l_beta, l_cnorm, logP, l_idx_phi,
            l_obs, l_nan ) = self.forwardBackward(index_obs, em=True)
            if gamma is not None:


                ll_alpha.append(l_alpha)
                ll_beta.append(l_beta)
                l_E.append(E )

                l_gamma_0.append(gamma_0)
                ll_cnorm.append(l_cnorm)
                l_gamma.append(gamma)
                l_logP.append(logP)

                ll_idx_phi.append(l_idx_phi)
                ll_signal.append(l_obs)
                ll_nan_circadian_factor.append(l_nan)
            else:
                print("Not conserved trace since nan in gamma")

        print("Done")
        return l_gamma_0, l_gamma, l_logP, ll_alpha,  ll_beta, l_E, ll_cnorm, \
                ll_idx_phi, ll_signal, ll_nan_circadian_factor

    def runOpti(self, normalized = False):
        """
        Run only the forward algorithm to compute logP, useful to optimize
        parameters.

        Returns
        -------
        The list of log-probabilities for all traces.
        """
        l_logP=[]
        for index_obs in range(len(self.ll_obs)):
            print("Current trace : ", index_obs)
            logP = self.forwardBackward(index_obs, backward=False,
                                                normalized = False)
            #print("logP: ", logP)
            l_logP.append(logP)
        print("Done")
        return l_logP
