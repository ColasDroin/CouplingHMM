# -*- coding: utf-8 -*-
### Module import
import numpy as np
import scipy.stats as st

class StateVarSemiCoupled:
    """
    This class is used to store and compute all the information related to the
    coupled hidden variables (e.g. intial condition, transition matrix, domain).
    """
    def __init__(self, name_variable, l_boundaries, nb_substates, f_trans,
                 l_parameters_f_trans, F, l_boundaries_covariable,
                 nb_substates_covariable, dt):
        """
        Constructor of StateVarSemiCoupled.

        Parameters
        ----------
        name_variable : string
            Name of the current hidden variable.
        l_boundaries : list
            Tuple containing the boundaries of the variable domain.
        nb_substates : integer
            Number of states of the current hidden variable.
        f_trans : function
            Transition kernel of the current process.
        l_parameters_f_trans : list
            List of parameters needed to compute the transition kernel.
        F : ndarray
            Coupling function
        l_boundaries_covariable : list
            Tuple containing the boundaries of the coupled variable domain.
        nb_substates_covariable : integer
            Number of states of the coupled hidden variable.
        dt : float
            Time resolution.
        """
        self.name_variable = name_variable
        self.l_boundaries = l_boundaries
        self.domain = np.linspace(l_boundaries[0],l_boundaries[1], nb_substates,
                                                                endpoint=False)
        self.nb_substates = nb_substates
        self.increment = (l_boundaries[1]-l_boundaries[0])/nb_substates

        self.codomain = np.linspace(l_boundaries_covariable[0],
                                    l_boundaries_covariable[1],
                                    nb_substates_covariable, endpoint=False)

        self.f_trans = f_trans
        self.l_parameters_f_trans = l_parameters_f_trans
        self.F = F
        self.TR = self.buildTransitionMatrix(dt)
        self.TR_no_coupling = self.buildUncoupledTransitionMatrix(dt)
        self.pi = self.computeInitialDistribution(dt)


    def buildTransitionMatrix(self, dt):
        """
        Build the transition matrix for every state according to a gaussian
        distribution.

        Parameters
        ----------
        dt : float
            Time resolution.

        Returns
        -------
        Transition matrix of the current variable.
        """
        TR = np.zeros([self.nb_substates, len(self.codomain),self.nb_substates])
        for idx1, var1 in enumerate(self.domain):
            for idx2, var2 in enumerate(self.codomain):

                mean, std = self.f_trans(var1, self.l_parameters_f_trans, dt,
                                         var2, self.F)
                for r in range(-2,3):
                    TR[idx1,idx2,:] = TR[idx1,idx2,:] \
                                + st.norm.pdf(self.domain, mean + r*2*np.pi,std)
                #TR[idx1,idx2,:] = TR[idx1,idx2,:]/np.sum(TR[idx1,idx2,:])
                """
                half_domain = (self.l_boundaries[1]-self.l_boundaries[0])/2
                mean, std = self.f_trans(0, self.l_parameters_f_trans, dt,
                                                                    None, None)
                distr = st.norm.cdf(np.linspace(-half_domain+self.increment/2,
                        half_domain+self.increment/2, self.nb_substates),
                        mean, std) \
                        -st.norm.cdf(np.linspace(-half_domain-self.increment/2,
                        half_domain-self.increment/2, self.nb_substates), mean,
                        std)
                distr = distr / np.sum(distr)
                for i, xi in enumerate(self.domain):
                    TR[i,:] = np.roll(distr, i-int(self.nb_substates/2))
                """
        return TR

    def buildUncoupledTransitionMatrix(self, dt):
        """
        Build the transition matrix for every state according to a gaussian
        distribution, ignoring the coupling (to deal with non-dividing traces).

        Parameters
        ----------
        dt : float
            Time resolution.

        Returns
        -------
        Transition matrix of the current variable ignoring all coupling.
        """
        TR = np.zeros([self.nb_substates, self.nb_substates])
        for idx1, var1 in enumerate(self.domain):

            mean, std = self.f_trans(var1, self.l_parameters_f_trans, dt,
                                     None, None)
            for r in range(-2,3):
                TR[idx1,:] = TR[idx1,:] + st.norm.pdf(self.domain,
                                                      mean + r*2*np.pi,std)
            #TR[idx1,:] = TR[idx1,:]/np.sum(TR[idx1,:])

            '''
            half_domain = (self.l_boundaries[1]-self.l_boundaries[0])/2
            mean, std = self.f_trans(0, self.l_parameters_f_trans, dt, None,
                                     None)
            distr = st.norm.cdf(np.linspace(-half_domain+self.increment/2,
                                half_domain+self.increment/2,
                                self.nb_substates), mean, std) \
                    -st.norm.cdf(np.linspace(-half_domain-self.increment/2,
                    half_domain-self.increment/2, self.nb_substates), mean, std)
            print(np.sum(distr))
            distr = distr / np.sum(distr)
            for i, xi in enumerate(self.domain):
                TR[i,:] = np.roll(distr, i-int(self.nb_substates/2))
            '''
        return TR


    def computeInitialDistribution(self, dt):
        """
        Compute the initial distribution of the current variable.

        Parameters
        ----------
        dt : float
            Time resolution.

        Returns
        -------
        Array containing the initial distribution of the current variable.
        """
        pi = np.array([1/self.nb_substates]*self.nb_substates)
        return pi

    def return_increment(self, var1, var2, dt):
        """
        Compute the random increment of the current process given the current
        state of the process and the coupled process (useful for stochastic
        simulations).

        Parameters
        ----------
        var1 : float
            State of the current process.
        var2 : float
            State of the coupled process.
        dt : float
            Time resolution.

        Returns
        -------
        New value of the current process after the random increment.
        """
        mean, std = self.f_trans(var1, self.l_parameters_f_trans, dt, var2,
                                                                        self.F)
        xi2 = st.norm.rvs( mean, std)
        return xi2%(2*np.pi)
