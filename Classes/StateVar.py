# -*- coding: utf-8 -*-
### Module import
import numpy as np
import scipy.stats as st

class StateVar:
    """
    This class is used to store and compute all the information related to the
    hidden variables (e.g. intial condition, transition matrix, domain).
    """
    def __init__(self, name_variable, l_boundaries, nb_substates,
                 f_trans, l_parameters_f_trans, dt):
        """
        Constructor of StateVar.

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
        dt : float
            Time resolution.
        """
        self.name_variable = name_variable
        self.l_boundaries = l_boundaries
        self.nb_substates = nb_substates
        self.increment = (l_boundaries[1]-l_boundaries[0])/nb_substates
        self.domain = np.linspace(l_boundaries[0],l_boundaries[1],
                                                nb_substates, endpoint=False)
        self.f_trans = f_trans
        self.l_parameters_f_trans = l_parameters_f_trans
        self.TR = self.buildTransitionMatrix(dt)
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

        TR = np.zeros([self.nb_substates, self.nb_substates])
        for i, xi in enumerate(self.domain):
            mean, std = self.f_trans(xi, self.l_parameters_f_trans, dt)
            #TR[i,:] = st.norm.cdf(self.domain+self.increment/2 , mean, std)
            #            - st.norm.cdf(self.domain-self.increment/2 , mean, std)
            TR[i,:] = st.norm.pdf(self.domain, mean, std)
            #TR[i,:] = TR[i,:]/np.sum(TR[i,:])
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

        #mean, std = self.f_trans(self.l_parameters_f_trans[0],
        #                                        self.l_parameters_f_trans, dt)
        #pi =  st.norm.pdf(self.domain, mean , std)
        #pi = pi/np.sum(pi)
        pi = np.array([1/self.nb_substates]*self.nb_substates)
        return pi

    def return_increment(self, xi, dt):
        """
        Compute the random increment of the current process given the current
        state (useful for stochastic simulations).

        Parameters
        ----------
        xi : float
            Current value of the process.
        dt : float
            Time resolution.

        Returns
        -------
        New value of the process after the random increment.
        """
        mean, std = self.f_trans(xi, self.l_parameters_f_trans, dt)
        xi2 = st.norm.rvs( mean, std)
        return xi2
