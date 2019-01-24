# -*- coding: utf-8 -*-
### Module import
import numpy as np
import sys
import os
import scipy.stats as st
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp1d

###Local modules
#sys.path.insert(0, os.path.realpath('./Classes'))
from StateVar import StateVar
from StateVarSemiCoupled import StateVarSemiCoupled

class HMMsim:
    """
    This class is used to simulate traces using a stochastic version of the
    system.
    """
    def __init__(self, l_stateVar, model, std_em, waveform = None, dt=0.5,
                 uniform = True, T_phi = 22 ):
        """
        Constructor of HMMsim.

        Parameters
        ----------
        l_stateVar : list
            List of hidden variables.
        model : function
            Signal model.
        std_em : float
            Observation noise.
        waveform : list
            Waveform
        dt : float
            Time resolution (default = 0.5)
        uniform : bool
            If true, uniform initial condition in the phase-space.
        T_phi : float
            Cell-cycle period used for simulation.
        """
        self.l_stateVar = l_stateVar
        self.std_em = std_em
        self.model = model
        if waveform is not None:
            self.waveform = interp1d(np.linspace(0,2*np.pi, len(waveform)+1,
                                                 endpoint = True),
                                     np.append(waveform, waveform[0] ))
        else:
            self.waveform = waveform
        self.dt = dt
        self.uniform = uniform
        self.w_phi = 2*np.pi/T_phi

    def simulate(self, tf=80):
        """
        Simulate a stochastic trace, for a given time. Either start from a
        completely uniform distribution on the phase space, either start from
        phi = 0, and theta located on the attractor. Always uniform distribution
        for OU processes

        Parameters
        ----------
        tf : int
            Simulation length.
        Returns
        -------
        A list of tuples (for theta and phi) of list (for each process) and a
        list of tuples for the observations.
        """
        t_l_xi=( [], [] )
        if self.uniform:
            t_l_xi[1].append(np.random.random()*2*np.pi)
        else:
            t_l_xi[1].append(  0  )
        for var in self.l_stateVar:
            if not var.name_variable == "Theta":
                t_l_xi[0].append(np.random.choice(var.domain, p=var.pi))
            else:
                if self.uniform:
                    t_l_xi[0].append(np.random.choice(var.domain, p=var.pi))
                else:
                    t_l_xi[0].append(  st.norm.rvs(37/48*2*np.pi, 0.02  )  )


        l_t_l_xi = [ copy.deepcopy(t_l_xi) ]
        l_t_obs = [ ( self.model(t_l_xi[0], self.waveform), np.nan )   ]

        #execute the HMM to produce a signal:
        t=0
        while t<tf:
            l_ph = copy.copy([  t_l_xi[0][0]  ,   t_l_xi[1][0]  ])
            for i, xi in enumerate(t_l_xi[0]):
                if not self.l_stateVar[i].name_variable == "Theta":
                    t_l_xi[0][i] = \
                        self.l_stateVar[i].return_increment(xi, self.dt)
                else:
                    t_l_xi[0][i] = self.l_stateVar[i].return_increment(l_ph[0],
                                                                       l_ph[1],
                                                                       self.dt)
            t_l_xi[1][0] += self.w_phi*self.dt
            if t_l_xi[1][0]>=2*np.pi:
                t_l_xi[1][0] = t_l_xi[1][0]%(2*np.pi)

            ###BUG TIME SAMPLING IF DT TOO SMALL
            if t%0.5<10**-4:
                l_t_obs.append((self.model(t_l_xi[0], self.waveform)\
                                + st.norm.rvs(0, self.std_em  ),  np.nan ))
                l_t_l_xi.append(copy.deepcopy(t_l_xi))
            t+=self.dt
        #return signal and associed substates
        return l_t_l_xi, l_t_obs


    def simulate_n_traces(self, nb_traces, tf=50):
        """
        Simulate several stochastic traces of a given length.

        Parameters
        ----------
        nb_traces : int
            Number of simulated traces
        tf : int
            Simulation length.
        Returns
        -------
        A list (for each trace) of list (for time) of tuples (for theta and phi)
        of list (for each process) and a list (for each trace) of list (for
        time) of tuples for the observations.
        """
        ll_t_l_xi=[]
        ll_t_obs = []
        for i in range(nb_traces):
            l_t_l_xi, t_l_obs = self.simulate(tf)
            ll_t_l_xi.append( l_t_l_xi)
            ll_t_obs.append(t_l_obs)
        return ll_t_l_xi, ll_t_obs
