# -*- coding: utf-8 -*-
### Module import
import sys
import os
import random
import matplotlib
import pickle
import numpy as np
import signal
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy import io
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy import interpolate
import scipy.integrate as spi
from multiprocessing import Pool
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from scipy.interpolate import BSpline
import matplotlib.animation as animation
import copy
import seaborn as sn

#nice plotting style
matplotlib.rcParams['pdf.fonttype'] = 42
sn.set_style("whitegrid", {'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

#sys.path.insert(0, os.path.realpath('./Classes'))

class DetSim:
    """
    This class is used to simulate traces using a deterministic version of the
    system.
    """
    def __init__(self, l_parameters, cell, temperature, upfolder = True):
        """
        Constructor of DetSim.

        Parameters
        ----------
        l_parameters : list
            A list of all parameters needed for simulation:
            dt : time increment (float)
            sigma_em_circadian : circadian observation noise (float)
            W : waveform (list)
            pi : initial state distribution (ndarray)
            N_theta : number of circadian states (integer)
            std_theta : circadian phase diffusion parameter (float)
            period_theta : circadian period (integer)
            l_boundaries_theta : Limits of the circadian domain (list)
            w_theta : circadian speed (float)
            N_phi : number of cell-cyle states (integer)
            std_phi : cell-cyle phase diffusion parameter (float)
            period_phi : cell-cyle period (integer)
            l_boundaries_phi : Limits of the cell-cycle domain (list)
            w_phi : cell-cycle speed (float)
            N_amplitude_theta : number of amplitude states (integer)
            mu_amplitude_theta : mean amplitude (float)
            std_amplitude_theta : amplitude standard deviation (float)
            gamma_amplitude_theta : amplitude regression constant (float)
            l_boundaries_amplitude_theta : limits of the amplitude domain (list)
            N_background_theta : number of background states (integer)
            mu_background_theta : mean background (float)
            std_background_theta : background standard deviation (float)
            gamma_background_theta : background regression constant (float)
            l_boundaries_background_theta : limits of the background domain
                                            (list)
            F : coupling function (ndarray)
        cell : string
            The type of cell used in the experiment.
            Either 'NIH3T3' or 'U2OS'
        temperature : integer
            The temperature condition used in the experiment.
            Either 34, 37 or 40
        """
        [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = l_parameters

        self.gamma_A = gamma_amplitude_theta
        self.gamma_B = gamma_background_theta
        self.T_theta = 2*np.pi/w_theta
        self.T_phi = 2*np.pi/w_phi
        self.F = F
        self.pi = pi
        self.sigma_theta = std_theta
        self.mu_A = mu_amplitude_theta
        self.mu_B = mu_background_theta
        self.waveform = W

        #compute F for a large domain not to have to deal with periodicity
        F_temp = np.hstack((self.F,self.F,self.F))
        F_temp = np.vstack((F_temp, F_temp, F_temp))
        dom_theta = np.linspace(-2*np.pi,4*np.pi, 3*self.F.shape[0],
                                endpoint = False)
        dom_phi = np.linspace(-2*np.pi,4*np.pi, 3*self.F.shape[1],
                                endpoint = False)
        self.F_func = interpolate.interp2d(dom_theta, dom_phi, F_temp.T,
                                           kind='cubic', bounds_error= True)
        #interpolate waveform
        waveform_temp = self.waveform + self.waveform[0]
        dom_w = np.linspace(0,2*np.pi, len(waveform_temp), endpoint = True)
        self.waveform_func = interpolate.interp1d(dom_w, waveform_temp)

        self.cell = cell
        self.temperature = temperature
        #smooth the function with Fourier approximation (DOESNT WORK FOR NOW)
        self.fourier= False

        if upfolder:
            self.upfolder ='../'
        else:
            self.upfolder =''

    def simulate(self, tf=80, full_simulation = True, rand = True):
        """
        Simulate a deterministic trace, for a given time.

        Parameters
        ----------
        tf : int
            Simulation length.
        full_simulation : bool
            Simulate only the phase (False) or the full system (True).
        rand : bool
            Random initial condition (True) or not (False).

        Returns
        -------
        An array for the time domain and a ndarray for the simulated variables.
        """

        #define system of ODE
        def system(Y, t):
            theta, phi = Y
            d_theta_dt = 2*np.pi/self.T_theta \
                         + self.F_func(theta%(2*np.pi), phi%(2*np.pi))
            d_phi_dt = 2*np.pi/self.T_phi
            return [d_theta_dt, d_phi_dt]

        def system_full(Y, t):
            theta, phi, A, B = Y
            d_theta_dt = 2*np.pi/self.T_theta \
                        + self.F_func(theta%(2*np.pi), phi%(2*np.pi))
            d_phi_dt = 2*np.pi/self.T_phi
            d_A_dt = -self.gamma_A*(A -self.mu_A )
            d_B_dt = -self.gamma_B*(B -self.mu_B )
            return [d_theta_dt, d_phi_dt, d_A_dt, d_B_dt]

        #define intial condition
        if rand:
            theta_0 = random.random()*2*np.pi
            phi_0 = random.random()*2*np.pi
        else:
            theta_0 = 6.
            phi_0 = 0.
        Y0 = [theta_0, phi_0]
        Y0_full = [theta_0, phi_0, random.random()*2*self.mu_A,
                   random.random()*2*self.mu_B]
        tspan = np.linspace(0, tf, abs(2*tf))

        if full_simulation:
            vect_Y = odeint(system_full, Y0_full, tspan)
        else:
            vect_Y = odeint(system, Y0, tspan)

            ''' COMMENTED SINCE ODEINT WORKS JUST AS WELL
            def system_vode(t, Y):
                Eqs = np.zeros((2))
                Eqs[0] = 2*np.pi/self.T_theta
                        + self.F_func(Y[0]%(2*np.pi), Y[1]%(2*np.pi))
                Eqs[1] = 2*np.pi/self.T_phi
                return Eqs

            ode =  spi.ode(system_vode)

            # BDF method suited to stiff systems of ODEs
            t_step = 0.5
            ode.set_integrator('dopri5',nsteps=500,method='bdf')
            ode.set_initial_value(Y0,0)

            ts = [ode.t]
            ys = [ode.y]

            while ode.successful() and ode.t < tf:
                ode.integrate(ode.t + t_step)
                ts.append(ode.t)
                ys.append(ode.y)


            tspan = np.vstack(ts)
            vect_theta, vect_phi = np.vstack(ys).T
            vect_Y = np.vstack((vect_theta, vect_phi)).T
            '''
        return tspan, vect_Y

    def plot_signal(self, tf = 200, full_simulation = True, save = False):
        """
        Plot a temporal trace.

        Parameters
        ----------
        tf : int
            Simulation length.
        full_simulation : bool
            Simulate only the phase (False) or the full system (True).
        save : bool
            Save plot (True) or not (False).
        """
        tspan, vect_Y = self.simulate(tf, full_simulation)

        l_div =[]
        for t1, t2, phi1, phi2 in zip(tspan[:-1], tspan[1:],
                                      vect_Y[:-1,1],vect_Y[1:,1]):

            if (phi1%(2*np.pi)-phi2%(2*np.pi))>0:
                l_div.append((t1+t2)/2)

        if full_simulation:
            l_signal =  np.exp(vect_Y[:,2]) \
                        *self.waveform_func(vect_Y[:,0]%(2*np.pi))+vect_Y[:,3]
        else:
            l_signal = self.waveform_func(vect_Y[:,0]%(2*np.pi))
        plt.plot(tspan, l_signal)
        for t_div in l_div:
            plt.axvline(t_div)
        plt.xlabel("Time")
        plt.ylabel("Circadian signal")
        if save:
            plt.savefig(self.upfolder+'Results/DetSilico/trace_'+self.cell+'_'
                        +str(self.temperature)+'.pdf')
        plt.show()
        plt.close()

    def plot_trajectory(self ,ti = 0, tf = 80, rand = True, save = False, K = 1,
                        T_phi  = None, full_simulation = False):
        """
        Plot a trace in the phase space.

        Parameters
        ----------
        ti : int
            First recorded point of the simulation.
        tf : int
            Simulation length.
        rand : bool
            Random initial condition (True) or not (False).
        save : bool
            Save plot (True) or not (False).
        K : float
            Couplint strength.
        T_phi : float
            Cell-cycle period.
        full_simulation : bool
            Simulate only the phase (False) or the full system (True).
        Returns
        -------
        A list of list for both the circadian and cell-cycle phases.
        Lists are truncated everytime a boundary of the phase domain is crossed.
        """
        real_coupling_function = copy.copy(self.F)
        real_T_phi = self.T_phi
        self.F = K * real_coupling_function

        F_temp = np.hstack((self.F,self.F,self.F))
        F_temp = np.vstack((F_temp, F_temp, F_temp))
        dom_theta = np.linspace(-2*np.pi,4*np.pi, 3*self.F.shape[0],
                                endpoint = False)
        dom_phi = np.linspace(-2*np.pi,4*np.pi, 3*self.F.shape[1],
                                endpoint = False)
        self.F_func = interpolate.interp2d(dom_theta, dom_phi, F_temp.T,
                                           kind='cubic', bounds_error= True)

        if T_phi is not None:
            self.T_phi = T_phi

        tspan, vect_Y = self.simulate(tf, full_simulation = full_simulation,
                                      rand = rand)
        tspan = tspan[ti:]
        vect_Y = vect_Y[ti:]
        l_phase_theta_flat =  vect_Y[:,0]%(2*np.pi)
        l_phase_phi_flat =  vect_Y[:,1]%(2*np.pi)

        #cut trace so that each time a boundary is crossed it creates a new plot
        ll_phase_theta=[]
        ll_phase_phi=[]
        l_phase_theta=[]
        l_phase_phi=[]
        for theta2, phi2, theta1, phi1 in zip(l_phase_theta_flat[1:],
                                              l_phase_phi_flat[1:],
                                              l_phase_theta_flat[:-1],
                                              l_phase_phi_flat[:-1]):
            l_phase_theta.append(theta1)
            l_phase_phi.append(phi1)
            if abs(theta2-theta1)>np.pi or abs(phi2-phi1)>np.pi:
                ll_phase_theta.append(np.array(l_phase_theta))
                ll_phase_phi.append(np.array(l_phase_phi))
                l_phase_theta=[]
                l_phase_phi=[]
        ll_phase_theta.append(np.array(l_phase_theta))
        ll_phase_phi.append(np.array(l_phase_phi))

        #plot
        plt.figure(figsize=(5,5))
        for l_phase_theta, l_phase_phi in zip(ll_phase_theta,ll_phase_phi):
            plt.plot(l_phase_theta[:-1], l_phase_phi[:-1], lw = 2,
                     color = 'lightblue')
        plt.xlim(10**-2,2*np.pi-10**-2)
        plt.ylim(10**-2,2*np.pi-10**-2)
        if T_phi is None:
            plt.xlabel("Circadian phase")
            plt.ylabel("Cell-cycle phase")
        else:
            if T_phi>=30:
                plt.xlabel("Circadian phase")

            if (T_phi-14)%4==0:
                plt.ylabel("Cell-cycle phase")
        if save:
            plt.savefig(self.upfolder+'Results/DetSilico/trajectory_'
                        +self.cell+'_'+str(self.temperature)+'_' + str(K)
                        +'_' + str(T_phi)+ '.pdf')
        #plt.show()
        plt.close()

        self.F = real_coupling_function
        F_temp = np.hstack((self.F,self.F,self.F))
        F_temp = np.vstack((F_temp, F_temp, F_temp))
        self.F_func = interpolate.interp2d(dom_theta, dom_phi, F_temp.T,
                                           kind='cubic', bounds_error= True)
        self.T_phi = real_T_phi

        return ll_phase_theta, ll_phase_phi

    def plot_vectorfield(self, save = True):
        """
        Plot vectofield in the phase space.

        Parameters
        ----------
        save : bool
            Save plot (True) or not (False).
        """

        X = np.linspace(0,2*np.pi,self.F.shape[0], endpoint = False)
        Y = np.linspace(0,2*np.pi,self.F.shape[1], endpoint = False)

        U = 2*np.pi/self.T_theta + self.F.T
        V = np.empty((self.F.shape[0], self.F.shape[1]))
        V.fill(2*np.pi/self.T_phi)


        fig0, ax0 = plt.subplots()
        strm = ax0.streamplot(X, Y, U, V, color=U, linewidth=1.5,
                              cmap='coolwarm')
        cbar = fig0.colorbar(strm.lines)
        ax0.axis([0, 2*np.pi, 0, 2*np.pi])
        plt.xlabel("Circadian phase")
        plt.ylabel("Cell-cycle phase")
        #plt.title("Speed vectorfield")
        cbar.set_label('Circadian speed')

        #fig1, (ax1, ax2) = plt.subplots(ncols=2)
        #ax1.streamplot(X, Y, U, V, density=[0.5, 1])
        #speed = np.sqrt(U*U + V*V)
        #lw = 5*speed / speed.max()
        #ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)

        if save:
            plt.savefig(self.upfolder+'Results/DetSilico/vectorfield_'
                        +self.cell+'_'+str(self.temperature)+'.pdf')
        plt.show()

    def plot_vectorfield_bis(self, T_phi, save = True):
        """
        Plot vectofield in the phase space (version 2).

        Parameters
        ----------
        T_phi : float
            Cell-cycle period.
        save : bool
            Save plot (True) or not (False).
        """
        F_temp = np.hstack((self.F,self.F,self.F))
        F_temp = np.vstack((F_temp, F_temp, F_temp))
        F_temp = F_temp[48:48+48+1,48:48+48+1]


        X = np.linspace(0,2*np.pi,F_temp.shape[0], endpoint = True)
        Y = np.linspace(0,2*np.pi,F_temp.shape[1], endpoint = True)

        U = 2*np.pi/self.T_theta + F_temp.T
        V = np.empty((F_temp.shape[0], F_temp.shape[1]))
        V.fill(2*np.pi/T_phi)

        #sample to reduce the number of arrows
        l_idx_x = [x for x in range(0,F_temp.shape[0],5)]
        l_idx_y = [x for x in range(0,F_temp.shape[1],5)]
        X = X[l_idx_x]
        Y = Y[l_idx_y]
        U = [u[l_idx_y] for u in U[l_idx_x]]
        V = [v[l_idx_y] for v in V[l_idx_x]]
        C = [c[l_idx_y] for c in F_temp.T[l_idx_x]]

        #get attractor
        ll_phase_theta, ll_phase_phi = self.plot_trajectory(ti = 5000,
                                                        tf = 5100,
                                                        rand = True,
                                                        save = False,
                                                        K = 1,
                                                        T_phi  = T_phi)
        #get repeller
        ll_phase_theta_rep, ll_phase_phi_rep = self.plot_trajectory(ti = 5000,
                                                        tf = -5100,
                                                        rand = True,
                                                        save = False,
                                                        K = 1,
                                                        T_phi  = T_phi,
                                                        full_simulation = True)

        #hide the grid
        sn.set_style("whitegrid",
                    {'grid.color': 'white',  'xtick.direction': 'out',
                    'xtick.major.size': 6.0, 'xtick.minor.size': 3.0,
                    'ytick.color': '.15', 'ytick.direction': 'out',
                    'ytick.major.size': 6.0, 'ytick.minor.size': 3.0})

        #plot
        grb, ax1 = plt.subplots(figsize=(5,5))

        #ax1.grid(b=False)
        #ax1.set_axis_bgcolor('white')

        #plot vectorfield
        #ax1.quiver(np.array(X)/(2*np.pi), np.array(Y)/(2*np.pi), U, V,
                    #C, alpha=.5, cmap = 'coolwarm')
        ax1.quiver(np.array(X)/(2*np.pi), np.array(Y)/(2*np.pi), U, V,
                    color = 'blue', alpha = 0.7)
                    #edgecolor='k', facecolor='None', linewidth=.1)

        #plot attractor
        for l_phase_theta, l_phase_phi in zip(ll_phase_theta,ll_phase_phi):
            ax1.plot(np.array(l_phase_theta)/(2*np.pi),
                    np.array(l_phase_phi)/(2*np.pi),
                    lw = 2, color = 'green', alpha = 0.1)

        #plot repeller
        for l_phase_theta_rep, l_phase_phi_rep in zip(ll_phase_theta_rep,
                                                      ll_phase_phi_rep):
            ax1.plot(np.array(l_phase_theta_rep)/(2*np.pi),
                    np.array(l_phase_phi_rep)/(2*np.pi),
                    lw = 2, color = 'red', alpha = 0.1)

        plt.xlabel(r'Circadian phase $\theta$')
        plt.ylabel(r'Cell-cycle phase $\phi$')
        '''
        plt.plot([0,1],[0.22,0.22], '--', color = 'grey')
        plt.text(x = 0.45, y = 0.14, s='G1', color = 'grey', fontsize=12)
        plt.text(x = 0.46, y = 0.27, s='S', color = 'grey', fontsize=12)
        plt.plot([0,1],[0.84,0.84], '--', color = 'grey')
        plt.text(x = 0.05, y = 0.84-0.08, s='S/G2', color = 'grey', fontsize=12)
        plt.text(x = 0.06, y = 0.84+0.05, s='M', color = 'grey', fontsize=12)
        '''
        plt.title(r"$T_{\phi} = $" + str(T_phi ))

        ax1.set_xlim([0.0,1.])
        ax1.set_ylim([0.0,1.])

        if save:
            plt.savefig(self.upfolder+'Results/DetSilico/vectorfield_'
                        +self.cell+'_'+str(self.temperature)+'_'+str(T_phi)
                        +'.pdf')
        plt.show()
        plt.close()

    def anim_vectorfield(self):
        """
        Animate and save the vectofield.
        """
        #hide the grid
        sn.set_style("whitegrid",
                    {'grid.color': 'white',  'xtick.direction': 'out',
                    'xtick.major.size': 6.0, 'xtick.minor.size': 3.0,
                    'ytick.color': '.15', 'ytick.direction': 'out',
                    'ytick.major.size': 6.0, 'ytick.minor.size': 3.0})

        F_temp = np.hstack((self.F,self.F,self.F))
        F_temp = np.vstack((F_temp, F_temp, F_temp))
        F_temp = F_temp[48:48+48+1,48:48+48+1]
        X = np.linspace(0,2*np.pi,F_temp.shape[0], endpoint = True)
        Y = np.linspace(0,2*np.pi,F_temp.shape[1], endpoint = True)

        #sample to reduce the number of arrows
        l_idx_x = [x for x in range(0,F_temp.shape[0],5)]
        l_idx_y = [x for x in range(0,F_temp.shape[1],5)]
        X = X[l_idx_x]
        Y = Y[l_idx_y]

        l_period = np.arange(8,48,0.05)



        def init_quiver():
            global Q
            global lines
            T_phi = l_period[0]
            U = 2*np.pi/self.T_theta + F_temp.T
            V = np.empty((F_temp.shape[0], F_temp.shape[1]))
            V.fill(2*np.pi/T_phi)
            U = np.array([u[l_idx_y] for u in U[l_idx_x]])
            V = np.array([v[l_idx_y] for v in V[l_idx_x]])
            C = [c[l_idx_y] for c in F_temp.T[l_idx_x]]

            Q = ax.quiver(np.array(X)/(2*np.pi), np.array(Y)/(2*np.pi),
                          U/np.sqrt(U**2+V**2), V/np.sqrt(U**2+V**2),
                          color = 'blue', alpha = 0.7)

            #get attractor
            ll_phase_theta, ll_phase_phi = self.plot_trajectory( ti = 5000,
                                                tf = 5100, rand = True,
                                                save = False, K = 1,
                                                T_phi  = T_phi)
            #get repeller
            ll_phase_theta_rep, ll_phase_phi_rep = self.plot_trajectory( \
                                                ti = 5000, tf = -5100,
                                                rand = True, save = False,
                                                K = 1, T_phi  = T_phi,
                                                full_simulation = True)

            lines = []
            #plot attractor
            for l_phase_theta, l_phase_phi in zip(ll_phase_theta,ll_phase_phi):
                lines.append(ax.plot(np.array(l_phase_theta)/(2*np.pi),
                            np.array(l_phase_phi)/(2*np.pi),
                            lw = 2, color = 'green', alpha = 0.1)[0])

            #plot repeller
            for l_phase_theta_rep, l_phase_phi_rep in zip(ll_phase_theta_rep,
                                                          ll_phase_phi_rep):
                lines.append(ax.plot(np.array(l_phase_theta_rep)/(2*np.pi),
                                     np.array(l_phase_phi_rep)/(2*np.pi),
                                     lw = 2, color = 'red', alpha = 0.1)[0])


            ax.set_title(r"$T_{\phi} = $" + str(T_phi )[:4])
            ax.set_xlabel(r'Circadian phase $\theta$')
            ax.set_ylabel(r'Cell-cycle phase $\phi$')
            ax.set_xlim([0.0,1.])
            ax.set_ylim([0.0,1.])


            return  Q, lines

        def update_quiver(i, ax, fig):
            for line in lines:
                line.set_data([],[])
            #global lines
            lines = []
            T_phi = l_period[i]
            V = np.empty((F_temp.shape[0], F_temp.shape[1]))
            V.fill(2*np.pi/T_phi)
            V = np.array([v[l_idx_y] for v in V[l_idx_x]])
            U = 2*np.pi/self.T_theta + F_temp.T
            U = np.array([u[l_idx_y] for u in U[l_idx_x]])
            Q.set_UVC( U/np.sqrt(U**2+V**2), V/np.sqrt(U**2+V**2))

            #get attractor
            ll_phase_theta, ll_phase_phi = self.plot_trajectory( ti = 1000,
                                                        tf = 1100, rand = True,
                                                        save = False, K = 1,
                                                        T_phi  = T_phi)
            #get repeller
            ll_phase_theta_rep, ll_phase_phi_rep = self.plot_trajectory( \
                                                        ti = 1000, tf = -1100,
                                                        rand = True,
                                                        save = False, K = 1,
                                                        T_phi  = T_phi,
                                                        full_simulation = True)

            #plot attractor
            for l_phase_theta, l_phase_phi in zip(ll_phase_theta,ll_phase_phi):
                lines.append(ax.plot(np.array(l_phase_theta)/(2*np.pi),
                                     np.array(l_phase_phi)/(2*np.pi),
                                     lw = 2, color = 'green', alpha = 0.1)[0])

            #plot repeller
            for l_phase_theta_rep, l_phase_phi_rep in zip(ll_phase_theta_rep,
                                                          ll_phase_phi_rep):
                lines.append(ax.plot(np.array(l_phase_theta_rep)/(2*np.pi),
                                     np.array(l_phase_phi_rep)/(2*np.pi),
                                     lw = 2, color = 'red', alpha = 0.1)[0])


            ax.set_title(r"$T_{\phi} = $" + str(T_phi )[:4])
            return Q, lines

        fig =plt.figure(figsize=(5,5))
        ax = fig.gca()
        #ax.set_title('$t$ = '+ str(t[0]))
        #ax.set_xlabel('$x$')
        #ax.set_ylabel('$y$')

        ani = animation.FuncAnimation(fig, update_quiver,
                                      frames = range(len(l_period)),
                                      init_func=init_quiver,
                                      interval=1,fargs=(ax, fig))

        ani.save(self.upfolder+'Results/DetSilico/vectorfield_'+self.cell+'_'
                 +str(self.temperature)+'.mp4',
                 fps=30, extra_args=['-vcodec', 'libx264'], dpi=600)
        #plt.show()




    def return_period(self, tspan, vect_Y, period_to_remove = 0):
        """
        Return the list of period of a given simulated trace.

        Parameters
        ----------
        tspan : list or array
            Time domain.
        vect_Y : array
            Vector returned by a simulation.
        period_to_remove : int
            Number of periods to remove (to ignore non-steady state)

        Returns
        -------
        A list of periods for a given simulated trace
        """

        l_T = []
        theta0 = vect_Y[0,0]%(2*np.pi)
        for t1, t2, theta1, theta2 in zip(tspan[:-1], tspan[1:],
                                          vect_Y[:-1,0], vect_Y[1:,0]):
            cond1 = (theta1%(2*np.pi)-theta0<0 and theta2%(2*np.pi)-theta0>=0)
            if cond1 and t1>0 and theta2-theta1>0:
                if (t1+t2)/2 - np.sum(l_T)>4 and (t1+t2)/2 - np.sum(l_T)<70:
                    l_T.append( (t1+t2)/2 - np.sum(l_T) )
        #print("periods", l_T)
        for i in range(period_to_remove):
            try:
                del l_T[0]
            except:
                print("Not enough simulation time to remove that many periods")
        return l_T


    def plot_sync_tongues(self, NK, NT, N, tf = 100):
        """
        Plot approximate version of the Arnold tongues

        Parameters
        ----------
        NK : list or array
            Strength domain.
        NT : list or array
            Time domain.
        N : int
            Number of simulated traces.
        tf : int
            Simulation length.
        """
        ll_arnold = np.zeros((NT, NK))
        #explore all possible coupling strength
        real_coupling_function = copy.copy(self.F)
        period_space = np.linspace(5,50,NT)
        coupling_space = np.linspace(0,2, NK)
        for iK, K in enumerate(coupling_space):
            self.F = K * real_coupling_function
            F_temp = np.hstack((self.F,self.F,self.F))
            F_temp = np.vstack((F_temp, F_temp, F_temp))
            dom_theta = np.linspace(-2*np.pi,4*np.pi, 3*self.F.shape[0],
                                    endpoint = False)
            dom_phi = np.linspace(-2*np.pi,4*np.pi, 3*self.F.shape[1],
                                    endpoint = False)
            self.F_func = interpolate.interp2d(dom_theta, dom_phi, F_temp.T,
                                               kind='cubic', bounds_error= True)

            #explore all possible cell-cycle periods
            for iT, T_phi in enumerate(period_space):
                self.T_phi = T_phi
                #generate uniform traces on the plan
                l_T = []
                #print("K, T_phi", K, T_phi)
                for i in range(N):
                    tspan, vect_Y = self.simulate(tf, False)
                    l_T.extend(self.return_period(tspan, vect_Y, 2))
                diff = T_phi - np.mean(l_T)
                ll_arnold[iT,iK] = abs(diff)

        #plt.pcolormesh(F.T, cmap='bwr', vmin=-0.6, vmax=0.6)
        plt.pcolormesh(period_space, coupling_space, ll_arnold.T,
                       cmap = 'binary', vmin = 0, vmax = 3)
        plt.colorbar()
        plt.xlabel("T cell cycle")
        plt.ylabel("K")
        plt.savefig(self.upfolder+'Results/DetSilico/sync_tongue_'+self.cell+'_'
                    +str(self.temperature)+'.pdf')
        plt.show()
        plt.close()

        self.F = real_coupling_function
        F_temp = np.hstack((self.F,self.F,self.F))
        F_temp = np.vstack((F_temp, F_temp, F_temp))
        self.F_func = interpolate.interp2d(dom_theta, dom_phi, F_temp.T,
                                           kind='cubic', bounds_error= True)

    def plot_devil_staircase(self, l_arg):
        """
        Plot Devil staircase and return average circadian speed for a given
        cell-cycle period list.

        Parameters
        ----------
        l_arg : list
            Arguments used by the function (used for multiprocessing):
            speed_space : Speed/frequency domain (array)
            coupling_strength : Coupling strength used for simulation (float)
            tf : Simulation length (int)
            random_init : Random initial condition (bool)
        """
        #print('new staircase')
        (speed_space, coupling_strength, tf, random_init) = l_arg
        print("K = " + str(coupling_strength))
        if self.fourier:
            dom1 = np.linspace(0,2*np.pi, self.F.shape[0], endpoint = False)
            dom2 = np.linspace(0,2*np.pi, self.F.shape[1], endpoint = False)
            Fp = interpolate.interp2d(dom1, dom2 , self.F)
            dom_theta = np.linspace(0,2*np.pi, N_theta, endpoint = False)
            Fp = Fp(dom_theta, dom_theta)
            F_t = np.fft.rfft2(Fp)
            F_mag = np.abs(np.fft.fftshift(F_t))
            F_phase = np.angle(np.fft.fftshift(F_t))
            F_t = np.fft.fftshift(F_t)

            tenth = np.percentile( F_mag.flatten(), 50)
            ll_idx_coef = [(i,j) for i in range(F_mag.shape[0]) \
                            for j in range(F_mag.shape[1]) if F_mag[i,j]<tenth]
            print("remaining coef:" ,
                  F_mag.shape[0]*F_mag.shape[1]-len(ll_idx_coef))

            for (i,j) in ll_idx_coef:
                F_t[i,j] = 0
                F_mag[i,j] = 0
                F_phase[i,j] = 0
            F_t = np.fft.ifftshift(F_t)
            def func(grid_phase):
                return np.fft.irfft2( grid_phase )
            self.F_func = func

        else:
            real_coupling_function = copy.copy(self.F)
            real_T_phi = self.T_phi
            self.F = self.F * coupling_strength
            print(self.F[0,10])
            F_temp = np.hstack((self.F,self.F,self.F))
            F_temp = np.vstack((F_temp, F_temp, F_temp))
            dom_theta = np.linspace(-2*np.pi,4*np.pi, 3*self.F.shape[0],
                                    endpoint = False)
            dom_phi = np.linspace(-2*np.pi,4*np.pi, 3*self.F.shape[1],
                                    endpoint = False)
            self.F_func = interpolate.interp2d(dom_theta, dom_phi, F_temp.T,
                                               kind='cubic', bounds_error= True)

        l_wm = []
        #explore all possible cell-cycle periods
        new_method = True
        for iw, w_phi in enumerate(speed_space):
            self.T_phi = 2*np.pi/w_phi
            tspan, vect_Y = self.simulate(tf, False, rand = random_init)
            if not new_method:
                try:
                    l_wm.append( w_phi-(vect_Y[-1,0]\
                                            -vect_Y[-int(tf/4),0])/(tf/4/2) )

                except:
                    print("point ", w_phi, "with strength ",
                          coupling_strength, " was ignored")
                    l_wm.append(np.nan)
            else:
                #first interpolate time
                real_tf = (len(vect_Y[:,0])-1)*0.5
                dom_interp = np.linspace(0,(len(vect_Y[:,0])-1)*0.5,
                                            len(vect_Y[:,0]), endpoint = True)
                func = interpolate.interp1d(vect_Y[:,0],
                                            dom_interp , kind='linear')
                real_ti = func(vect_Y[-1,0]-8*np.pi)
                wm = (8*np.pi)/(real_tf-real_ti)
                #check if wm is very close to a quotient of integers:
                done = False
                best_k = -1
                val_min = 100
                l_ratio = [1,2,1/2,2/3,3/2,1/3,3,1/4,3/4,5/4, 4/3, 4/5, 7/4,
                           9/4, 4/7, 4/9]
                for k in l_ratio:
                    #if (abs(k*w_phi-wm))<10**-4:
                    #    w_final = 0
                    #    done = True
                    #    break
                    #else:
                    if abs(k*w_phi-wm)<val_min:
                        val_min = abs(k*w_phi-wm)
                        best_k = k
                if not done:
                    w_final = wm
                    #w_final = val_min
                l_wm.append(w_final)
            ''' OLD VERSION
            l_phase_theta_flat =  vect_Y[:,0]%(2*np.pi)
            l_phase_phi_flat =  vect_Y[:,1]%(2*np.pi)

            #cut traces so that when a boundary is crossed it creates a new plot
            ll_phase_theta=[]
            ll_phase_phi=[]
            l_phase_theta=[]
            l_phase_phi=[]
            for theta2, phi2, theta1, phi1 in zip(l_phase_theta_flat[1:],
                                                  l_phase_phi_flat[1:],
                                                  l_phase_theta_flat[:-1],
                                                  l_phase_phi_flat[:-1]):
                l_phase_theta.append(theta1)
                l_phase_phi.append(phi1)
                if abs(theta2-theta1)>np.pi or abs(phi2-phi1)>np.pi:
                    ll_phase_theta.append(np.array(l_phase_theta))
                    ll_phase_phi.append(np.array(l_phase_phi))
                    l_phase_theta=[]
                    l_phase_phi=[]
            ll_phase_theta.append(np.array(l_phase_theta))
            ll_phase_phi.append(np.array(l_phase_phi))

            #plot
            plt.figure(figsize=(5,5))
            for l_phase_theta, l_phase_phi in zip(ll_phase_theta,ll_phase_phi):
                plt.plot(l_phase_theta[:-1], l_phase_phi[:-1], lw = 2,
                         color = 'lightblue')
                #plt.quiver(l_phase_theta[:-1], l_phase_phi[:-1],
                            -(l_phase_theta[1:]-l_phase_theta[:-1]),
                            -(l_phase_phi[1:]-l_phase_phi[:-1]),
                            scale = 5)#, scale_units='xy', angles='xy', scale=1)


            plt.xlim(10**-2,2*np.pi-10**-2)
            plt.ylim(10**-2,2*np.pi-10**-2)
            plt.show()
            plt.close()
            '''

        if not self.fourier:
            self.F = real_coupling_function
            F_temp = np.hstack((self.F,self.F,self.F))
            F_temp = np.vstack((F_temp, F_temp, F_temp))
            dom_theta = np.linspace(-2*np.pi,4*np.pi, 3*self.F.shape[0],
                                    endpoint = False)
            dom_phi = np.linspace(-2*np.pi,4*np.pi, 3*self.F.shape[1],
                                    endpoint = False)
            self.F_func = interpolate.interp2d(dom_theta, dom_phi, F_temp.T,
                                               kind='cubic', bounds_error= True)
        self.T_phi = real_T_phi
        """
        for j in range(1,len(l_wm)-1):
            if (abs(l_wm[j]-l_wm[j-1]) + abs(l_wm[j]-l_wm[j+1])) > 0.2 :
                l_wm[j] = (l_wm[j+1]+l_wm[j-1])/2
        """

        #hide the grid
        sn.set_style("whitegrid",
                    {'grid.color': 'white',  'xtick.direction': 'out',
                    'xtick.major.size': 6.0, 'xtick.minor.size': 3.0,
                    'ytick.color': '.15', 'ytick.direction': 'out',
                    'ytick.major.size': 6.0, 'ytick.minor.size': 3.0})
        #matplotlib.style.use('ggplot')
        #plt.rcParams['axes.facecolor']='white'
        plt.figure(figsize=(5,5))
        plt.plot(speed_space, l_wm, '.', color = 'blue', markersize = 5)
        plt.plot([0,100],[0,100], '--', color = 'grey', alpha = 0.5)
        plt.plot([0,100],[0,200], '--', color = 'grey', alpha = 0.5)
        #plt.plot([0,100],[0,300], '--', color = 'grey', alpha = 0.5)
        plt.plot([0,200],[0,100], '--', color = 'grey', alpha = 0.5)
        plt.plot([0,300],[0,100], '--', color = 'grey', alpha = 0.5)
        #plt.plot([0,200],[0,300], '--', color = 'grey', alpha = 0.5)
        #plt.plot([0,300],[0,200], '--', color = 'grey', alpha = 0.5)
        plt.plot([0,300],[0,400], '--', color = 'grey', alpha = 0.5)

        plt.text(0.345,0.335,'1:1', color = 'grey')
        plt.text(0.175,0.335,'1:2', color = 'grey')
        plt.text(0.68,0.335,'2:1', color = 'grey')
        plt.text(0.735,0.24, '3:1', color = 'grey')
        plt.text(0.26,0.335, '3:4', color = 'grey')
        #plt.text(0.22,0.32,'2:3', color = 'grey')
        #plt.text(0.48,0.32,'3:2', color = 'grey')

        plt.xlabel(r'$\omega_{\phi}$')
        plt.ylabel(r'<$\omega_{\theta}$>')
        plt.xlim([0.1,0.8])
        plt.ylim([0.22,0.34])
        plt.tight_layout()
        #plt.show()
        plt.savefig(self.upfolder+'Results/DetSilico/devil_'+self.cell+'_'
                    +str(self.temperature)+'.pdf')
        plt.close()
        return l_wm

    def plot_arnold_tongues(self, NK, NW, tf = 400, random_init = False):
        """
        Plot and save the Arnold tongues.

        Parameters
        ----------
        NK : list or array
            Strength domain.
        NT : list or array
            Time domain.
        tf : int
            Simulation length.
        random init : bool
            Random initial condition.
        """
        ll_arnold = np.zeros((NW-1, NK))
        ll_arnold_test = np.zeros((NW-1, NK))
        ll_arnold_bis = np.zeros((NW, NK))

        #explore all possible coupling strength
        speed_space = np.linspace(2*np.pi/(2.5*24), 2*np.pi/(24/3), NW)
        #speed_space = np.linspace(2*np.pi/(1*24), 2*np.pi/(24/2.5), NW)
        period_space = list(reversed(2*np.pi/speed_space))
        coupling_space = np.linspace(0.,2., NK)
        #coupling_space = np.linspace(1.,2., NK)
        if random_init:
            n_rand = 5
        else:
            n_rand = 1

        l_arg = [(speed_space, K,  tf, True) for idx_rand in range(n_rand)\
                 for iK, K in enumerate(coupling_space)]
        n_cpu = 2
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGINT, original_sigint_handler)
        pool = Pool(n_cpu)
        try:
            #results = pool.map(self.plot_devil_staircase, l_arg)
            results = list(map(self.plot_devil_staircase, l_arg))
        except:
            print("BUG")
            pool.terminate()
        else:
            print("Normal termination")

        pool.close()
        pool.join()
        idx_mp = 0

        for idx_rand in range(n_rand):
            for iK, K in enumerate(coupling_space):
                l_wm = np.array(results[idx_mp])
                idx_mp+=1

                dspeed = np.absolute(l_wm[1:]-l_wm[:-1])
                dt2 = (speed_space[1]-speed_space[0])
                ll_arnold[:,iK] +=  list(reversed(dspeed/dt2))
                ll_arnold_bis[:,iK] += list(reversed(l_wm))

        test = np.reshape(results, (n_rand,
                          len(coupling_space),len(speed_space)))
        var_test = np.var(test,axis=0)


        ll_arnold[:,iK] = ll_arnold[:,iK]/n_rand
        ll_arnold_bis[:,iK] = ll_arnold_bis[:,iK]/n_rand

        plt.pcolormesh(period_space[:-1], coupling_space, ll_arnold.T,
                       cmap = 'binary', vmin = 0, vmax = 10**-2)
        plt.colorbar()
        plt.xlim([period_space[0], period_space[-2]])
        plt.xlabel(r"$T_{\phi}$")
        plt.ylabel("K")
        plt.savefig(self.upfolder+'Results/DetSilico/arnold_'+self.cell+'_'
                    +str(self.temperature)+'.pdf')
        plt.show()
        plt.close()

        plt.pcolormesh(period_space, coupling_space, ll_arnold_bis.T,
                       cmap = 'plasma')#, vmin = 0, vmax = 0.2)
        plt.colorbar()
        plt.xlabel(r"$T_{\phi}$")
        plt.ylabel("K")
        plt.savefig(self.upfolder+'Results/DetSilico/arnold_bis_'+self.cell+'_'
                    +str(self.temperature)+'.pdf')
        plt.show()
        plt.close()

        plt.pcolormesh(period_space, coupling_space, var_test, cmap = 'binary')
        plt.colorbar()
        plt.xlabel(r"$T_{\phi}$")
        plt.ylabel("K")
        plt.savefig(self.upfolder+'Results/DetSilico/arnold_bis2_'+self.cell+'_'
                    +str(self.temperature)+'.pdf')
        plt.show()
        plt.close()


        pickle.dump(ll_arnold.T, open(self.upfolder+"Results/DetSilico/arnold_"
                                      +self.cell+'_'+str(self.temperature)+".p",
                                       "wb"))
        pickle.dump(ll_arnold_bis.T, open( self.upfolder
                                      +"Results/DetSilico/arnold_bis_"
                                      +self.cell+'_'+str(self.temperature)+".p",
                                       "wb"))
        pickle.dump(var_test, open( self.upfolder
                                    +"Results/DetSilico/arnold_bis2_"+self.cell
                                    +'_'+str(self.temperature)+".p", "wb"))
        return period_space, speed_space, coupling_space

    def refine_arnold_tongues(self, period_space, speed_space, coupling_space):
        """
        Load, refine and plot Arnold tongues from existing plot.
        """
        ll_arnold =  pickle.load(open(self.upfolder+"Results/DetSilico/arnold_"
                                      +self.cell+'_'+str(self.temperature)+".p",
                                      "rb" ) )
        var_test = pickle.load(open (self.upfolder
                                      +"Results/DetSilico/arnold_bis2_"
                                      +self.cell+'_'+str(self.temperature)+".p",
                                       "rb" ) )

        #ll_arnold = ll_arnold[-50:,-50:]

        #data1= griddata( (period_space[-50:], coupling_space[-50:]),
                            #ll_arnold, (xnew, ynew))
        #f = interp2d(period_space[:-1], coupling_space, ll_arnold,
            #kind='linear')
        #f = RectBivariateSpline(period_space[-50:], coupling_space[-50:],
                            #ll_arnold, s= 10)


        #Xn, Yn = np.meshgrid(period_space[-50:], coupling_space[-50:])
        #f = Rbf(Xn, Yn, ll_arnold)
        #xnew = np.linspace(period_space[0], period_space[-2], 1000)
        #ynew = np.linspace(coupling_space[0], coupling_space[-1], 1000)
        #Xn, Yn = np.meshgrid(xnew, ynew)
        #data1 = f(xnew,ynew)

        #plt.pcolormesh(xnew, ynew, data1, cmap = 'binary', vmin = 0, vmax = 5,
                        #shading='gouraud')

        plt.pcolormesh(period_space[:-1], coupling_space, ll_arnold,
                        cmap = 'binary', vmin = 0, vmax = 2, shading='gouraud')
        #plt.imshow(ll_arnold, cmap = 'binary', vmin = 0, vmax =5 ,
                    #origin='lower', extent=[period_space[0],period_space[-2],
                    #coupling_space[0],coupling_space[-1]], aspect='auto')
        plt.colorbar()
        plt.xlim([period_space[0], period_space[-2]])
        plt.xlabel(r"$T_{\phi}:T_{\theta}$")
        plt.ylabel("K")
        locs, labels = plt.xticks()
        plt.xticks([12,2/3*24, 24, 24*4/3, 48],
                   ['2:1', '3:2','1:1', '3:4', '1:2'])
        plt.savefig(self.upfolder+'Results/DetSilico/arnold_'+self.cell+'_'
                    +str(self.temperature)+'.pdf')
        plt.show()
        plt.close()




        plt.figure(figsize=(5,5))
        plt.pcolormesh(period_space, coupling_space, var_test[:,::-1],
                       cmap = 'binary', vmin = 0, vmax = 10**-8,
                       shading='gouraud')
        #plt.imshow(ll_arnold_bis.T, cmap = 'plasma', origin='lower',
                    #extent=[speed_space[0],speed_space[-2],coupling_space[0],
                    #coupling_space[-1]], aspect='auto')
        plt.xlim([period_space[0], period_space[-1]])
        #plt.colorbar()
        plt.xlabel(r"$T_{\phi}:T_{\theta}$")
        plt.ylabel("K")
        locs, labels = plt.xticks()
        plt.xticks([12,2/3*24, 24, 24*4/3, 48],
                   ['2:1', '3:2','1:1', '3:4', '1:2'])
        plt.tight_layout()
        plt.savefig(self.upfolder+'Results/DetSilico/arnold_bis2_'+self.cell+'_'
                    +str(self.temperature)+'.pdf')
        plt.show()
        plt.close()



    def plot_moving_attractor(self, save = False):
        """
        Plot attractor for evolving cell-cycle period.

        Parameters
        ----------
        save : bool
            Save plot (True) or not (False).
        """
        ll_theta= []
        ll_phi = []
        space_period = list(range(19,28,1))
        for T_phi in space_period:
            l_theta, l_phi = self.plot_trajectory(ti = 2500, tf = 3000,
                                                  rand = True, save = False,
                                                  K = 1, T_phi  = T_phi,
                                                  full_simulation = False)
            ll_theta.append(l_theta)
            ll_phi.append(l_phi)

        #hide the grid
        sn.set_style("whitegrid",
                    {'grid.color': 'white',  'xtick.direction': 'out',
                    'xtick.major.size': 6.0, 'xtick.minor.size': 3.0,
                    'ytick.color': '.15', 'ytick.direction': 'out',
                    'ytick.major.size': 6.0, 'ytick.minor.size': 3.0})

        current_palette = sn.color_palette("Greens", len(space_period)+2)
        fig = plt.figure(figsize=(5,5))
        plt.plot([0,1],[0.00,0.00], '--', color = 'blue', alpha = 0.5)
        plt.text(x = 0.1, y = 0.01, s=r'$\phi = 0$')

        plt.plot([0,1],[0.33,0.33], '--', color = 'red', alpha = 0.5)
        plt.text(x = 0.4, y = 0.34, s=r'$\phi = 2\pi/3$')

        plt.plot([0,1],[0.66,0.66], '--', color = 'green', alpha = 0.5)
        plt.text(x = 0.8, y = 0.67, s=r'$\phi = 4\pi/3$')
        for idx, (l_theta, l_phi) in enumerate(zip(ll_theta, ll_phi)):
            flag = False
            for l_phase_theta, l_phase_phi in zip(l_theta,l_phi):
                if not flag:
                    plt.plot(np.array(l_phase_theta)/(2*np.pi),
                            np.array(l_phase_phi)/(2*np.pi),
                            color = current_palette[idx+2] ,
                            label = r'$T_{\phi}=$' +str(space_period[idx])+'h')
                    flag = True
                else:
                    plt.plot(np.array(l_phase_theta)/(2*np.pi),
                    np.array(l_phase_phi)/(2*np.pi),
                    color = current_palette[idx+2] )

        plt.xlim(0,1)
        plt.ylim(-0.01,1)

        #plt.axhspan(0, 0.34, alpha=0.1, color='red')
        #plt.axhspan(0.34, 0.65, alpha=0.1, color='green')
        #plt.axhspan(0.65, 1, alpha=0.1, color='blue')

        #plt.text(1.2,0.7,r'$\phi = 0$')
        #plt.text(1.2,0.4,r'$\phi = 4\pi/3$')
        #plt.text(1.2,0.1,r'$\phi = 2\pi/3$')

        plt.xlabel(r'Circadian phase $\theta$')
        plt.ylabel(r'Cell-cycle phase $\phi$')

        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
        #plt.tight_layout()

        if not save:
            plt.show()
        else:
            plt.savefig(self.upfolder+'Results/DetSilico/shift_attractor_'
                        +self.cell+'_'
                        +str(self.temperature)+'.pdf', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    #load parameters
    with open('../Parameters/Real/opt_parameters_div_None_NIH3T3.p', 'rb') as f:
        l_parameters = [dt, sigma_em_circadian, W, pi,
        N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
        N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
        N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,
        gamma_amplitude_theta, l_boundaries_amplitude_theta,
        N_background_theta, mu_background_theta, std_background_theta,
        gamma_background_theta, l_boundaries_background_theta,
        F] = pickle.load(f)
    detSim = DetSim(l_parameters, 'NIH3T3', None)

    #detSim.plot_signal(tf = 200, full_simulation = True, save = False)
    #detSim.plot_trajectory(ti = 0, tf = 80, rand = True, save = True,
    #                       K = 1, T_phi  = None, full_simulation = False)
    #detSim.plot_vectorfield(save = True)
    #detSim.plot_vectorfield_bis(T_phi = 22, save = True)
    #detSim.anim_vectorfield()
    #detSim.plot_sync_tongues(NK = 4, NT = 10, N = 4, tf = 100)
    #speed_space = np.linspace(2*np.pi/(3*24), 2*np.pi/(24/3), 50)
    #l_arg = (speed_space, 3, 400, True)
    #detSim.plot_devil_staircase(l_arg)
    #t = detSim.plot_arnold_tongues(NK = 4, NW = 50, tf = 200)
    #(period_space, speed_space, coupling_space) = t
    #detSim.refine_arnold_tongues(period_space, speed_space, coupling_space)
    #detSim.plot_moving_attractor(save = True)
