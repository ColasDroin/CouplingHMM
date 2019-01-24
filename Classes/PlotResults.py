# -*- coding: utf-8 -*-
### Module import
import os
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import seaborn as sn
#nice plotting style
sn.set_style("whitegrid", { 'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

###Local modules
from StateVar import StateVar

class PlotResults():
    """
    This class is used to plot with different representations the inferred
    distributions of the hidden variables, along with the experimental points.
    """
    def __init__(self, gamma, l_var, model, signal, waveform = None,
                 logP = None, temperature = 37, cell = 'NIH3T3'):
        """
        Constructor of PlotResults.

        Parameters
        ----------
        gamma : ndarray
            matrix of probability of being in a given state given all the
                observations.
        l_var : list
            List of hidden variables.
        model : function
            Model used for the circadian signal.
        signal : list
            Observed signal.
        waveform : list
            Waveform
        logP : list
            List of log-probabilities for the input traces.
        temperature : int
            Temperature condition.
        cell : string
            Cell type condition.
        """
        self.gamma = gamma
        self.l_var = l_var
        self.model = model
        self.signal = signal
        if waveform is not None:
            self.waveform = interpolate.interp1d(np.linspace(0,2*np.pi,
                                                            len(waveform),
                                                            endpoint = True),
                                                 waveform)
        else:
            self.waveform = waveform
        if logP is None:
            self.logP = ""
        else:
            self.logP = logP
        self.temperature = temperature
        self.cell = cell



    def plotEverythingEsperance(self, save=False, index_trace=0, folder = None,
                                tag = '', l_obs_phi = None):
        """
        Plot the expected value of the hidden variables.

        Parameters
        ----------
        save : bool
            Save or not the figure.
        index_trace : integer
            Index of the trace to plot.
        folder : string
            Folder in which the plot must be stored.
        tag : string
            Tag to add to the name of the saved file for the plot.
        l_obs_phi : list
            List of observations for the cell-cycle.

        Returns
        -------
        The expected value of the signal and the different hidden variables.
        """

        tspan = np.linspace(0,self.gamma.shape[0]/2,self.gamma.shape[0])
        Y_model=[]
        Y_var=[]
        for t, gamma_t in enumerate(self.gamma):
            Y_t_var=[]
            Y_t_var_rectified=[]
            for idx_var, var in enumerate(self.l_var):
                l_tmp = [i for i in range(len(self.l_var)) if i!=idx_var]
                p_var = np.sum(gamma_t,axis=tuple(l_tmp))
                Y_t_var.append(np.sum(np.multiply(p_var, var.domain)))
                if var.name_variable!="Theta":
                    Y_t_var_rectified.append(np.sum(np.multiply(p_var,
                                                                   var.domain)))
                else:
                    v_tmp = np.multiply(p_var,np.exp(1j*np.array(var.domain)))
                    Y_t_var_rectified.append(np.angle(np.sum(v_tmp))%(2*np.pi))

            Y_model.append( self.model(Y_t_var_rectified,self.waveform ) )
            #Y_var.append(Y_t_var)
            Y_var.append(Y_t_var_rectified)
        Y_var = np.array(Y_var)
        #Then plot


        plt.figure(figsize=(5,5))
        plt.plot(tspan, self.signal, '.', label=r'$O_t$')
        plt.plot(tspan, Y_model, '-', label=r'$E[s_t]$')
        #plt.title("logP: "+ str(self.logP))
        plt.xlabel('Time (h)')
        plt.ylabel(r'RevErb-$\alpha$-YFP fluorescence (a.u.)')
        if len(Y_var[0])>=1:
            #plt.plot(tspan,Y_var[:,0]/(2*np.pi), '--', label =  r'$\phi$')
            #if len(Y_var[0])==2:
                #plt.plot(tspan,np.exp(Y_var[:,1]), '--',label = "Amplitude")
            pass

        if len(Y_var[0])==3:
            plt.plot(tspan,Y_var[:,0]/(2*np.pi),'--', label = r'$E[\theta_t]$')
            plt.plot(tspan,np.exp(Y_var[:,1]), '--',label = r"$E[A_t]$")
            plt.plot(tspan,Y_var[:,2], '--',label = r"$E[B_t]$")

        if l_obs_phi is not None:
            plt.plot(tspan,np.array(l_obs_phi)/(2*np.pi), '--',
                     label = r'$\phi_t$')
        plt.legend(bbox_to_anchor=(0.,1.02,1,0.2), loc="lower left",
                   mode = 'expand', ncol = 6)
        if not save:
            #plt.show()
            pass

        else:
            if folder is None:
                plt.savefig('Results/Fits/Esperance_'+tag+'_'+str(index_trace)
                            +'_'+str(self.temperature)+"_"+self.cell+'.pdf')
            else:
                plt.savefig(folder+'/Results/Fits/Esperance_'+tag+'_'
                            +str(index_trace)+'_'+str(self.temperature)
                            +"_"+self.cell+'.pdf')
        plt.close()
        return Y_model, Y_var[:,0]/(2*np.pi), np.exp(Y_var[:,1]), Y_var[:,2]

    def plotEverythingMostLikely(self, save = False, index_trace = 0,
                                                                folder = None):
        """
        Plot the most likely state of the hidden variables for a given trace.

        Parameters
        ----------
        save : bool
            Save or not the figure.
        index_trace : integer
            Index of the trace to plot.
        folder : string
            Folder in which the plot must be stored.

        Returns
        -------
        The plotted figure.
        """


        tspan = np.linspace(0,self.gamma.shape[0]/2,self.gamma.shape[0])
        Y_model = []
        Y_var = []
        for gamma_t in self.gamma:
            best_state_idx = gamma_t.argmax()
            best_state_idx = np.unravel_index(best_state_idx, gamma_t.shape)
            Y_var.append([var.domain[i] for var,i \
                                            in zip(self.l_var,best_state_idx)])
            Y_model.append(self.model([var.domain[i] for var,i in\
                                zip(self.l_var,best_state_idx)], self.waveform))
        Y_var = np.array(Y_var)
        #Then plot
        fig = plt.figure()
        plt.title("logP: "+ str(self.logP))
        plt.plot(tspan, Y_model, '-',label='Pmax[model]')
        plt.plot(tspan, self.signal, '.',label='Signal')
        plt.xlabel('t')
        if len(Y_var[0])>=1:
            plt.plot(tspan,Y_var[:,0]/(2*np.pi), '--', label =  r'$\phi$')
            if len(Y_var[0])==2:
                plt.plot(tspan,np.exp(Y_var[:,1]), '--',label = "Amplitude")

        if len(Y_var[0])==3:
            plt.plot(tspan,Y_var[:,0]/(2*np.pi),'--', label = r'$\theta$')
            plt.plot(tspan,np.exp(Y_var[:,1]), '--',label = "Amplitude")
            plt.plot(tspan,Y_var[:,2], '--',label = "Background")
        #plt.legend()
        if not save:
            plt.show()
        else:
            if folder is None:
                plt.savefig('Results/Fits/MostLikely_'+str(index_trace)+'_'
                            +str(self.temperature)+"_"+self.cell+'.pdf')
            else:
                plt.savefig(folder+'/Results/Fits/MostLikely_'+str(index_trace)
                            +'_'+str(self.temperature)+"_"+self.cell+'.pdf')
        plt.close()
        return fig

    def plot3D(self, save = False, index_trace = 0):
        """
        Plot the phase distribution in 3D (2nd axis is the time).

        Parameters
        ----------
        save : bool
            Save or not the figure.
        index_trace : integer
            Index of the trace to plot.
        """
        current_palette = sn.color_palette()

        tspan = np.linspace(0,self.gamma.shape[0]/2,self.gamma.shape[0]+1)
        l_P_theta=[ [0]*self.l_var[0].nb_substates ]
        for t, gamma_t in enumerate(self.gamma):
            l_P_theta.append(np.sum(gamma_t,axis=(1,2)))


        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X,Y = np.meshgrid(self.l_var[0].domain, tspan )
        surf = ax.plot_surface(X,Y , np.array(l_P_theta), cstride = 50,
                                rstride = 8, shade = False,
                                color = current_palette[1], alpha = 0.8)
        plt.xlim([0,2*np.pi])
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.tick_params(axis='x', labelbottom= 'off', labeltop = 'off',
                                            labelleft = 'off', labelright='off')
        plt.tick_params(axis='y', labelbottom= 'off', labeltop = 'off',
                                            labelleft = 'off', labelright='off')
        plt.tick_params(axis='z', labelbottom= 'off', labeltop = 'off',
                                            labelleft = 'off', labelright='off')
        #ax.set_axis_off()
        ax.view_init(40, 90)
        plt.xlabel(r"$\theta$")
        plt.ylabel("Time")
        ax.set_zlabel(r'$P(\theta)$')
        #plt.tight_layout()
        #plt.legend()
        if not save:
            plt.show()
        else:
            plt.savefig('Results/Fits/3D_'+str(index_trace)+'_'
                        +str(self.temperature)+"_"+self.cell+'.pdf')
        plt.close()


    def plotEsperancePhaseSpace(self, l_phi, save = False, index_trace = 0,
                                    tag = '', folder = None, attractor = None):
        """
        Plot the expected value of the phase on the phase-space.

        Parameters
        ----------
        l_phi : list
            List of cell-cycle values.
        save : bool
            Save or not the figure.
        index_trace : integer
            Index of the trace to plot.
        tag : string
            Tag to add to the name of the saved file for the plot.
        folder : string
            Folder in which the plot must be stored.
        attractor : tuple
            Coordinates of the simulated attractor. If None, not plotted.
        """
        #copy paste from plotEverythingEsperance
        tspan = np.linspace(0,self.gamma.shape[0]/2,self.gamma.shape[0])
        Y_model=[]
        Y_var=[]
        for t, gamma_t in enumerate(self.gamma):
            Y_t_var=[]
            Y_t_var_rectified=[]
            for idx_var, var in enumerate(self.l_var):
                p_var = np.sum(gamma_t,axis=tuple([i for i \
                                      in range(len(self.l_var)) if i!=idx_var]))
                Y_t_var.append(np.sum(np.multiply(p_var, var.domain)))
                if var.name_variable!="Theta":
                    tmp = np.sum(np.multiply(p_var, var.domain))
                    Y_t_var_rectified.append(tmp)
                else:
                    tmp = np.multiply(p_var,np.exp(1j*np.array(var.domain)))
                    Y_t_var_rectified.append(np.angle(np.sum(tmp))%(2*np.pi))

            Y_model.append( self.model(Y_t_var_rectified,self.waveform ) )
            #Y_var.append(Y_t_var)
            Y_var.append(Y_t_var_rectified)
        Y_var = np.array(Y_var)
        #end of copy paste
        l_theta = Y_var[:,0]

        """ interpolate signal """
        from scipy.interpolate import interp1d
        l_theta = np.unwrap(l_theta)
        l_phi = np.unwrap(l_phi)
        l_t = np.linspace(0,1,len(l_theta))
        l_theta = interp1d(l_t, l_theta, kind = 'cubic')(np.linspace(0,1,1000))
        l_phi = interp1d(l_t, l_phi, kind = 'cubic')(np.linspace(0,1,1000))
        #smooth phi
        l_phi = signal.savgol_filter(l_phi, 301, 3)
        l_theta = l_theta%(2*np.pi)
        l_phi = l_phi%(2*np.pi)

        """"""""" REMOVE VERTICAL LINES AT BOUNDARIES  """""""""
        abs_d_data_x = np.abs(np.diff(l_theta))
        mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()\
                                                +2*abs_d_data_x.std(), [False]])
        masked_l_theta = np.array([x if not m else np.nan for x,m \
                                                    in zip(l_theta, mask_x)  ])

        abs_d_data_x = np.abs(np.diff(l_phi))
        mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()\
                                                +2*abs_d_data_x.std(), [False]])
        masked_l_phi = np.array([x if not m else np.nan for x,m \
                                                        in zip(l_phi, mask_x)])

        """"""""" COMPUTE IDENDITIY  """""""""

        #plot identity
        x_domain = np.linspace(0,2*np.pi,100)
        f =  (1.15 + np.linspace(0,2*np.pi,100))%(2*np.pi)
        abs_d_data_x = np.abs(np.diff(f))
        mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()\
                                                +3*abs_d_data_x.std(), [False]])
        masked_identity = np.array([x if not m else np.nan \
                                                    for x,m in zip(f, mask_x) ])
        """"""""" PLOT  """""""""
        plt.figure(figsize=(5,5))
        plt.plot(masked_l_theta/(2*np.pi), masked_l_phi/(2*np.pi),
                                                    color = 'lightblue', lw = 4)
        #plt.plot(x_domain/(2*np.pi),masked_identity/(2*np.pi) , '--'   )
        if attractor is not None:
            for l_phase_theta, l_phase_phi in zip(attractor[0],attractor[1]):
                plt.plot(np.array(l_phase_theta)/(2*np.pi),
                        np.array(l_phase_phi)/(2*np.pi), color = 'grey',
                        lw = 0.5)

        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel(r'Circadian phase $\theta$')
        plt.ylabel(r'Cell-cycle phase $\phi$')
        plt.tight_layout()
        #plt.legend()
        if not save:
            plt.show()
        else:
            if folder is None:
                plt.savefig('Results/PhaseSpace/Trace_'+tag+'_'+str(index_trace)
                                +'_'+str(self.temperature)+"_"+self.cell+'.pdf')
            else:
                plt.savefig('../Results/PhaseSpace/Trace_'+tag+'_'
                                +str(index_trace)+'_'+str(self.temperature)
                                +"_"+self.cell+'.pdf')
        plt.close()
