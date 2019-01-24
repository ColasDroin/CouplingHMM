# -*- coding: utf-8 -*-
### Module import
import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle
import seaborn as sn
#nice plotting style
sn.set_style("whitegrid", { 'xtick.direction': 'out', 'xtick.major.size': 6.0,
            'xtick.minor.size': 3.0, 'ytick.color': '.15',
            'ytick.direction': 'out', 'ytick.major.size': 6.0,
            'ytick.minor.size': 3.0})

class PlotStochasticSpeedSpace():
    """
    This class was initially used to compute the coupling function from the
    derivatives of the phase. It is principally used to compute the distribution
    of phase speeds at given couple of cell-cycle and circadian phases.
    """
    def __init__(self, t_lll_coordinates, l_var, dt, w_phi, cell, temperature,
                 cmap = None):
        """
        Constructor of PlotStochasticSpeedSpace.

        Parameters
        ----------
        t_lll_coordinates : tuple
            Tuple (for circadian and cell-cycle oscillators) of list (trace
            index) of list (time) of list (hidden variables).
        l_var : list
            List of hidden variables.
        dt : float
            Time resolution.
        w_phi : float
            Circadian frequency in radians.
        cell : string
            Cell type.
        temperature : integer
            Temperature condition.
        cmap : string or colormap
            Colormap to use for the plots.
        """

        self.t_lll_coordinates = t_lll_coordinates
        self.l_var = l_var
        #if not os.path.exists("Results/StochasticSilico"):
        #    os.makedirs("Results/StochasticSilico")
        self.domain_phi = l_var[0].codomain
        self.dt = dt
        self.w_theta = l_var[0].l_parameters_f_trans[0]
        self.w_phi = w_phi
        if cmap is not None:
            self.cmap = cmap
        else:
            self.cmap = 'bwr'
        self.cell = cell
        self.temperature = temperature

    def getll_vFromCoordinates(self ):
        """
        Compute and returns distribution of phase speeds depending on phase
        coordinate.

        Returns
        -------
        A list of list (for first and second coordinate) of list (speed
        distribution).
        """
        t_Y_var = [ [], [] ]
        Y_var = []
        for idx_hmm, lll_coordinates in enumerate(self.t_lll_coordinates):
            for ll_coordinates in lll_coordinates:
                for t, l_coordinates in enumerate(ll_coordinates):
                    temp=[]
                    if idx_hmm==0:
                        #first circadian variables
                        for var, coor in zip(self.l_var, l_coordinates):
                            i =  int(round( (coor - var.l_boundaries[0]) \
                                 / (var.l_boundaries[1]-var.l_boundaries[0]) \
                                 * var.nb_substates)%var.nb_substates)

                            temp.append( (i, var.domain[i]) )
                    else:
                        coor = l_coordinates[0]
                        i =  int(round( (coor )/ (2*np.pi) \
                                 * len(self.domain_phi))%len(self.domain_phi))
                        temp.append( (i, self.domain_phi[i]) )

                    Y_var = np.array(temp)
                    t_Y_var[idx_hmm].append(Y_var)

                Y_interval = np.copy(Y_var)
                Y_interval.fill(np.nan)
                t_Y_var[idx_hmm].append(Y_interval)

            t_Y_var[idx_hmm] = np.array(t_Y_var[idx_hmm])
        index_theta = [i for  i,var in enumerate(self.l_var) \
                                            if var.name_variable == "Theta"][0]

        ll_v=[ [ [] for j in  range(len(self.domain_phi))] \
                        for i in range(self.l_var[index_theta].nb_substates)]
        z_tot = zip(t_Y_var[0][:-1,index_theta,0],
                    t_Y_var[0][:-1,index_theta,1],
                    t_Y_var[0][1:,index_theta,0],
                    t_Y_var[0][1:,index_theta,1],
                    t_Y_var[1][:-1,0,0], t_Y_var[1][:-1,0,1],
                     t_Y_var[1][1:,0,0],  t_Y_var[1][1:,0,1])
        for idx11, ph11, idx12, ph12, idx21, ph21, idx22, ph22 in z_tot:

            #print(idx11, ph11, idx12, ph12, idx21, ph21, idx22, ph22)

            #remove speeds computed from phases from 2 different signals
            if (np.isnan(ph11) or np.isnan(ph12) or np.isnan(ph21) or
                np.isnan(ph22)):
                #print(ph11,ph12,ph21,ph22)
                continue

            #correct phases
            ph11c = ph11
            ph12c = ph12
            ph21c = ph21
            ph22c = ph22
            if ph12-ph11>np.pi:
                ph11c = ph11 + 2*np.pi
            if ph11-ph12>np.pi:
                ph12c = ph12 + 2*np.pi
            if ph22-ph21>np.pi:
                ph21c = ph21 + 2*np.pi
            if ph21-ph22>np.pi:
                ph22c = ph22 + 2*np.pi
            if abs((ph12c-ph11c)/self.dt)>2:
                #print(ph12, ph11)
                #print(ph12c, ph11c)
                pass
            ll_v[int(idx11)][int(idx21)].append(((ph12c-ph11c)/self.dt,
                                                        (ph22c-ph21c)/self.dt))
        return ll_v

    def getPhaseSpace(self):
        """
        Compute and returns the average cell-cycle and circadian speeds on the
        phase-space, plus a matrix of density.

        Returns
        -------
        Three arrays of the dimension of the phase-space : the average circadian
        speed, the average cell-cycle speed, and the number of counts for each
        state of the phase-space.
        """
        ll_v = self.getll_vFromCoordinates()
        #average all speeds and build heatmap
        space_theta = np.zeros( (len(ll_v),len(ll_v[0]) ))
        space_phi = np.zeros( (len(ll_v),len(ll_v[0]) ))
        space_count = np.zeros( (len(ll_v),len(ll_v[0]) ))
        for idx_theta, l_idx_phi in enumerate(ll_v):
            for idx_phi, l_speed in enumerate(l_idx_phi):
                avg_speed_theta = 0
                avg_speed_phi = 0
                norm = 0
                for v_theta, v_phi in l_speed:
                    #print(idx_theta, idx_phi, v_theta, v_phi)
                    avg_speed_theta += v_theta-self.w_theta
                    avg_speed_phi += v_phi-self.w_phi
                    norm+=1
                if avg_speed_theta!=0:
                    avg_speed_theta  = avg_speed_theta/norm
                if avg_speed_phi!=0:
                    avg_speed_phi  = avg_speed_phi/norm
                space_theta[idx_theta, idx_phi] = avg_speed_theta
                space_phi[idx_theta, idx_phi] = avg_speed_phi
                space_count[idx_theta, idx_phi] = norm
        return space_theta, space_phi, space_count

    def plotPhaseSpace(self, save_plot=False):
        """
        Plot the average cell-cycle and circadian speeds on the
        phase-space, plus a matrix of density.

        Returns
        -------
        Three arrays of the dimension of the phase-space : the average circadian
        speed, the average cell-cycle speed, and the number of counts for each
        state of the phase-space.
        """

        space_theta, space_phi, space_count = self.getPhaseSpace()

        #plt.pcolor(np.array(space_theta).T,cmap='bwr', vmin=-0.6, vmax=0.6)
        plt.imshow(np.array(space_theta.T), cmap= self.cmap, vmin=-0.3,
                   vmax=0.3, origin='lower', interpolation = "spline16" )
        plt.colorbar()
        plt.xlabel("theta")
        plt.ylabel("phi")
        if save_plot:
            plt.savefig('Results/StochasticSilico/ThetaGenerated_'+self.cell
                        +'_'+str(self.temperature)+'.pdf')
            plt.close()
        else:
            plt.show()
        #plt.pcolor(np.array(space_phi).T, cmap='bwr', vmin=-0.6, vmax=0.6)
        plt.imshow(np.array(space_phi.T), cmap=self.cmap, vmin=-0.3, vmax=0.3,
                   origin='lower', interpolation = "spline16" )
        plt.colorbar()
        plt.xlabel("theta")
        plt.ylabel("phi")
        if save_plot:
            plt.savefig('Results/StochasticSilico/PhiGenerated_'+self.cell
                        +'_'+str(self.temperature)+'.pdf')
            plt.close()
        else:
            plt.show()
        #plt.pcolor(np.array(space_count).T, cmap='jet', vmin=0, vmax=70)
        plt.imshow(np.array(space_count.T), cmap='bwr',  origin='lower')
        plt.colorbar()
        plt.xlabel("theta")
        plt.ylabel("phi")
        if save_plot:
            plt.savefig('Results/StochasticSilico/CountGenerated_'+self.cell
                        +'_'+str(self.temperature)+'.pdf')
            plt.close()
        else:
            plt.show()

        plt.close()
        return space_theta, space_phi, space_count
