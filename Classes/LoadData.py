# -*- coding: utf-8 -*-
### Module import
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import sys
import os

###Local modules
#sys.path.insert(0, os.path.realpath('./Classes'))
from TraceInformation import TraceInformation

class LoadData():
    """
    This class is used to load, convert and normalize the experimental traces.
    """
    def __init__(self, path, nb_traces = 10000000, temperature = None,
                 division = True, several_cell_cycles = False,
                 remove_odd_traces = False, several_circadian_cycles = False):
        """
        Constructor of LoadData.

        Parameters
        ----------
        path : string
            Path from which the data must be loaded.
        nb_traces : integer
            Number of traces to be loaded (if superior to the number of traces
            present in the data, is set to this number).
        temperature : integer
            Temperature condition (34, 37 or 40).
        division : bool
            Set if traces come from dividing cells (True) or not (False).
        several_cell_cycles : bool
            Set if traces must contain several cell cycles divisions (True) or
            not (False).
        remove_odd_traces : bool
            Set if odd traces must be filtered out (True) or not (False).
        several_circadian_cycles : bool
            Set if traces must contain several circadian cycles (True) or
            not (False).
        """
        self.path = path
        self.nb_traces = nb_traces
        self.temperature = temperature #34, 37 or 40
        self.division = division #either no division (False) either 2 or more
        self.cc = several_cell_cycles
        self.remove = remove_odd_traces
        self.cc_bis = several_circadian_cycles

    def load_object(self, t_lists, trace_object):
        """
        Load empty lists with data coming from a trace_object.

        Parameters
        ----------
        t_lists : tuple
            Tuple of empty lists to be loaded with (cf. function filter data).
        trace_object : TraceInformation
            Object containing the data.
        """
        (ll_area, ll_signal, ll_peak, ll_cell_cycle_start,
                                                    ll_mitosis_start) = t_lists
        ll_area.append(trace_object.normalized_area)
        ll_signal.append(trace_object.normalized_circa)
        ll_peak.append(trace_object.peaks)
        ll_cell_cycle_start.append(trace_object.cell_cycle_start)
        ll_mitosis_start.append(trace_object.mitosis_start)

    def filter_data(self):
        """
        Load and filter data according to several set of conditions passed to
        the constructor.

        Returns
        -------
        Several lists of lists (1st dim = traces, 2nd dim = time) :
        ll_area : cell-cycle area
        ll_signal : fluorescence circadian signals
        ll_peak : annotated circadian peaks
        ll_cell_cycle_start : first point after division
        ll_mitosis_start : beginning of division
        """
        ll_area = []
        ll_signal = []
        ll_peak = []
        ll_cell_cycle_start = []
        ll_mitosis_start = []
        t_lists = (ll_area, ll_signal, ll_peak, ll_cell_cycle_start,
                    ll_mitosis_start)
        """ LOAD DATA """
        for trace_object in pickle.load(  open( self.path , "rb" ) ):
            n_div = np.sum(trace_object.cell_cycle_start)
            if self.temperature is None:
                if self.division and n_div>=2:
                    self.load_object(t_lists, trace_object)
                elif not self.division and n_div==0:
                    self.load_object(t_lists, trace_object)
            else:
                if trace_object.temp == self.temperature:
                    if self.division and n_div>=2:
                        self.load_object(t_lists, trace_object)
                    elif not self.division and n_div==0:
                        self.load_object(t_lists, trace_object)
        return t_lists

    def process_data(self, t_lists, load_annotation = False):
        """
        Process the traces (crop, annotate, filter, etc.).

        Parameters
        ----------
        t_lists : tuple
        A tuple of lists (1st dim = traces, 2nd dim = time) :
            ll_area : cell-cycle area
            ll_signal : fluorescence circadian signals
            ll_peak : annotated circadian peaks
            ll_cell_cycle_start : first point after division
            ll_mitosis_start : beginning of division
        load_annotation : bool
            Load or not supplementary annotations.

        Returns
        -------
        Sevral lists corresponding to the cell-cycle signal,
        the circadian signal and possible annotations.
        """
        (ll_area, ll_signal, ll_peak, ll_cell_cycle_start,
                                                    ll_mitosis_start) = t_lists

        """ NaN FOR CIRCADIAN SIGNAL """
        ll_nan_circadian_factor = []
        if self.division:
            for l_mitosis_start, l_cell_start, l_peak in zip(ll_mitosis_start,
                                                            ll_cell_cycle_start,
                                                            ll_peak):
                l_temp = [False]*len(l_mitosis_start)
                NaN = False
                for ind, (m,c) in enumerate(zip(l_mitosis_start, l_cell_start)):
                    if m==1:
                        NaN = True
                    if c==1:
                        NaN = False
                    if NaN:
                        try:
                            l_temp[ind-1] = True #TO BE REMOVED POTENTIALLY
                        except:
                            pass
                        try:
                            l_temp[ind+1] = True
                        except:
                            pass
                        try:
                            l_temp[ind+2] = True
                        except:
                            pass

                        l_temp[ind] = True
                ll_nan_circadian_factor.append(  l_temp   )


        """ CROP TRACES """
        ll_idx_cell_cycle_start =[]
        if self.division:
            dom = zip(enumerate(ll_mitosis_start), ll_cell_cycle_start)
            for (idx, l_mitosis_start), l_cell_cycle_start in dom :
                l_idx_mitosis_start = [idx for idx, i in \
                                            enumerate(l_mitosis_start) if i==1]
                l_idx_cell_cycle_start = [idx for idx, i in\
                                         enumerate(l_cell_cycle_start) if i==1]
                ll_idx_cell_cycle_start.append(l_idx_cell_cycle_start)


        """ COMPUTE CIRCADIAN AND CELL-CYCLE PERIODs """
        l_T = []
        for l_peak in ll_peak:
            l_idx_peak = [idx for idx, i in enumerate(l_peak) if i==1]
            for t_peak_1, t_peak_2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
                l_T.append( (t_peak_2-t_peak_1)/2)
        T_theta = np.mean(l_T)


        l_T2 = []
        for l_idx_cell_cycle_start in ll_idx_cell_cycle_start:
            for t_start_1, t_start_2 in zip(l_idx_cell_cycle_start[:-1],
                                            l_idx_cell_cycle_start[1:]):
                l_T2.append( (t_start_2-t_start_1)/2)
        if len(l_T2)>0:
            T_phi = np.mean(l_T2)
        else:
            T_phi = 0

        """ GET PHI OBS """
        ll_obs_phi = []
        if self. division:
            for l_signal, l_idx_cell_cycle_start in zip(ll_signal,
                                                       ll_idx_cell_cycle_start):
                l_obs_phi = [-1]*l_idx_cell_cycle_start[0]
                first = True
                for idx_div_1, idx_div_2 in zip(l_idx_cell_cycle_start[:-1],
                                                l_idx_cell_cycle_start[1:]):
                    if not first:
                        del l_obs_phi[-1]
                    l_obs_phi.extend([i%(2*np.pi) for i in\
                                  np.linspace(0,2*np.pi,idx_div_2-idx_div_1+1)])
                    first = False

                #l_obs_phi.append(0) #first phase avec the last spike
                l_obs_phi.extend([-1]*(len(l_signal)\
                                                -l_idx_cell_cycle_start[-1]-1))
                ll_obs_phi.append(l_obs_phi)

        """ REMOVE SIGNALS WITH NO DIVISION """
        idx_to_remove = [i for i, l_obs_phi in enumerate(ll_obs_phi) if \
                                            l_obs_phi == len(l_obs_phi) * [-1]]
        ll_area = [ll_area[i] for i in range(len(ll_area)) if \
                                                        i not in idx_to_remove]
        ll_signal = [ll_signal[i] for i in range(len(ll_signal)) if \
                                                        i not in idx_to_remove]
        ll_peak = [ll_peak[i] for i in range(len(ll_peak)) if \
                                                        i not in idx_to_remove]
        if self.division:
            ll_nan_circadian_factor = [ll_nan_circadian_factor[i] for i \
                                        in range(len(ll_nan_circadian_factor)) \
                                        if i not in idx_to_remove]
            ll_obs_phi = [ll_obs_phi[i] for i in range(len(ll_obs_phi)) if \
                                                        i not in idx_to_remove]
            ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] for i \
                                       in range(len(ll_idx_cell_cycle_start))\
                                       if i not in idx_to_remove]
        #avg length of the traces
        #print(np.mean([len(l_signal)-1 for l_signal in ll_signal])/2)

        ''' KEEP THE TRACES WITH AT LEAST 2 CELL-CYCLES '''
        if self.cc:
            idx_to_remove = [i for i, l_idx_cell_cycle_start in \
                            enumerate(ll_idx_cell_cycle_start) \
                            if len(l_idx_cell_cycle_start)<=2 ]
            ll_area = [ll_area[i] for i in range(len(ll_area)) \
                       if i not in idx_to_remove]
            ll_signal = [ll_signal[i] for i in range(len(ll_signal)) \
                        if i not in idx_to_remove]
            ll_nan_circadian_factor = [ll_nan_circadian_factor[i] \
                                for i in range(len(ll_nan_circadian_factor)) \
                                if i not in idx_to_remove]
            ll_obs_phi = [ll_obs_phi[i] for i in range(len(ll_obs_phi)) \
                          if i not in idx_to_remove]
            ll_peak = [ll_peak[i] for i in range(len(ll_peak)) \
                       if i not in idx_to_remove]
            ll_idx_cell_cycle_start = [ ll_idx_cell_cycle_start[i] \
                                for i in range(len(ll_idx_cell_cycle_start)) \
                                if i not in idx_to_remove]

        if self.cc_bis:
            idx_to_remove = [i for i, l_peak in enumerate(ll_peak) \
                             if np.sum(l_peak)<3 ]
            ll_area = [ll_area[i] for i in range(len(ll_area)) \
                       if i not in idx_to_remove]
            ll_signal = [ll_signal[i] for i in range(len(ll_signal)) \
                         if i not in idx_to_remove]
            ll_nan_circadian_factor = [ll_nan_circadian_factor[i] \
                         for i in range(len(ll_nan_circadian_factor)) \
                         if i not in idx_to_remove]
            ll_obs_phi = [ll_obs_phi[i] for i in range(len(ll_obs_phi)) \
                          if i not in idx_to_remove]
            ll_peak = [ll_peak[i] for i in range(len(ll_peak)) \
                       if i not in idx_to_remove]
            ll_idx_cell_cycle_start = [ll_idx_cell_cycle_start[i] \
                                  for i in range(len(ll_idx_cell_cycle_start)) \
                                  if i not in idx_to_remove]

        """ REMOVE THE TRACES WITH TOO LOW PEAKS AND TOO HIGH PERIODS """
        if self.remove:
            idx_to_remove = []
            for (idx, l_signal), l_area, l_peak in zip(enumerate(ll_signal),
                                                        ll_area, ll_peak):
                l_idx_peak = [int(idx) for idx, i in enumerate(l_peak) if i>0]
                for idx_peak in l_idx_peak:
                    if l_signal[idx_peak]< 0.25 and l_area[idx_peak]!=-1:
                        idx_to_remove.append(idx)
                        break
                for idx_peak_1, idx_peak_2 in zip(l_idx_peak[:-1],
                                                  l_idx_peak[1:]):
                    if idx_peak_2-idx_peak_1 >= 36*2:
                        idx_to_remove.append(idx)
                        break

                # plt.plot(l_signal)
                # plt.plot(l_peak)
                # plt.show()
                # plt.close()


            ll_area = [ll_area[i] for i in range(len(ll_area)) \
                       if i not in idx_to_remove]
            ll_signal = [ll_signal[i] for i in range(len(ll_signal)) \
                         if i not in idx_to_remove]
            ll_nan_circadian_factor = [ll_nan_circadian_factor[i] \
                                for i in range(len(ll_nan_circadian_factor)) \
                                if i not in idx_to_remove]
            ll_obs_phi = [ll_obs_phi[i] for i in range(len(ll_obs_phi)) \
                          if i not in idx_to_remove]
            ll_peak = [ll_peak[i] for i in range(len(ll_peak)) \
                       if i not in idx_to_remove]
            ll_idx_cell_cycle_start = [ ll_idx_cell_cycle_start[i] \
                                for i in range(len(ll_idx_cell_cycle_start)) \
                                if i not in idx_to_remove]

        """ SHUFFLE THE TRACES AND KEEP ONLY nb_traces """
        random.seed(1)
        # Given list1 and list2
        ll_area_shuf = []
        ll_signal_shuf = []
        ll_nan_circadian_factor_shuf = []
        ll_obs_phi_shuf = []
        ll_peak_shuf = []
        ll_idx_cell_cycle_start_shuf = []
        index_shuf = list(range(len(ll_area)))
        random.shuffle(index_shuf)
        for count, i in enumerate(index_shuf):
            if count<self.nb_traces:
                ll_signal_shuf.append(ll_signal[i])
                ll_peak_shuf.append(ll_peak[i])
                ll_area_shuf.append(ll_area[i])
                if self.division:
                    ll_nan_circadian_factor_shuf.append(\
                                                    ll_nan_circadian_factor[i])
                    ll_obs_phi_shuf.append(ll_obs_phi[i])
                    ll_idx_cell_cycle_start_shuf.append(\
                                                    ll_idx_cell_cycle_start[i])
            else:
                break


        if not load_annotation:
            return (ll_area_shuf, ll_signal_shuf, ll_nan_circadian_factor_shuf,
                    ll_obs_phi_shuf, T_theta, T_phi)
        else:
            return (ll_area_shuf, ll_signal_shuf, ll_nan_circadian_factor_shuf,
                    ll_obs_phi_shuf, ll_peak_shuf, ll_idx_cell_cycle_start_shuf,
                    T_theta, T_phi)

    def load(self, period_phi = None, period_theta = None,
            load_annotation = False, force_dpd = False,
            force_temperature = False):
        """
        Load and filter the traces with some supplementary conditions.

        Parameters
        ----------
        period_phi : float
            Cell-cycle period according to which the traces must be selected
            (if None, ignored).
        period_theta : float
            Circadian period according to which the traces must be selected
            (if None, ignored).
        load_annotation : bool
            Load or not the list of annotations going with the traces.
        force_dpd : bool
            Force the selection of traces such that there are successively
            divison, peak and division.
        force_temperature : bool
            Force the temperature condition even though a given cell-cycle
            period is selected.
        Returns
        -------
        Sevral lists corresponding to the cell-cycle signal,
        the circadian signal and possible annotations.
        """
        t_lists = self.filter_data()
        if period_phi is None and period_theta is None:
            return self.process_data(t_lists, load_annotation = load_annotation)
        elif period_phi is not None:
            if self.temperature is not None and not force_temperature:
                print("Since a cell-cycle period was specified,"+ \
                       "temperature is ignored")
                self.temperature = None
                t_lists = self.filter_data()

            (ll_area_shuf, ll_signal_shuf, ll_nan_circadian_factor_shuf,
            ll_obs_phi_shuf, ll_peak_shuf, ll_idx_cell_cycle_start_shuf,
            T_theta, T_phi) = self.process_data(t_lists, load_annotation = True)
            ll_area_kept = []
            ll_signal_kept = []
            ll_nan_circadian_factor_kept = []
            ll_obs_phi_kept = []
            l_Tphi_tot = []
            l_Ttheta_tot = []
            ll_peak_kept = []
            ll_idx_cell_cycle_start_kept = []
            #keep only traces with cell-cycle between period-1 and period+1
            for (l_area, l_signal, l_nan_circadian_factor, l_obs_phi, l_peak,
                l_idx_cell_cycle_start) in zip(ll_area_shuf, ll_signal_shuf,
                                              ll_nan_circadian_factor_shuf,
                                              ll_obs_phi_shuf, ll_peak_shuf,
                                              ll_idx_cell_cycle_start_shuf):
                l_idx_peak = [idx for idx, i in enumerate(l_peak) if i==1]


                l_Tphi = []
                for t_start_1, t_start_2 in zip(l_idx_cell_cycle_start[:-1],
                                                l_idx_cell_cycle_start[1:]):
                    l_Tphi.append( (t_start_2-t_start_1)/2)
                T_phi = np.mean(l_Tphi)
                std_T_phi = np.std(l_Tphi)

                l_Ttheta = []
                for t_peak_1, t_peak_2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
                    l_Ttheta.append( (t_peak_2-t_peak_1)/2)

                if T_phi>=period_phi-1 and T_phi<=period_phi+1 and std_T_phi<=2:
                    ll_area_kept.append(l_area)
                    ll_signal_kept.append(l_signal)
                    ll_nan_circadian_factor_kept.append(l_nan_circadian_factor)
                    ll_obs_phi_kept.append(l_obs_phi)

                    l_Tphi_tot.extend(l_Tphi)
                    l_Ttheta_tot.extend(l_Ttheta)

                    ll_peak_kept.append(l_peak)
                    ll_idx_cell_cycle_start_kept.append(l_idx_cell_cycle_start)

            if not load_annotation:
                return (ll_area_kept, ll_signal_kept,
                        ll_nan_circadian_factor_kept, ll_obs_phi_kept,
                        np.mean(l_Ttheta_tot), np.std(l_Ttheta_tot),
                        np.mean(l_Tphi_tot) , np.std(l_Tphi_tot))
            else:
                return (ll_area_kept, ll_signal_kept,
                        ll_nan_circadian_factor_kept, ll_obs_phi_kept,
                        ll_peak_shuf, ll_idx_cell_cycle_start_shuf,
                        np.mean(l_Ttheta_tot), np.std(l_Ttheta_tot),
                        np.mean(l_Tphi_tot) , np.std(l_Tphi_tot))
        else:
            if self.temperature is not None:
                print("Since a circadian period was specified, \
                        temperature is ignored")
                self.temperature = None
                t_lists = self.filter_data()
            (ll_area_shuf, ll_signal_shuf, ll_nan_circadian_factor_shuf,
                    ll_obs_phi_shuf, ll_peak_shuf, ll_idx_cell_cycle_start_shuf,
                    T_theta, T_phi) \
                            = self.process_data(t_lists, load_annotation = True)
            ll_area_kept = []
            ll_signal_kept = []
            ll_nan_circadian_factor_kept = []
            ll_obs_phi_kept = []
            l_Tphi_tot = []
            l_Ttheta_tot = []
            #keep traces with circadian period between period-1 and period+1
            for l_area, l_signal, l_peak in zip(ll_area_shuf, ll_signal_shuf,
                                                ll_peak_shuf):

                l_Ttheta = []
                l_idx_peak = [idx for idx, i in enumerate(l_peak) if i==1]
                if len(l_idx_peak)<2:
                    continue

                for t_peak_1, t_peak_2 in zip(l_idx_peak[:-1], l_idx_peak[1:]):
                    l_Ttheta.append( (t_peak_2-t_peak_1)/2)
                T_theta = np.mean(l_Ttheta)
                std_theta = np.std(l_Ttheta)

                c1 =T_theta<=period_theta+0.5
                c2 = T_theta>=period_theta-0.5
                c3 = std_theta<=0.5
                if c1 and c2 and c3:
                    ll_signal_kept.append(l_signal)
                    ll_area_kept.append(l_area)
            return (ll_area_kept, ll_signal_kept, ll_nan_circadian_factor_kept,
                    ll_obs_phi_kept)


if __name__ == '__main__':
    path =  "../Data/NIH3T3.ALL.2017-04-04/ALL_TRACES_INFORMATION.p"
    ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, T_theta, T_phi \
                                = LoadData(path, temperature = None,
                                            division=False,
                                            several_cell_cycles = False).load()
    print(len(ll_signal))
    path = "../Data/U2OS-2017-03-20/ALL_TRACES_INFORMATION_march_2017.p"
    ll_area, ll_signal, ll_nan_circadian_factor, ll_obs_phi, T_theta, T_phi \
                                = LoadData(path, temperature = 37,
                                            division=True,
                                            several_cell_cycles = False).load()
    print(len(ll_signal))
