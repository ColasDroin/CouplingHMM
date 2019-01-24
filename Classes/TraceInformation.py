import pdb
import numpy as np
import matplotlib.pyplot as plt

"""
This file was not coded by me (Colas Droin) but by Eric Paquet, and is therefore
not thoroughly commented, as I'm not sure about the meaning of each variable.
"""

class TraceInformation:
    """
    This class is used to store and compute some information about the
    experimental traces. Each trace is then enclosed in an instance of this
    class.
    """
    def __init__(self, hmm_area_signal_s, orig_idx, ind_s, raw_area, raw_signal,
                 peaks, cell_cycle_start, mitosis_start, serum, temp):
        """
        Constructor of TraceInformation.

        Parameters
        ----------
        hmm_area_signal_s : dictionnary
            todo.
        orig_idx : integer
            todo.
        ind_s : integer
            todo.
        raw_area : list
            Normalized experimental signal for the cell-cycle.
        raw_signal : list
            Normalized experimental signal for the circadian clock.
        peaks : list
            Indexes of the circadian peaks.
        cell_cycle_start : list
            Indexes of the first point after mitosis.
        mitosis_start : list
            Indexes of the beginning of the mitosis.
        serum : string
            Serum condition.
        temp : int
            Temperature condition.
        """

        # size checks
        expected_size = len(hmm_area_signal_s['area'])
        assert(expected_size == \
                            len(hmm_area_signal_s['area_hmm']['mean']['theta']))
        assert(expected_size == \
                            len(hmm_area_signal_s['area_hmm']['max']['theta']))
        assert(expected_size == len(hmm_area_signal_s['signal']))

        # Process the raw signal
        raw_area = [val for val, ind in zip(raw_area,ind_s) if ind!=0]
        raw_signal = [val for val, ind in zip(raw_signal,ind_s) if ind!=0]
        cell_cycle_start = [val for val, ind in zip(cell_cycle_start,ind_s) \
                                                                    if ind!=0]
        mitosis_start = [val for val, ind in zip(mitosis_start,ind_s) if ind!=0]
        peaks = [val for val,ind in zip(peaks, ind_s) if ind!=0]

        # Just need to test one since they all use ind_s
        assert(expected_size ==len(raw_area))

        self.hmm_area_signal = hmm_area_signal_s
        self.hmm_area_mean = hmm_area_signal_s['area_hmm']['mean']['theta']
        self.hmm_area_max = hmm_area_signal_s['area_hmm']['max']['theta']

        self.hmm_circa_mean = hmm_area_signal_s['circadian_hmm']['mean']['theta']
        self.hmm_circa_max = hmm_area_signal_s['circadian_hmm']['max']['theta']

        self.hmm_area_logP = hmm_area_signal_s['area_hmm']['logP']
        self.hmm_circa_logP = hmm_area_signal_s['circadian_hmm']['logP']

        self.raw_area = np.array(raw_area)
        self.raw_circa = np.array(raw_signal)

        self.normalized_area = hmm_area_signal_s['area']
        self.normalized_circa = hmm_area_signal_s['signal']

        self.cell_cycle_start = np.array(cell_cycle_start)
        self.mitosis_start = np.array(mitosis_start)

        self.peaks = np.array(peaks)

        # Original indices in the raw data generated from matlab ie the .mat
        #files
        # IMPORTANT!!! this is a 1-base indice need to -1 to be use in python
        self.orig_idx = orig_idx

        self.temp = temp
        self.serum = serum

    def getNumPeaks(self):
        """
        Get the number of circadian peaks of the current trace.

        Returns
        -------
        The number of circadian peaks of the current trace.
        """
        return sum(self.peaks)

    def getNumDivs(self):
        """
        Get the number of divisons of the current trace.

        Returns
        -------
        The number of divisons of the current trace.
        """
        return sum(self.cell_cycle_start)

    def validAreaHMM(self):
        """
        Check that the cell-cycle trace has been correctly annotated.

        Returns
        -------
        A boolean indicating if the cell-cycle trace has been correctly
        annotated.
        """
        # just be sure the annotated mitosis start/cell cycle start
        # have 1-0 values
        is_valid = True
        cc_idx = [ind for ind,val in zip(range(len(self.cell_cycle_start)),
                                            self.cell_cycle_start) if val == 1]
        look_around = 3
        for ci in cc_idx:
            cur_hmm = self.hmm_area_mean[ max(0,ci-look_around) : \
                                  min(len(self.hmm_area_mean),ci+look_around+1)]
            if (cur_hmm > 0.1).all():
                is_valid = False
                break

        # cc_idx = [ind for ind,val in zip(range(len(self.cell_cycle_start)),
        #                               self.hmm_area_mean) if 0 <= val <= 0.05]
        # for ci in cc_idx:
        #     cur_hmm = np.array(self.cell_cycle_start[max(0,ci-look_around) : \
        #                        min(len(self.hmm_area_mean),ci+look_around+1)])
        #     if (cur_hmm < 1).all():
        #         is_valid = False
        #         break

        return is_valid

    def validCircaHMM(self):
        """
        Check that the circadian trace has been correctly annotated.

        Returns
        -------
        A boolean indicating if the circadian trace has been correctly
        annotated.
        """
        is_valid = True
        cc_idx = [ind for ind, val in zip(range(len(self.peaks)),self.peaks) \
                                                                    if val == 1]
        look_around = 3
        for ci in cc_idx:
            cur_hmm = self.hmm_circa_mean[max(0, ci-look_around) : \
                                         min(len(self.peaks), ci+look_around+1)]
            if (cur_hmm > 0.1).all():
                is_valid = False
                break

        # cc_idx = [ind for ind,val in zip(range(len(self.cell_cycle_start)),
        #                            self.hmm_circa_mean) if 0 <= val <= 0.05]
        # for ci in cc_idx:
        #     cur_hmm = np.array(self.peaks[max(0, ci-look_around) : \
        #                        min(len(self.hmm_area_mean),ci+look_around+1)])
        #     if (cur_hmm < 1).all():
        #         is_valid = False
        #         break

        return is_valid

    def __len__(self):
        """
        Redefine the length function for the current class.

        Returns
        -------
        The length of the circadian trace.
        """
        return len(self.raw_circa)

    def getEvents(self,n):
        """
        Store in a string the cell-cycle events of the trace.

        Parameters
        ----------
        n : int
            ###TO CHECK
            Number of points to ignore around mitosis ?

        Returns
        -------
        The string of events.
        """
        assert(n > 1)

        events = ["" for i in range(len(self))]

        for i in range(len(self)):
            if self.cell_cycle_start[i] == 1:
                events[i] += "C"

            if self.mitosis_start[i] == 1:
                events[i] += "M"

        temp = []
        for i in range(len(self)):
            if events[i] != "":
                temp.append([i,events[i]])

        to_ret = []
        if len(temp) > n:
            for i in range(len(temp) - n + 1):
                to_ret.append({'idx':[temp[i][0], temp[i+n-1][0]],
                        'tag':"".join(list(map(lambda a:a[1],temp[i:(i+n)])))})
        return to_ret

    def plotCircadian(self,dt=0.5,assignment='mean'):
        """
        Plot the circadian trace.

        Parameters
        ----------
        dt : float
            Time resolution
        assignment : string
            todo
        """
        circadian_part = self.hmm_area_signal['circadian_hmm'][assignment]

        plt.figure()
        plt.plot(circadian_part['tspan'], circadian_part['model'],
                                                label='P%s[model]' % assignment)
        plt.plot(circadian_part['tspan'], self.normalized_circa, label='Signal')
        plt.xlabel('t')

        plt.plot(circadian_part['tspan'], circadian_part['theta'], '--',
                                                           label =  r'$\theta$')
        plt.plot(circadian_part['tspan'], circadian_part['amplitude'], '--',
                                                            label = "Amplitude")

        dm_tmp = [ind for ind,val in zip(range(len(self)),
                                            self.cell_cycle_start) if val == 1]
        for cci in dm_tmp:
            plt.plot([circadian_part['tspan'][cci],
                                    circadian_part['tspan'][cci]], [0,1], 'c--')

        dm_tmp_2 = [ind for ind,val in zip(range(len(self)),
                                                        self.peaks) if val == 1]
        for cci in dm_tmp_2:
            plt.plot([circadian_part['tspan'][cci],
                                      circadian_part['tspan'][cci]],[0,1],'m--')
        plt.plot([0,circadian_part['tspan'][-1]],[0,0],'k-')
        plt.legend()

    def plotArea(self,dt=0.5,assignment='mean'):
        """
        Plot the cell-cycle trace.

        Parameters
        ----------
        dt : float
            Time resolution
        assignment : string
            todo
        """
        area_part = self.hmm_area_signal['area_hmm'][assignment]

        plt.figure()
        plt.plot(area_part['tspan'], area_part['model'],
                                                label='P%s[model]' % assignment)
        plt.plot(area_part['tspan'], self.normalized_area, label='Area')
        plt.xlabel('t')

        plt.plot(area_part['tspan'], area_part['theta'], '--',
                                                           label =  r'$\theta$')
        plt.plot(area_part['tspan'], area_part['amplitude'], '--',
                                                            label = "Amplitude")

        for cci in [ind for ind,val in zip(range(len(self)),
                                            self.cell_cycle_start) if val == 1]:
            plt.plot([area_part['tspan'][cci], area_part['tspan'][cci]],
                                                                   [0,1], 'c--')

        for cci in [ind for ind,val in zip(range(len(self)),
                                                       self.peaks) if val == 1]:
            plt.plot([area_part['tspan'][cci], area_part['tspan'][cci]],
                                                                   [0,1], 'm--')

        plt.plot([0,area_part['tspan'][-1]],[0,0],'k-')
        plt.legend()
