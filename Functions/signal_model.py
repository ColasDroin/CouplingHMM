# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np

""""""""""""""""""""" FUNCTION """""""""""""""""""""
def signal_model(l_parameters, waveform=None):
    """
    Defines the model for the signal, given the hidden variables.
    CAUTION : This model is also defined in the HMM class, so each change made
    on this function must also be done there.

    Parameters
    ----------
    l_parameters : list
        List of hidden variables.
    waveform : list
        Waveform. If not given, a default waveform is used.

    Returns
    -------
    The value of the signal given the hidden variables.
    """
    if waveform is None:
        exponent = 1.6
        return ((1+np.cos( np.array(l_parameters[0])))/2)**exponent \
                                    * np.exp(l_parameters[1]) + l_parameters[2]
    else:
        return waveform( l_parameters[0] )*np.exp(l_parameters[1]) \
                                                              + l_parameters[2]
