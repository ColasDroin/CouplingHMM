# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
import numpy as np
import sys

""""""""""""""""""""" FUNCTION """""""""""""""""""""
def clean_waveform(W):
    """
    Take the raw waveform and clean/resize it.

    Parameters
    ----------
    W : list
        Raw waveform.

    Returns
    -------
    The processed waveform.
    """

    #rescale waveform between 0 and 1
    W = W -np.min(W)
    W = W/np.max(W)

    #smooth signal using Fourier
    rft = np.fft.rfft(W)
    #rft[15:] = 0
    rft[5:] = 0
    W = np.fft.irfft(rft)

    #slide the waveform such that max is at 0
    idx_max = np.argmax(W)
    W = np.hstack((W[idx_max:],W[:idx_max]))

    #rescale waveform between 0 and 1
    W = W -np.min(W)
    W = W/np.max(W)

    return W
