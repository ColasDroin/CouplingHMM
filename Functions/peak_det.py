# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np
import sys
""""""""""""""""""""" FUNCTION """""""""""""""""""""
def peakdet(v, delta, x = None):
    """
    Detect the peaks and trough of a given signal.

    Parameters
    ----------
    v : list
        Input signal.
    delta : integer
        Scale parameter

    Returns
    -------
    A list of peaks and a list of troughs for the input signal.
    """

    maxtab = []
    mintab = []

    if x is None:
        x = range(len(v))

    v = np.array(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = 10000000, -10000000
    mnpos, mxpos = np.nan, np.nan

    lookformax = True

    for i in range(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)
