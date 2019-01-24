# -*- coding: utf-8 -*-
""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

""""""""""""""""""""" FUNCTION """""""""""""""""""""
def make_colormap(seq):
    """
    Return a LinearSegmentedColormap

    Parameters
    ----------
    seq : list
    A sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).

    Returns
    -------
    The desired colormap.
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

""""""""""""""""""""" TEST """""""""""""""""""""
if __name__ == '__main__':
    #colormap
    c = mcolors.ColorConverter().to_rgb
    bwr = make_colormap([c('blue'), c('white'), 0.25, c('white'),
                         0.75,c('white'),  c('red')])
