"""
Module of extension functions to be applied to numpy objects (e.g., Arrays)

Author: Qu Tang

Date: 2020-02-03

License: see LICENSE file
"""

import numpy as np


def atleast_scalar(arg):
    if np.isscalar(arg):
        result = np.float64(arg)
    else:
        result = np.array(arg, dtype='float64')
    return result


def atleast_float_2d(arr):
    arr = np.float64(arr)
    arr = np.atleast_2d(arr)
    return arr