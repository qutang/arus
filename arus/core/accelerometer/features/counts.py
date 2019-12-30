"""

Computing features about different versions of counts

Author: Qu Tang

Date: Jul 10, 2018

"""
import numpy as np
from .. import transformation as accel_transform
from ...libs import num as arus_num


def enmo(X):
    """

    Computing ENMO value of accelerometer data

    Arguments:
        X {numpy.ndarray} -- 2D numpy array M x N, M is the number of samples
         and N is the dimension of the data

    Returns:
        [numpy.ndarray] -- 1 x 1
    """
    X = arus_num.format_arr(X)
    vm = accel_transform.vector_magnitude(X)
    result = np.nanmean(np.clip(vm - 1, a_min=0, a_max=None),
                        axis=0, keepdims=True)
    return result, 'ENMO'
