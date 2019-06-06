from scipy.signal import butter, filtfilt, resample_poly
from ..num import atleast_scalar
import numpy as np


def _get_wn(cut_offs, sr):
    nyquist = sr / 2.0
    return atleast_scalar(cut_offs) / nyquist


def butterworth(X, sr, cut_offs, order=-1, filter_type='low'):
    """Apply butterworth IIR filter to a 2D array

    Parameters
    ----------
        X {ndarray} -- A 1D or 2D numpy array
        sr {float} -- sampling rate in Hz
        cut_offs {float or list} -- cut off frequencies in Hz
        order {float} -- order of the filter

    Keyword Arguments:
        filter_type {str} -- type of the filter. Could be 'low', 'high', 'pass' or 'stop' (default: {'low'})

    Returns:
        [ndarray] -- A numpy array with the same shape as X
    """
    Wn = _get_wn(cut_offs, sr)
    B, A = butter(order, Wn, btype=filter_type, output='ba')
    result = filtfilt(B, A, X, axis=0, method='pad', padtype='even')
    return result


def resample(X, sr, new_sr):
    """Apply resampling IIR filter to a 2D array

    Parameters
    ----------
        X {ndarray} -- A 1D or 2D numpy array
        sr {float} -- sampling rate in Hz
        new_sr {float} -- resampled sampling rate in Hz

    Returns:
        [ndarray] -- A numpy array with the new sampling rate
    """
    lcm = np.lcm(sr, new_sr)
    up = lcm / sr
    down = lcm / new_sr
    # TODO: use prime factorization to generate a cascade of downsampling filter
    new_X = resample_poly(X, up=up, down=down, axis=0)
    return new_X
