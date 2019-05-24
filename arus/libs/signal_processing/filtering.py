from scipy.signal import butter, filtfilt
from ..data_formatting import as_scalar_or_np_array


def _get_wn(cut_offs, sr):
    nyquist = sr / 2.0
    return as_scalar_or_np_array(cut_offs) / nyquist


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
