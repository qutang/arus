"""
Module of extension functions to be applied to numpy objects (e.g., Arrays)

Author: Qu Tang

Date: 2020-02-03

License: see LICENSE file
"""

import numpy as np
import numpy.linalg as la
from scipy import interpolate
from scipy import signal as sp_signal


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


def mutate_nan(X):
    X = atleast_float_2d(X)
    X_new = np.apply_along_axis(_fill_nan_1d, axis=0, arr=X)
    return X_new


def _fill_nan_1d(y):
    xnew = range(len(y))
    x, y = _remove_nan_1d(y)
    if len(x) < 3:
        ynew = np.interp(xnew, x, y)
    else:
        s = interpolate.InterpolatedUnivariateSpline(x, y)
        ynew = s(xnew)
    return ynew


def regularize_sr(t, X, sr, st=None, et=None):
    X = atleast_float_2d(X)
    st = st or t[0]
    et = et or t[-1]
    total_seconds = et - st
    new_t = np.linspace(st, et, num=int(
        np.floor(total_seconds * sr)), endpoint=False)
    new_X = np.apply_along_axis(
        _regularize_sr, axis=0, arr=X, t=t, new_t=new_t)
    return new_t, new_X


def _regularize_sr(y, t, new_t):
    f = interpolate.InterpolatedUnivariateSpline(t, y, k=3)
    new_y = f(new_t)
    return new_y


def _remove_nan_1d(y):
    x = np.arange(len(y))
    selection = np.logical_not(np.isnan(y))
    x = x[selection]
    y = y[selection]
    return x, y


def butterworth(X, sr, cut_offs, order=-1, filter_type='low'):
    """Apply butterworth IIR filter to a 2D array

    Args:
        X (numpy.ndarray): A 1D or 2D numpy array
        sr (float): sampling rate in Hz
        cut_offs (float or list): cut off frequencies in Hz
        order (float): order of the filter

    Keyword Arguments:
        filter_type (str): type of the filter. Could be 'low', 'high', 'pass' or 'stop'

    Returns:
        filtered_X (numpy.ndarray): A numpy array with the same shape as X
    """
    Wn = _get_wn(cut_offs, sr)
    B, A = sp_signal.butter(order, Wn, btype=filter_type, output='ba')
    result = sp_signal.filtfilt(B, A, X, axis=0, method='pad', padtype='even')
    return result


def resample(X, sr, new_sr):
    """Apply resampling IIR filter to a 2D array

    Args:
        X (numpy.ndarray): A 1D or 2D numpy array
        sr (float): sampling rate in Hz
        new_sr (float): resampled sampling rate in Hz

    Returns:
        resampled_X (numpy.ndarray): A numpy array with the new sampling rate
    """
    lcm = np.lcm(sr, new_sr)
    up = lcm / sr
    down = lcm / new_sr
    # TODO: use prime factorization to generate a cascade of downsampling filter
    new_X = sp_signal.resample_poly(X, up=up, down=down, axis=0)
    return new_X


def resample_timestamps(ts, new_n):
    """Resample timestamps to new sampling rate

    This function should be used together with `arus.extensions.numpy.resample`.

    Args:
        ts (numpy.ndarray): A 1D numpy array storing timestamps in `np.datetime64` format
        new_n (int): number of samples in the resampled sequence

    Returns:
        resampled_ts (numpy.ndarray): A 1D numpy array with the resampled timestamps
    """
    assert type(ts[0]) == np.datetime64
    st = ts[0].astype('datetime64[ms]').astype('float64')
    et = ts[-1].astype('datetime64[ms]').astype('float64')
    new_ts = np.linspace(st, et, num=new_n)
    vf = np.vectorize(lambda x: np.datetime64(int(x), 'ms'))
    new_ts = vf(new_ts)
    return new_ts


def _get_wn(cut_offs, sr):
    nyquist = sr / 2.0
    return atleast_scalar(cut_offs) / nyquist


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) &
                           (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) &
                           (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(
            np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind)
    return ind


def _plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        plt.show()


def apply_over_subwins(X, func, subwins=None, subwin_samples=None, has_names=False, **kwargs):
    X = atleast_float_2d(X)
    if subwins is not None:
        # compute the length of each sub window if the number of sub windows is given.
        if subwins == 0:
            # if subwins is zero, treat as one
            subwins = 1
            win_length = X.shape[0]
        else:
            win_length = int(np.floor(X.shape[0] / subwins))
            if win_length == 0:
                # if subwins is zero, treat as one
                subwins = 1
                win_length = X.shape[0]
    elif subwin_samples is not None:
        # or if the number of samples of each sub window is provided, compute the number of sub windows.
        if subwin_samples == 0:
            subwins = 1
            win_length = X.shape[0]
        else:
            win_length = subwin_samples
            subwins = int(np.floor(X.shape[0] / subwin_samples))
            if subwins == 0:
                # if subwins is zero, treat as one
                subwins = 1
                win_length = X.shape[0]
    else:
        # or treat the entire input array as a single sub window
        subwins = 1
        win_length = X.shape[0]

    # Use the sub windows evenly reside in the middle of the bigger window
    start_index = np.ceil((X.shape[0] - subwins * win_length) / 2)

    result = []
    for i in range(subwins):
        indices = int(start_index) + np.array(range(
            i * win_length,
            (i + 1) * win_length
        ))
        subwin_X = X[indices, :]
        if has_names:
            subwin_result, func_names = func(subwin_X, **kwargs)
        else:
            subwin_result = func(subwin_X, **kwargs)
        subwin_result = atleast_float_2d(subwin_result)
        result.append(subwin_result)
    # each row is the result from one sub window
    final_result = np.concatenate(result, axis=0)
    if has_names:
        return final_result, func_names
    else:
        return final_result


def vector_magnitude(X):
    X = atleast_float_2d(X)
    result = la.norm(X, ord=2, axis=1, keepdims=True)
    return result


def flip_and_swap(X, x_flip, y_flip, z_flip):
    X = atleast_float_2d(X)
    X_clone = np.copy(X)
    x = np.copy(X_clone[:, 0])
    y = np.copy(X_clone[:, 1])
    z = np.copy(X_clone[:, 2])
    x_flip = x_flip.lower()
    y_flip = y_flip.lower()
    z_flip = z_flip.lower()
    if x_flip == 'x':
        X_clone[:, 0] = x
    elif x_flip == '-x':
        X_clone[:, 0] = -x
    elif x_flip == 'y':
        X_clone[:, 0] = y
    elif x_flip == '-y':
        X_clone[:, 0] = -y
    elif x_flip == 'z':
        X_clone[:, 0] = z
    elif x_flip == '-z':
        X_clone[:, 0] = -z

    if y_flip == 'x':
        X_clone[:, 1] = x
    elif y_flip == '-x':
        X_clone[:, 1] = -x
    elif y_flip == 'y':
        X_clone[:, 1] = y
    elif y_flip == '-y':
        X_clone[:, 1] = -y
    elif y_flip == 'z':
        X_clone[:, 1] = z
    elif y_flip == '-z':
        X_clone[:, 1] = -z

    if z_flip == 'x':
        X_clone[:, 2] = x
    elif z_flip == '-x':
        X_clone[:, 2] = -x
    elif z_flip == 'y':
        X_clone[:, 2] = y
    elif z_flip == '-y':
        X_clone[:, 2] = -y
    elif z_flip == 'z':
        X_clone[:, 2] = z
    elif z_flip == '-z':
        X_clone[:, 2] = -z

    return X_clone
