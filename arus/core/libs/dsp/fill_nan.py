import numpy as np
from scipy import interpolate
from ..num import format_arr


def _remove_nan_1d(y):
    x = np.arange(len(y))
    selection = np.logical_not(np.isnan(y))
    x = x[selection]
    y = y[selection]
    return x, y


def _fill_nan_1d(y):
    xnew = range(len(y))
    x, y = _remove_nan_1d(y)
    if len(x) < 3:
        ynew = np.interp(xnew, x, y)
    else:
        s = interpolate.InterpolatedUnivariateSpline(x, y)
        ynew = s(xnew)
    return ynew


def fill_nan(X):
    X = format_arr(X)
    X_new = np.apply_along_axis(_fill_nan_1d, axis=0, arr=X)
    return X_new
