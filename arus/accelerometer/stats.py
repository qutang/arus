"""
Computing features of descriptive statistics for accelerometer data
"""
import numpy as np
from scipy import stats as sp_stats
from .. import extensions


def mean(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = np.nanmean(X, axis=0, keepdims=True)
    return result, ['MEAN_' + str(i) for i in range(X.shape[1])]


def median(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = np.nanmedian(X, axis=0, keepdims=True)
    return result, ['MEDIAN_' + str(i) for i in range(X.shape[1])]


def std(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    return result, ['STD_' + str(i) for i in range(X.shape[1])]


def skew(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = np.atleast_2d(sp_stats.skew(X, axis=0, nan_policy='omit'))
    result = result, ['SKEW_' + str(i) for i in range(X.shape[1])]
    return result


def kurtosis(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = np.atleast_2d(sp_stats.kurtosis(X, axis=0, nan_policy='omit'))
    result = result, ['KURTOSIS_' + str(i) for i in range(X.shape[1])]
    return result


def max_value(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = np.nanmax(X, axis=0, keepdims=True)
    return result, ['MAX_' + str(i) for i in range(X.shape[1])]


def min_value(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = np.nanmin(X, axis=0, keepdims=True)
    return result, ['MIN_' + str(i) for i in range(X.shape[1])]


def max_minus_min(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = max_value(X)[0] - min_value(X)[0]
    return result, ['RANGE_' + str(i) for i in range(X.shape[1])]


def abs_max_value(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = np.nanmax(np.abs(X), axis=0, keepdims=True)
    return result, ['ABS_MAX_' + str(i) for i in range(X.shape[1])]


def abs_min_value(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = np.nanmin(np.abs(X), axis=0, keepdims=True)
    return result, ['ABS_MIN_' + str(i) for i in range(X.shape[1])]


def correlation(X):
    X = extensions.numpy.atleast_float_2d(X)
    corr_mat = np.corrcoef(X, rowvar=False)
    if np.isscalar(corr_mat) and np.isnan(corr_mat):
        result = np.repeat(np.nan, X.shape[1])
    else:
        inds = np.tril_indices(n=corr_mat.shape[0], k=-1, m=corr_mat.shape[1])
        result = []
        for i, j in zip(inds[0], inds[1]):
            result.append(corr_mat[i, j])
    result = np.atleast_2d(result)
    return result, ['CORRELATION_' + str(i) for i in range(result.shape[1])]
