"""
Computing features of descriptive statistics for accelerometer data
"""
import numpy as np
from scipy import stats as sp_stats
from .. import extensions as ext


STAT_FEATURE_NAME_PREFIX = [
    'MEAN',
    'MEDIAN',
    'STD',
    'SKEW',
    'KURTOSIS',
    'MAX',
    'MIN',
    'RANGE',
    'ABS_MAX',
    'ABS_MIN',
    'ZCR',
    'MCR',
    'TOTAL_POWER',
    'CORRELATION'
]


def mean(X):
    X = ext.numpy.atleast_float_2d(X)
    result = np.nanmean(X, axis=0, keepdims=True)
    return result, [f'{STAT_FEATURE_NAME_PREFIX[0]}_{i}' for i in range(X.shape[1])]


def median(X):
    X = ext.numpy.atleast_float_2d(X)
    result = np.nanmedian(X, axis=0, keepdims=True)
    return result, [f'{STAT_FEATURE_NAME_PREFIX[1]}_{i}' for i in range(X.shape[1])]


def std(X):
    X = ext.numpy.atleast_float_2d(X)
    result = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    return result, [f'{STAT_FEATURE_NAME_PREFIX[2]}_{i}' for i in range(X.shape[1])]


def skew(X):
    X = ext.numpy.atleast_float_2d(X)
    result = np.atleast_2d(sp_stats.skew(X, axis=0, nan_policy='omit'))
    result = result, [
        f'{STAT_FEATURE_NAME_PREFIX[3]}_{i}' for i in range(X.shape[1])]
    return result


def kurtosis(X):
    X = ext.numpy.atleast_float_2d(X)
    result = np.atleast_2d(sp_stats.kurtosis(X, axis=0, nan_policy='omit'))
    result = result, [
        f'{STAT_FEATURE_NAME_PREFIX[4]}_{i}' for i in range(X.shape[1])]
    return result


def max_value(X):
    X = ext.numpy.atleast_float_2d(X)
    result = np.nanmax(X, axis=0, keepdims=True)
    return result, [f'{STAT_FEATURE_NAME_PREFIX[5]}_{i}' for i in range(X.shape[1])]


def min_value(X):
    X = ext.numpy.atleast_float_2d(X)
    result = np.nanmin(X, axis=0, keepdims=True)
    return result, [f'{STAT_FEATURE_NAME_PREFIX[6]}_{i}' for i in range(X.shape[1])]


def max_minus_min(X):
    X = ext.numpy.atleast_float_2d(X)
    result = max_value(X)[0] - min_value(X)[0]
    return result, [f'{STAT_FEATURE_NAME_PREFIX[7]}_{i}' for i in range(X.shape[1])]


def abs_max_value(X):
    X = ext.numpy.atleast_float_2d(X)
    result = np.nanmax(np.abs(X), axis=0, keepdims=True)
    return result, [f'{STAT_FEATURE_NAME_PREFIX[8]}_{i}' for i in range(X.shape[1])]


def abs_min_value(X):
    X = ext.numpy.atleast_float_2d(X)
    result = np.nanmin(np.abs(X), axis=0, keepdims=True)
    return result, [f'{STAT_FEATURE_NAME_PREFIX[9]}_{i}' for i in range(X.shape[1])]


def zcr(X):
    """Compute zero crossing rate
    """
    X = ext.numpy.atleast_float_2d(X)
    X_offset = X[1:, :]
    X_orig = X[:-1, :]
    zero_crossings = np.sum(np.multiply(X_orig, X_offset)
                            < 0, axis=0, keepdims=True)
    zc_rates = zero_crossings / (X.shape[0] - 1)
    return zc_rates, [f'{STAT_FEATURE_NAME_PREFIX[10]}_{i}' for i in range(X.shape[1])]


def mcr(X):
    """Compute mean crossing rate
    """
    X = ext.numpy.atleast_float_2d(X)
    X_demean = X - np.mean(X, axis=0, keepdims=True)
    mc_rates, _ = zcr(X_demean)
    return mc_rates, [f'{STAT_FEATURE_NAME_PREFIX[11]}_{i}' for i in range(X.shape[1])]


def total_power(X):
    """Compute total power
    """
    X = ext.numpy.atleast_float_2d(X)
    X = np.power(X, 2)
    total_energy = np.sum(X, axis=0, keepdims=True)
    result = total_energy / X.shape[0]
    return result, [f'{STAT_FEATURE_NAME_PREFIX[12]}_{i}' for i in range(X.shape[1])]


def correlation(X):
    X = ext.numpy.atleast_float_2d(X)
    if X.shape[1] != 3:
        result = np.repeat(np.nan, X.shape[1])
    else:
        corr_mat = np.corrcoef(X, rowvar=False)
        if np.isscalar(corr_mat) and np.isnan(corr_mat):
            result = np.repeat(np.nan, X.shape[1])
        else:
            inds = np.tril_indices(
                n=corr_mat.shape[0], k=-1, m=corr_mat.shape[1])
            result = []
            for i, j in zip(inds[0], inds[1]):
                result.append(corr_mat[i, j])
    result = np.atleast_2d(result)
    return result, [f'{STAT_FEATURE_NAME_PREFIX[13]}_{i}' for i in range(result.shape[1])]


def stat_features(X, selected=STAT_FEATURE_NAME_PREFIX):
    X = ext.numpy.atleast_float_2d(X)
    stat_funcs = [
        mean,
        median,
        std,
        skew,
        kurtosis,
        max_value,
        min_value,
        max_minus_min,
        abs_max_value,
        abs_min_value,
        zcr,
        mcr,
        total_power,
        correlation
    ]

    fv = []
    fv_names = []
    for func, prefix in zip(stat_funcs, STAT_FEATURE_NAME_PREFIX):
        if prefix in selected:
            result, names = func(X)
            fv.append(result)
            fv_names += names

    if len(fv) == 0:
        return None, None

    result = np.concatenate(fv, axis=1)
    return result, fv_names
