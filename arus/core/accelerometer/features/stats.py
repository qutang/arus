"""

Computing features of descriptive statistics for accelerometer data

Author: Qu Tang

Date: Oct 16, 2019

"""
import numpy as np
from ...libs.num import format_arr


def mean(X):
    X = format_arr(X)
    result = np.nanmean(X, axis=0, keepdims=True)
    return result, ['MEAN_' + str(i) for i in range(X.shape[1])]


def std(X):
    X = format_arr(X)
    result = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    return result, ['STD_' + str(i) for i in range(X.shape[1])]


def max_value(X):
    X = format_arr(X)
    result = np.nanmax(X, axis=0, keepdims=True)
    return result, ['MAX_' + str(i) for i in range(X.shape[1])]


def min_value(X):
    X = format_arr(X)
    result = np.nanmin(X, axis=0, keepdims=True)
    return result, ['MIN_' + str(i) for i in range(X.shape[1])]


def max_minus_min(X):
    X = format_arr(X)
    result = max_value(X)[0] - min_value(X)[0]
    return result, ['RANGE_' + str(i) for i in range(X.shape[1])]


def abs_max_value(X):
    X = format_arr(X)
    result = np.nanmax(np.abs(X), axis=0, keepdims=True)
    return result, ['ABS_MAX_' + str(i) for i in range(X.shape[1])]


def abs_min_value(X):
    X = format_arr(X)
    result = np.nanmin(np.abs(X), axis=0, keepdims=True)
    return result, ['ABS_MIN_' + str(i) for i in range(X.shape[1])]
