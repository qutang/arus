import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress
from .. import extensions as ext


EXTREMA_FEATURE_NAME_PREFIX = [
    'MAX_COUNT',
    'MIN_COUNT',
    'MAX_VAL_MEAN',
    'MAX_VAL_STD',
    'MIN_VAL_MEAN',
    'MIN_VAL_STD',
    'EXTREMA_RANGE_MEAN',
    'EXTREMA_RANGE_STD',
    'EXTREMA_RANGE_MAX',
    'EXTREMA_RANGE_MIN',
    'EXTREMA_DUR_MEAN',
    'EXTREMA_DUR_STD',
    'EXTREMA_CORR_MEAN',
    'EXTREMA_CORR_STD'
]


def extrema_features(X, sr, X_unfiltered=None, threshold=None, selected=EXTREMA_FEATURE_NAME_PREFIX):
    X = ext.numpy.atleast_float_2d(X)

    fv = []
    fv_names = []

    if EXTREMA_FEATURE_NAME_PREFIX[0] in selected:
        f, _ = extrema_count(X, threshold=threshold)
        fv += f[0, ::2].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[0]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[1] in selected:
        f, _ = extrema_count(X, threshold=threshold)
        fv += f[0, 1::2].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[1]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[2] in selected:
        f, _ = extrema_value(X, threshold=threshold)
        fv += f[0, ::4].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[2]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[3] in selected:
        f, _ = extrema_value(X, threshold=threshold)
        fv += f[0, 1::4].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[3]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[4] in selected:
        f, _ = extrema_value(X, threshold=threshold)
        fv += f[0, 2::4].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[4]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[5] in selected:
        f, _ = extrema_value(X, threshold=threshold)
        fv += f[0, 3::4].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[5]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[6] in selected:
        f, _ = extrema_range(X, threshold=threshold)
        fv += f[0, ::4].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[6]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[7] in selected:
        f, _ = extrema_range(X, threshold=threshold)
        fv += f[0, 1::4].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[7]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[8] in selected:
        f, _ = extrema_range(X, threshold=threshold)
        fv += f[0, 2::4].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[8]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[9] in selected:
        f, _ = extrema_range(X, threshold=threshold)
        fv += f[0, 3::4].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[9]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[10] in selected:
        f, _ = extrema_duration(X, sr=sr, threshold=threshold)
        fv += f[0, ::2].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[10]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[11] in selected:
        f, names = extrema_duration(X, sr=sr, threshold=threshold)
        fv += f[0, 1::2].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[11]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[12] in selected:
        f, _ = extrema_corr(
            X, sr=sr, X_unfilered=X_unfiltered, threshold=threshold)
        fv += f[0, ::2].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[12]}_{i}' for i in range(X.shape[1])]

    if EXTREMA_FEATURE_NAME_PREFIX[13] in selected:
        f, _ = extrema_corr(
            X, sr=sr, X_unfilered=X_unfiltered, threshold=threshold)
        fv += f[0, 1::2].tolist()
        fv_names += [
            f'{EXTREMA_FEATURE_NAME_PREFIX[13]}_{i}' for i in range(X.shape[1])]

    if len(fv) == 0:
        return None, None

    result = np.atleast_2d(fv)

    return result, fv_names


def get_extrema(x):
    maxima_inds, _ = find_peaks(x)
    minima_inds, _ = find_peaks(-x)
    return maxima_inds, minima_inds


def extrema_count(X, threshold=None):
    X = ext.numpy.atleast_float_2d(X)
    X_extrema_count = []
    names = []
    for i in range(X.shape[1]):
        max_ids, min_ids = get_extrema(X[:, i])
        X_extrema_count.append(max_ids.shape[0])
        X_extrema_count.append(min_ids.shape[0])
        names.append(f'MAX_COUNT_{i}')
        names.append(f'MIN_COUNT_{i}')
    result = ext.numpy.atleast_float_2d(X_extrema_count)
    return result, names


def extrema_value(X, threshold=None):
    X = ext.numpy.atleast_float_2d(X)
    X_extrema_value = []
    names = []
    for i in range(X.shape[1]):
        max_ids, min_ids = get_extrema(X[:, i])
        max_values, min_values = X[max_ids, i], X[min_ids, i]
        if len(max_values) == 0:
            max_value_mean = np.nan
            max_value_std = np.nan
        else:
            max_value_mean = np.nanmean(max_values)
            max_value_std = np.nanstd(max_values)
        if len(min_values) == 0:
            min_value_mean = np.nan
            min_value_std = np.nan
        else:
            min_value_mean = np.nanmean(min_values)
            min_value_std = np.nanstd(min_values)
        X_extrema_value.append(max_value_mean)
        X_extrema_value.append(max_value_std)
        X_extrema_value.append(min_value_mean)
        X_extrema_value.append(min_value_std)
        names.append(f'MAX_VAL_MEAN_{i}')
        names.append(f'MAX_VAL_STD_{i}')
        names.append(f'MIN_VAL_MEAN_{i}')
        names.append(f'MIN_VAL_STD_{i}')
    result = ext.numpy.atleast_float_2d(X_extrema_value)
    return result, names


def extrema_range(X, threshold=None):
    X = ext.numpy.atleast_float_2d(X)
    X_extrema_range = []
    names = []
    for i in range(X.shape[1]):
        max_ids, min_ids = get_extrema(X[:, i])
        extrema_ids = sorted(max_ids.tolist() + min_ids.tolist())
        extrema_ranges = np.abs(np.diff(X[extrema_ids, i]))
        if len(extrema_ranges) == 0:
            X_extrema_range += [np.nan] * 4
        else:
            X_extrema_range.append(np.nanmean(extrema_ranges))
            X_extrema_range.append(np.nanstd(extrema_ranges))
            X_extrema_range.append(np.nanmax(extrema_ranges))
            X_extrema_range.append(np.nanmin(extrema_ranges))
        names.append(f'EXTREMA_RANGE_MEAN_{i}')
        names.append(f'EXTREMA_RANGE_STD_{i}')
        names.append(f'EXTREMA_RANGE_MAX_{i}')
        names.append(f'EXTREMA_RANGE_MIN_{i}')
    result = ext.numpy.atleast_float_2d(X_extrema_range)
    return result, names


def extrema_duration(X, sr, threshold=None):
    X = ext.numpy.atleast_float_2d(X)
    X_extrema_duration = []
    names = []
    for i in range(X.shape[1]):
        max_ids, min_ids = get_extrema(X[:, i])
        extrema_ids = sorted(max_ids.tolist() + min_ids.tolist())
        extrema_durations = np.diff(extrema_ids)
        extrema_durations_mean = np.nanmean(
            extrema_durations) / float(sr)  # in seconds
        extrema_durations_std = np.nanstd(
            extrema_durations) / float(sr)  # in seconds
        X_extrema_duration.append(extrema_durations_mean)
        X_extrema_duration.append(extrema_durations_std)
        names.append(f'EXTREMA_DUR_MEAN_{i}')
        names.append(f'EXTREMA_DUR_STD_{i}')
    result = ext.numpy.atleast_float_2d(X_extrema_duration)
    return result, names


def extrema_corr(X, sr, X_unfilered=None, threshold=None):
    X = ext.numpy.atleast_float_2d(X)
    X_extrema_corr = []
    names = []
    for i in range(X.shape[1]):
        max_ids, min_ids = get_extrema(X[:, i])
        extrema_ids = sorted(max_ids.tolist() + min_ids.tolist())
        corr_list = []
        for j in range(len(extrema_ids) - 1):
            start = extrema_ids[j]
            stop = extrema_ids[j + 1]
            start_value = X[start, i]
            stop_value = X[stop, i]
            if stop - start < 2:
                continue
            x = np.arange(start=start, stop=stop)
            if X_unfilered is None:
                y_origin = X[x, i]
            else:
                y_origin = X_unfilered[x, i]
            slope, intercept, _, _, _ = linregress(
                [start, stop], [start_value, stop_value])
            y = slope * x + intercept
            corr_list.append(np.correlate(y, y_origin)[0])
        if len(corr_list) == 0:
            X_extrema_corr.append(np.nan)
            X_extrema_corr.append(np.nan)
        else:
            X_extrema_corr.append(np.nanmean(corr_list))
            X_extrema_corr.append(np.nanstd(corr_list))
        names.append(f'EXTREMA_CORR_MEAN_{i}')
        names.append(f'EXTREMA_CORR_STD_{i}')
    result = ext.numpy.atleast_float_2d(X_extrema_corr)
    return result, names
