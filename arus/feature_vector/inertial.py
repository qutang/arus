import functools

import pandas as pd
import numpy as np

from .. import accelerometer as accel
from .. import extensions as ext
from .. import mhealth_format as mh

INERTIAL_FEATURE_NAME_PREFIX = accel.STAT_FEATURE_NAME_PREFIX + \
    accel.SPECTRUM_FEATURE_NAME_PREFIX + \
    accel.ACTIVATION_FEATURE_NAME_PREFIX + \
    accel.ORIENTATION_FEATURE_NAME_PREFIX


def assemble_fv_names(selected=INERTIAL_FEATURE_NAME_PREFIX, use_vm=True, num_of_axes=3):
    fv_names = []
    for prefix in selected:
        if prefix in accel.ORIENTATION_FEATURE_NAME_PREFIX:
            fv_names += [f"{prefix}_{i}" for i in [0, 1, 2]]
        else:
            if use_vm:
                fv_names += [f"{prefix}_0"]
            else:
                fv_names += [f"{prefix}_{i}" for i in range(num_of_axes)]
    return fv_names


def single_triaxial(raw_df, sr, st, et, subwin_secs=2, ori_unit='rad', activation_threshold=0.2, use_vm=True, selected=INERTIAL_FEATURE_NAME_PREFIX):

    result = {
        mh.TIMESTAMP_COL: [st],
        mh.START_TIME_COL: [st],
        mh.STOP_TIME_COL: [et]
    }

    num_samples = round((et - st).total_seconds() * sr)

    if raw_df.shape[0] == 0 or raw_df.dropna().shape[0] < num_samples * 0.8:
        # TO BE IMPROVED
        # Now input raw data should include no less than 80% of the whole window
        fv_names = assemble_fv_names(selected=selected, use_vm=use_vm)
        for name in fv_names:
            result[name] = [np.nan]
        feature_names = fv_names
    else:
        # Prepare parameters
        subwin_samples = int(subwin_secs * sr)

        # Unify input matrix
        X = ext.numpy.atleast_float_2d(raw_df.values[:, 1:4])
        # fill nan at first, nan will be filled by spline interpolation
        X = ext.numpy.mutate_nan(X)

        # Smoothing input raw data
        if use_vm:
            X_vm = ext.numpy.vector_magnitude(X)
            X_vm_filtered = ext.numpy.butterworth(
                X_vm, sr=sr, cut_offs=20, order=4, filter_type='low')

        X_filtered = ext.numpy.butterworth(X, sr=sr, cut_offs=20,
                                           order=4, filter_type='low')

        ts_feature_funcs = [
            functools.partial(accel.stat_features, selected=selected),
            functools.partial(accel.spectrum_features,
                              sr=sr, n=1, selected=selected),
            functools.partial(accel.activation_features,
                              threshold=activation_threshold, selected=selected)
        ]
        orient_feature_funcs = [
            functools.partial(accel.orientation_features,
                              subwin_samples=subwin_samples, unit=ori_unit, selected=selected)
        ]
        if use_vm:
            X_for_ts = X_vm_filtered
        else:
            X_for_ts = X_filtered

        feature_names = []
        for func in ts_feature_funcs:
            values, names = func(X_for_ts)
            if values is None:
                continue
            for value, name in zip(values.transpose(), names):
                result[name] = value.tolist()
                feature_names.append(name)

        for func in orient_feature_funcs:
            values, names = func(X_filtered)
            if values is None:
                continue
            for value, name in zip(values.transpose(), names):
                result[name] = value.tolist()
                feature_names.append(name)
    result = pd.DataFrame.from_dict(result)
    return result, feature_names
