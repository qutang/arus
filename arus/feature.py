"""
Compute common features for activity recognition
"""

from . import extensions as ext
from . import accelerometer as accel
import numpy as np
import pandas as pd
import functools


def time_freq_orient(raw_df, sr, st, et, subwin_secs=2, ori_unit='rad', activation_threshold=0.2, use_vm=True):

    result = {
        'HEADER_TIME_STAMP': [st],
        'START_TIME': [st],
        'STOP_TIME': [et]
    }

    VM_TIME_FREQ_FEATURE_NAMES = ['MEAN_0',
                                  'STD_0',
                                  'MAX_0',
                                  'RANGE_0',
                                  'DOM_FREQ_0', 'FREQ_POWER_RATIO_ABOVE_3DOT5_0', 'DOM_FREQ_POWER_RATIO_0',
                                  'ACTIVE_SAMPLES_0',
                                  'ACTIVATIONS_0',
                                  'STD_ACTIVATION_DURATIONS_0']

    AXIAL_TIME_FREQ_FEATURE_NAMES = ['MEAN_0', 'MEAN_1', 'MEAN_2',
                                     'STD_0', 'STD_1', 'STD_2',
                                     'MAX_0', 'MAX_1', 'MAX_2',
                                     'RANGE_0', 'RANGE_1', 'RANGE_2', 'DOM_FREQ_0', 'DOM_FREQ_1', 'DOM_FREQ_2', 'FREQ_POWER_RATIO_ABOVE_3DOT5_0', 'FREQ_POWER_RATIO_ABOVE_3DOT5_1', 'FREQ_POWER_RATIO_ABOVE_3DOT5_2', 'DOM_FREQ_POWER_RATIO_0', 'DOM_FREQ_POWER_RATIO_1', 'DOM_FREQ_POWER_RATIO_2', 'ACTIVE_SAMPLES_0', 'ACTIVE_SAMPLES_1', 'ACTIVE_SAMPLES_2',
                                     'ACTIVATIONS_0', 'ACTIVATIONS_1', 'ACTIVATIONS_2',
                                     'STD_ACTIVATION_DURATIONS_0', 'STD_ACTIVATION_DURATIONS_1', 'STD_ACTIVATION_DURATIONS_2']

    ORIENT_FEATURE_NAMES = ["MEDIAN_G_ANGLE_X", "MEDIAN_G_ANGLE_Y",
                            "MEDIAN_G_ANGLE_Z", "RANGE_G_ANGLE_X", "RANGE_G_ANGLE_Y", "RANGE_G_ANGLE_Z"]

    # Prepare parameters
    subwin_samples = subwin_secs * sr

    # Unify input matrix
    X = ext.numpy.atleast_float_2d(raw_df.values[:, 1:4])

    # Smoothing input raw data
    if use_vm:
        feature_names = VM_TIME_FREQ_FEATURE_NAMES + ORIENT_FEATURE_NAMES
        X_vm = accel.vector_magnitude(X)
        X_vm_filtered = ext.numpy.butterworth(
            X_vm, sr=sr, cut_offs=20, order=4, filter_type='low')
    else:
        feature_names = AXIAL_TIME_FREQ_FEATURE_NAMES + ORIENT_FEATURE_NAMES
    X_filtered = ext.numpy.butterworth(X, sr=sr, cut_offs=20,
                                       order=4, filter_type='low')

    if raw_df.shape[0] < sr:
        # TO BE IMPROVED
        # Now input raw data should include no less than 1s data
        for name in feature_names:
            result[name] = [np.nan]
    else:
        time_freq_feature_funcs = [
            accel.mean,
            accel.std,
            accel.max_value,
            accel.max_minus_min,
            functools.partial(accel.spectrum_features,
                              sr=sr, n=1, preset='muss'),
            functools.partial(accel.stats_active_samples,
                              threshold=activation_threshold)
        ]
        orient_feature_funcs = [
            functools.partial(accel.gravity_angle_stats,
                              subwin_samples=subwin_samples, unit=ori_unit)
        ]
        if use_vm:
            X_for_time_freq = X_vm_filtered
        else:
            X_for_time_freq = X_filtered

        for func in time_freq_feature_funcs:
            values, names = func(X_for_time_freq)
        for value, name in zip(values.transpose(), names):
            if name in feature_names:
                result[name] = value.tolist()

        for func in orient_feature_funcs:
            values, names = func(X_filtered)
            for value, name in zip(values.transpose(), names):
                if name in feature_names:
                    result[name] = value.tolist()
    result = pd.DataFrame.from_dict(result)
    return result


class FeatureSet:
    def __init__(self, raw_dfs, placements):
        """
        Raw data from multiple sources
        """
        self._raw_dfs = raw_dfs
        self._placements = placements

    def compute(self, window_size, feature_func, feature_names, **kwargs):
        joint_feature_set = None
        for raw_df, placement in zip(self._raw_dfs, self._placements):
            grouped = raw_df.groupby(pd.Grouper(key='HEADER_TIME_STAMP',
                                                freq="{}ms".format(window_size)))
            feature_set = grouped.apply(feature_func, **kwargs)
            feature_set = FeatureSet._append_placement_suffix(
                feature_set, placement, feature_names)
            if joint_feature_set is None:
                joint_feature_set = feature_set
            else:
                joint_feature_set.merge(
                    feature_set, on=['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'])
        self._feature_set = joint_feature_set
        self._feature_names = list(filter(lambda name: name.split('_')
                                          [-1] in self._placements, joint_feature_set.columns))

    def compute_per_window(self, feature_func, feature_names, **kwargs):
        joint_feature_set = None
        for raw_df, placement in zip(self._raw_dfs, self._placements):
            feature_set = feature_func(raw_df, **kwargs)
            feature_set = FeatureSet._append_placement_suffix(
                feature_set, placement, feature_names)
            if joint_feature_set is None:
                joint_feature_set = feature_set
            else:
                joint_feature_set.merge(
                    feature_set, on=['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'])
        self._feature_set = joint_feature_set
        self._feature_names = list(filter(lambda name: name.split('_')
                                          [-1] in self._placements, joint_feature_set.columns))

    def get_feature_set(self):
        return self._feature_set.copy(deep=True)

    def get_feature_names(self):
        return self._feature_names

    @staticmethod
    def _append_placement_suffix(feature_set, placement, feature_names):
        new_cols = []
        for col in feature_set.columns:
            if col in feature_names and placement != '':
                col = col + '_' + placement
            new_cols.append(col)
        feature_set.columns = new_cols
        return(feature_set)
