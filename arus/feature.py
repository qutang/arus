"""
Compute common features for activity recognition
"""

import enum
import functools
import sys

import numpy as np
import pandas as pd
from alive_progress import alive_bar
from loguru import logger

from . import accelerometer as accel
from . import extensions as ext
from . import mhealth_format as mh
from .error_code import ErrorCode

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


class PresetFeatureSet(enum.Enum):
    MUSS = enum.auto()


def preset_names(featureset_name=PresetFeatureSet.MUSS):
    if featureset_name == PresetFeatureSet.MUSS:
        return VM_TIME_FREQ_FEATURE_NAMES + ORIENT_FEATURE_NAMES


def preset(raw_df, sr, st, et, featureset_name=PresetFeatureSet.MUSS):
    if featureset_name == PresetFeatureSet.MUSS:
        return time_freq_orient(raw_df, sr, st, et, subwin_secs=2,
                                ori_unit='rad', activation_threshold=0.2, use_vm=True)


def time_freq_orient(raw_df, sr, st, et, subwin_secs=2, ori_unit='rad', activation_threshold=0.2, use_vm=True):

    result = {
        mh.TIMESTAMP_COL: [st],
        mh.START_TIME_COL: [st],
        mh.STOP_TIME_COL: [et]
    }

    # Prepare parameters
    subwin_samples = subwin_secs * sr

    # Unify input matrix
    X = ext.numpy.atleast_float_2d(raw_df.values[:, 1:4])
    # fill nan at first, nan will be filled by spline interpolation
    X = ext.numpy.mutate_nan(X)

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

    if raw_df.dropna().shape[0] < raw_df.shape[0] * 0.9:
        # TO BE IMPROVED
        # Now input raw data should include no less than 90% of the whole window
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
    def __init__(self, raw_sources, placements, srs=None):
        """
        Raw data from multiple sources
        """
        self._raw_sources = raw_sources
        self._placements = placements
        self._srs = srs
        self.reset()

    def _validate_input_as_df(self):
        if len(self._raw_sources) == 0:
            return
        if type(self._raw_sources[0]) is not pd.DataFrame:
            logger.error(
                '[Error code: {ErrorCode.INPUT_ARGUMENT_FORMAT_ERROR.name}] To compute features offline, the input raw data should be feature dataframe stored in mhealth format.')
            sys.exit(ErrorCode.INPUT_ARGUMENT_FORMAT_ERROR.name)

    def compute_offline(self, window_size, feature_func, feature_names, start_time=None, stop_time=None, step_size=None, **kwargs):
        self._validate_input_as_df()
        step_size = step_size or window_size
        joint_feature_set = None
        window_start_markers = ext.pandas.split_into_windows(
            *self._raw_sources, step_size=step_size, st=start_time, et=stop_time)
        feature_sets = []
        if self._srs is None:
            self._srs = [kwargs['sr'] for p in self._placements]
        else:
            self._srs = [sr or kwargs['sr'] for sr in self._srs]
        kwargs.pop('sr', None)
        for raw_df, placement, sr in zip(self._raw_sources, self._placements, self._srs):
            feature_vector_list = []
            with alive_bar(len(window_start_markers), bar='blocks') as bar:
                for window_st in window_start_markers:
                    window_et = window_st + pd.Timedelta(window_size, unit='s')
                    df = ext.pandas.segment_by_time(
                        raw_df, seg_st=window_st, seg_et=window_et)
                    if df.dropna().empty:
                        bar(f'Skipped features for nan window: {window_st}')
                        continue
                    if df.empty:
                        bar(f'Skipped features for empty window: {window_st}')
                        continue
                    fv = feature_func(df, st=window_st,
                                      et=window_et, sr=sr, **kwargs)
                    feature_vector_list.append(fv)
                    bar(f'Computed features for window: {window_st}')
            if len(feature_vector_list) == 0:
                continue
            feature_set = pd.concat(
                feature_vector_list, axis=0, ignore_index=True, sort=False)
            feature_sets.append(feature_set)

        if len(feature_sets) == 0:
            self._feature_set = None
            self._feature_names = None
        else:
            self._feature_set, self._feature_names = ext.pandas.merge_all(
                *feature_sets,
                suffix_names=self._placements,
                suffix_cols=feature_names,
                on=mh.FEATURE_SET_TIMESTAMP_COLS,
                how='inner',
                sort=False
            )

    def compute_per_window(self, feature_func, feature_names, **kwargs):
        joint_feature_set = None
        self._validate_input_as_df()
        if self._srs is None:
            self._srs = [kwargs['sr'] for p in self._placements]
        else:
            self._srs = [sr or kwargs['sr'] for sr in self._srs]
        kwargs.pop('sr', None)
        for raw_df, placement, sr in zip(self._raw_sources, self._placements, self._srs):
            feature_set = feature_func(raw_df, sr=sr, **kwargs)
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
        if self._feature_set is not None:
            return self._feature_set.copy(deep=True)
        else:
            return None

    def get_feature_names(self):
        return self._feature_names

    def reset(self):
        self._feature_set = None
        self._feature_names = None

    @staticmethod
    def _append_placement_suffix(feature_set, placement, feature_names):
        new_cols = []
        for col in feature_set.columns:
            if col in feature_names and placement != '':
                col = col + '_' + placement
            new_cols.append(col)
        feature_set.columns = new_cols
        return(feature_set)
