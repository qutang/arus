

import pandas as pd
import numpy as np
import pytest
from .. import feature
from .. import mhealth_format as mh
from .. import extensions as ext


@pytest.fixture(scope='module')
def test_data(spades_lab_data):
    sensor_file = spades_lab_data['subjects']['SPADES_1']['sensors']['DW'][0]
    data = pd.read_csv(sensor_file, parse_dates=[0]).iloc[:1000, :]
    st = data.iloc[0, 0]
    et = data.iloc[-1, 0]
    return data, st, et


@pytest.fixture(scope='module')
def test_data_multisources(spades_lab_data):
    dw_sensor_file = spades_lab_data['subjects']['SPADES_1']['sensors']['DW'][0]
    da_sensor_file = spades_lab_data['subjects']['SPADES_1']['sensors']['DA'][0]
    dw_data = pd.read_csv(dw_sensor_file, parse_dates=[0])
    da_data = pd.read_csv(da_sensor_file, parse_dates=[0])
    st = max([dw_data.iloc[0, 0], da_data.iloc[0, 0]])
    dw_data = ext.pandas.segment_by_time(
        dw_data, st, st + pd.Timedelta(12.8 * 5, unit='seconds'))
    da_data = ext.pandas.segment_by_time(
        da_data, st, st + pd.Timedelta(12.8 * 5, unit='seconds'))
    return [dw_data, da_data], ['DW', 'DA']


class TestFeatureSet:
    @ pytest.mark.parametrize('use_vm', [True, False])
    def test_time_freq_orient(self, test_data, use_vm):

        raw_df, st, et = test_data

        feature_vector = feature.time_freq_orient(
            raw_df, sr=80, st=st, et=et, subwin_secs=2, ori_unit='rad', activation_threshold=0.2, use_vm=use_vm)

        if use_vm:
            np.testing.assert_array_equal(feature_vector.columns,
                                          mh.FEATURE_SET_TIMESTAMP_COLS + feature.VM_TIME_FREQ_FEATURE_NAMES + feature.ORIENT_FEATURE_NAMES)

        else:
            np.testing.assert_array_equal(feature_vector.columns,
                                          mh.FEATURE_SET_TIMESTAMP_COLS +
                                          feature.AXIAL_TIME_FREQ_FEATURE_NAMES + feature.ORIENT_FEATURE_NAMES)

        assert feature_vector.shape[0] == 1
        assert feature_vector.notna().all(axis=None)

    def test_preset_muss(self, test_data):
        raw_df, st, et = test_data

        feature_vector = feature.preset(
            featureset_name=feature.PresetFeatureSet.MUSS, raw_df=raw_df, sr=80, st=st, et=et)
        np.testing.assert_array_equal(feature_vector.columns,
                                      mh.FEATURE_SET_TIMESTAMP_COLS + feature.VM_TIME_FREQ_FEATURE_NAMES + feature.ORIENT_FEATURE_NAMES)
        assert feature_vector.shape[0] == 1
        assert feature_vector.notna().all(axis=None)

    @pytest.mark.parametrize('step_size', [None, 6.4])
    @pytest.mark.parametrize('start_time', [None, "2015-09-24 14:17:45.000"])
    @pytest.mark.parametrize('stop_time', [None, "2015-09-24 14:17:00.000"])
    def test_compute_offline(self, test_data_multisources, step_size, start_time, stop_time):
        raw_sources, placements = test_data_multisources
        feat_computer = feature.FeatureSet(raw_sources, placements)
        feat_computer.compute_offline(window_size=12.8,
                                      feature_func=feature.preset, feature_names=feature.preset_names(),
                                      sr=80,
                                      step_size=step_size,
                                      start_time=start_time,
                                      stop_time=stop_time,
                                      featureset_name=feature.PresetFeatureSet.MUSS)
        feature_matrix = feat_computer.get_feature_set()
        feature_names = feat_computer.get_feature_names()

        if stop_time is not None:
            assert feature_matrix is None
            assert feature_names is None
        else:
            assert feature_matrix.shape[1] == 3 + len(feature_names)
            assert len(feature_names) == 2 * \
                len(feature.preset_names(
                    featureset_name=feature.PresetFeatureSet.MUSS))
            if step_size is None and start_time is None and stop_time is None:
                assert feature_matrix.shape[0] == 5
                assert feature_matrix.notna().all(axis=None)
            elif step_size is None and start_time is not None and stop_time is None:
                assert feature_matrix.shape[0] == 6
                assert feature_matrix.notna().all(axis=None)
            elif step_size == 6.4 and start_time is None and stop_time is None:
                assert feature_matrix.shape[0] == 10
                assert feature_matrix.notna().all(axis=None)
            elif step_size == 6.4 and start_time is not None and stop_time is None:
                assert feature_matrix.shape[0] == 11
                assert feature_matrix.notna().all(axis=None)
            elif stop_time is not None:
                assert feature_matrix is None
