

import pandas as pd
import numpy as np
import pytest
from .. import class_label
from .. import mhealth_format as mh
from .. import extensions as ext
from .. import spades_lab as slab
from .. import extensions as ext


@pytest.fixture(scope='module')
def test_data(spades_lab_data):
    annotation_file = spades_lab_data['subjects']['SPADES_1']['annotations']['SPADESInLab'][0]
    data = pd.read_csv(annotation_file, parse_dates=[
                       0, 1, 2], infer_datetime_format=True)
    data = ext.pandas.segment_by_time(
        data, seg_st=data.iloc[0, 1], seg_et=data.iloc[0, 1] + pd.Timedelta(12.8, 's'), st_col=1, et_col=2)
    st = data.iloc[0, 1]
    et = data.iloc[-1, 2]
    return data, 'SPADES_1', st, et


@pytest.fixture(scope='module')
def test_data_multiwindows(spades_lab_data):
    annotation_file = spades_lab_data['subjects']['SPADES_1']['annotations']['SPADESInLab'][0]
    data = pd.read_csv(annotation_file, parse_dates=[
                       0, 1, 2], infer_datetime_format=True)
    data = ext.pandas.segment_by_time(
        data, seg_st=data.iloc[0, 1], seg_et=data.iloc[0, 1] + pd.Timedelta(12.8 * 5, 's'), st_col=1, et_col=2)
    st = data.iloc[0, 1]
    et = st + pd.Timedelta(12.8 * 5, 's')
    return data, 'SPADES_1', st, et


class TestClassSet:
    @ pytest.mark.parametrize('task_names', [['ACTIVITY_VALIDATED'], ['POSTURE_VALIDATED', 'INTENSITY']])
    def test_slab_class_set(self, test_data, task_names):

        raw_df, pid, st, et = test_data

        class_vector = slab.class_set(
            raw_df, st=st, et=et, task_names=task_names, pid=pid)

        np.testing.assert_array_equal(class_vector.columns,
                                      mh.FEATURE_SET_TIMESTAMP_COLS + task_names)

        assert class_vector.shape[0] == 1
        assert class_vector.notna().all(axis=None)

    @pytest.mark.parametrize('step_size', [None, 6.4])
    @pytest.mark.parametrize('start_time', [None, "2015-09-24 14:35:00.000"])
    @pytest.mark.parametrize('stop_time', [None, "2015-09-24 14:17:00.000"])
    @pytest.mark.parametrize('task_names', [['ACTIVITY_VALIDATED'], ['POSTURE_VALIDATED', 'INTENSITY']])
    def test_compute_offline(self, test_data_multiwindows, step_size, start_time, stop_time, task_names):
        raw_df, pid, st, et = test_data_multiwindows
        class_computer = class_label.ClassSet([raw_df], ['SPADESInLab'])
        class_computer.compute_offline(window_size=12.8,
                                       class_func=slab.class_set, task_names=task_names,
                                       step_size=step_size,
                                       start_time=start_time or st,
                                       stop_time=stop_time or et,
                                       pid=pid)
        class_set = class_computer.get_class_set()
        names = class_computer.get_task_names()

        if stop_time is not None:
            assert class_set is None
            np.testing.assert_array_equal(names, task_names)
        else:
            assert class_set.shape[1] == 3 + len(task_names)
            if step_size is None and start_time is None and stop_time is None:
                assert class_set.shape[0] == 5
                assert class_set.notna().all(axis=None)
            elif step_size is None and start_time is not None and stop_time is None:
                assert class_set.shape[0] == 6
                assert class_set.notna().all(axis=None)
            elif step_size == 6.4 and start_time is None and stop_time is None:
                assert class_set.shape[0] == 10
                assert class_set.notna().all(axis=None)
            elif step_size == 6.4 and start_time is not None and stop_time is None:
                assert class_set.shape[0] == 12
                assert class_set.notna().all(axis=None)
            elif stop_time is not None:
                assert class_set is None
