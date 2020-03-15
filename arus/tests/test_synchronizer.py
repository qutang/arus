import pytest
from .. import synchronizer
import time
import numpy as np


class TestSynchronizer:
    def test_single_source(self):
        time0 = time.time()
        times = [time0 + i for i in range(0, 5)]
        sync = synchronizer.Synchronizer()
        sync.add_source()

        for st in times:
            sync.run(1, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test'})
        sts = []
        for data, context in sync.get_result():
            assert data is not None
            np.testing.assert_array_equal(data[0], [1])
            assert 'data_ids' in context
            sts.append(context['start_time'])
        np.testing.assert_array_equal(sts, times)

    def test_multiple_sources(self):
        time0 = time.time()
        times = [time0 + i for i in range(0, 10)]
        sync = synchronizer.Synchronizer()
        sync.add_sources(2)
        # in order one by one alternatively
        for st in times:
            sync.run(1, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test1'})
            sync.run(2, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test2'})
        sts = []
        for data, context in sync.get_result():
            np.testing.assert_array_equal(data, [1, 2])
            assert 'data_ids' in context
            sts.append(context['start_time'])
        np.testing.assert_array_equal(sts, times)

        # source 1 adds all at first
        for st in times:
            sync.run(1, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test1'})
        for st in times:
            sync.run(2, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test2'})
        sts = []
        for data, context in sync.get_result():
            np.testing.assert_array_equal(data, [1, 2])
            assert 'data_ids' in context
            sts.append(context['start_time'])
        np.testing.assert_array_equal(sts, times)

        # source 2 has different time order
        for st in times:
            sync.run(1, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test1'})
        for st in times[5:] + times[:5]:
            sync.run(2, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test2'})
        sts = []
        for data, context in sync.get_result():
            np.testing.assert_array_equal(data, [1, 2])
            assert 'data_ids' in context
            sts.append(context['start_time'])
        np.testing.assert_array_equal(sts, times[5:] + times[:5])

    def test_dynamic_added_sources(self):
        time0 = time.time()
        times = [time0 + i for i in range(0, 10)]
        sync = synchronizer.Synchronizer()
        sync.add_source()
        # in order one by one alternatively
        for st in times[:5]:
            sync.run(1, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test1'})
        sts = []
        for data, context in sync.get_result():
            np.testing.assert_array_equal(data, [1])
            assert 'data_ids' in context
            sts.append(context['start_time'])
        np.testing.assert_array_equal(sts, times[:5])

        sync.add_source()
        # source 1 adds all at first
        for st in times[5:8]:
            sync.run(1, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test1'})
        for st in times[5:8]:
            sync.run(2, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test2'})
        sts = []
        for data, context in sync.get_result():
            np.testing.assert_array_equal(data, [1, 2])
            assert 'data_ids' in context
            sts.append(context['start_time'])
        np.testing.assert_array_equal(sts, times[5:8])
        # source 2 has different time order
        for st in times[8:]:
            sync.run(1, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test1'})
            sync.run(2, src=None, context={
                'start_time': st, 'stop_time': st + 2, 'data_id': 'test2'})
        sts = []
        for data, context in sync.get_result():
            np.testing.assert_array_equal(data, [1, 2])
            assert 'data_ids' in context
            sts.append(context['start_time'])
        np.testing.assert_array_equal(sts, times[8:])
