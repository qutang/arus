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
            result = sync.sync(1, st, st + 2, 1, name='test')
            assert result is not None
            np.testing.assert_array_equal(result[0], [1])
            np.testing.assert_array_equal(result[1], [1])
            np.testing.assert_array_equal(list(result[2][0].keys()), ['name'])
            assert result[3].to_unix_timestamp() == st

    def test_multiple_sources(self):
        time0 = time.time()
        times = [time0 + i for i in range(0, 10)]
        sync = synchronizer.Synchronizer()
        sync.add_sources(2)
        # in order one by one alternatively
        for st in times:
            result = sync.sync(1, st, st + 2, 1, name='test1')
            assert result is None
            result = sync.sync(2, st, st + 2, 2, name='test2')
            assert result is not None
            np.testing.assert_array_equal(result[0], [1, 2])
            np.testing.assert_array_equal(result[1], [1, 2])
            assert len(result[2]) == 2
            assert result[3].to_unix_timestamp() == st
        # source 1 adds all at first
        for st in times:
            result = sync.sync(1, st, st + 2, 1, name='test1')
            assert result is None
        for st in times:
            result = sync.sync(2, st, st + 2, 2, name='test2')
            assert result is not None
            np.testing.assert_array_equal(result[0], [1, 2])
            np.testing.assert_array_equal(result[1], [1, 2])
            assert len(result[2]) == 2
            assert result[3].to_unix_timestamp() == st
        # source 2 has different time order
        for st in times:
            result = sync.sync(1, st, st + 2, 1, name='test1')
            assert result is None
        for st in times[5:] + times[:5]:
            result = sync.sync(2, st, st + 2, 2, name='test2')
            assert result is not None
            np.testing.assert_array_equal(result[0], [1, 2])
            np.testing.assert_array_equal(result[1], [1, 2])
            assert len(result[2]) == 2
            assert result[3].to_unix_timestamp() == st

    def test_dynamic_added_sources(self):
        time0 = time.time()
        times = [time0 + i for i in range(0, 10)]
        sync = synchronizer.Synchronizer()
        sync.add_source()
        # in order one by one alternatively
        for st in times[:5]:
            result = sync.sync(1, st, st + 2, 1, name='test1')
            assert result is not None
            np.testing.assert_array_equal(result[0], [1])
            np.testing.assert_array_equal(result[1], [1])
            np.testing.assert_array_equal(list(result[2][0].keys()), ['name'])
            assert result[3].to_unix_timestamp() == st
        sync.add_source()
        # source 1 adds all at first
        for st in times[5:8]:
            result = sync.sync(1, st, st + 2, 1, name='test1')
            assert result is None
        for st in times[5:8]:
            result = sync.sync(2, st, st + 2, 2, name='test2')
            assert result is not None
            np.testing.assert_array_equal(result[0], [1, 2])
            np.testing.assert_array_equal(result[1], [1, 2])
            assert len(result[2]) == 2
            assert result[3].to_unix_timestamp() == st
        # source 2 has different time order
        for st in times[8:]:
            result = sync.sync(1, st, st + 2, 1, name='test1')
            assert result is None
            result = sync.sync(2, st, st + 2, 2, name='test2')
            assert result is not None
            np.testing.assert_array_equal(result[0], [1, 2])
            np.testing.assert_array_equal(result[1], [1, 2])
            assert len(result[2]) == 2
            assert result[3].to_unix_timestamp() == st
