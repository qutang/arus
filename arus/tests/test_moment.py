from .. import moment
import numpy as np
import pandas as pd
import datetime as dt
import tzlocal
import pytz
import time


class TestMoment:
    def test_moment(self):
        test0 = float(1580780607)
        test = test0
        assert moment.Moment(test)._posix == test0
        test = dt.datetime.fromtimestamp(test0)
        assert moment.Moment(test)._posix == test0
        test = np.datetime64(test)
        assert moment.Moment(test)._posix == test0
        test = pd.Timestamp.fromtimestamp(test0)
        assert moment.Moment(test)._posix == test0
        test = pd.Timestamp(test0, unit='s', tz=tzlocal.get_localzone())
        assert moment.Moment(test)._posix == test0

    def test_to(self):
        test0 = float(1580780607)
        test = moment.Moment(test0)
        assert test.to_unix_timestamp() == test0
        assert test.to_datetime() == dt.datetime.fromtimestamp(test0)
        assert test.to_datetime(
            tz=moment.Moment.get_utc_timezone()) == dt.datetime.fromtimestamp(test0, moment.Moment.get_utc_timezone())
        assert test.to_datetime(
            tz=moment.Moment.get_local_timezone()) == dt.datetime.fromtimestamp(test0, moment.Moment.get_local_timezone())
        assert test.to_datetime64() == np.datetime64(dt.datetime.fromtimestamp(test0))
        assert test.to_pandas_timestamp() == pd.Timestamp.fromtimestamp(test0)
        assert test.to_pandas_timestamp(tz=moment.Moment.get_utc_timezone(
        )) == pd.Timestamp(test0, unit='s', tz=moment.Moment.get_utc_timezone())
        assert test.to_string(fmt='%Y') == '2020'
        assert test.to_string(fmt='%Y%z') == '2020'
        assert test.to_string(
            fmt='%Y%z', tz=moment.Moment.get_utc_timezone()) == '2020+0000'

    def test_time_zones(self):
        assert moment.Moment.get_timezone('UTC') == pytz.utc

    def test_get_duration(self):
        ts = time.time()
        assert moment.Moment.get_duration(ts, ts + 1) == 1

        ts = time.time()
        a = [ts] * 10
        b = [ts + 10] * 10
        np.testing.assert_array_equal(
            moment.Moment.get_durations(a, b), 10)

    def test_get_sequence(self):
        st = time.time()
        result = moment.Moment.get_sequence(st, sr=1, N=2)
        assert moment.Moment.get_duration(result[0], result[1]) == 1
        result = moment.Moment.get_sequence(st, sr=100, N=2)
        assert moment.Moment.get_duration(result[0], result[1]) == 1 / 100.0
        result = moment.Moment.get_sequence(st, sr=1000, N=2)
        assert moment.Moment.get_duration(result[0], result[1]) == 1 / 1000.0
        result = moment.Moment.get_sequence(st, sr=1, N=2, format='posix')
        assert moment.Moment.get_duration(result[0], result[1]) == 1
        result = moment.Moment.get_sequence(st, sr=100, N=2, format='posix')
        np.testing.assert_almost_equal(moment.Moment.get_duration(
            result[0], result[1]), 1 / 100.0, decimal=2)
        result = moment.Moment.get_sequence(st, sr=1000, N=2, format='posix')
        np.testing.assert_almost_equal(moment.Moment.get_duration(
            result[0], result[1]), 1 / 1000.0, decimal=2)

        result = moment.Moment.get_sequence(
            st, sr=1, N=2, endpoint_as_extra=False, format='posix')
        assert moment.Moment.get_duration(result[0], result[1]) == 1
        assert len(result) == 2
