"""
Module of util functions to process date and time

Author: Qu Tang

Date: 2020-02-03

License: see LICENSE file
"""

import numpy as np
import pandas as pd
import datetime as dt
import time
import tzlocal
import pytz


class Moment:
    def __init__(self, obj):
        """Date time object that unifies various date time types

        Args:
            obj (int, float, datetime, np.datetime64, pd.Timestamp): If the obj is a datetime, np.datetime64 or pd.Timestamp object and does not have time zone, it will be assumed to be local time.

        Raises:
            NotImplementedError: When input argument type is not recognized.
        """
        if type(obj) == int or type(obj) == float:
            self._posix = float(obj)
        elif type(obj) == str:
            # if it is string, always ignore time zone
            ts = pd.Timestamp(obj).tz_localize(None)
            self._posix = obj.to_pydatetime().timestamp()
        elif type(obj) == dt.datetime:
            # assume local time, always to posix time
            self._posix = obj.timestamp()
        elif type(obj) == np.datetime64:
            # assume local time, so convert to utc first
            local_ts = obj.astype(float) / 10 ** 6
            self._posix = local_ts + Moment.get_local_to_utc_offset(local_ts)
        elif type(obj) == pd.Timestamp:
            self._posix = obj.to_pydatetime().timestamp()
        else:
            raise NotImplementedError('Input type is not supported')

    def to_unix_timestamp(self):
        return self._posix

    def to_pandas_timestamp(self, tz=None):
        if tz is not None:
            return pd.Timestamp(self._posix, unit='s', tz=tz)
        else:
            return pd.Timestamp.fromtimestamp(self._posix)

    def to_datetime(self, tz=None):
        if tz is not None:
            return dt.datetime.fromtimestamp(self._posix, tz=tz)
        else:
            return dt.datetime.fromtimestamp(self._posix)

    def to_datetime64(self):
        return np.datetime64(self.to_datetime())

    def to_string(self, fmt, tz=None):
        obj = self.to_datetime(tz=tz)
        return obj.strftime(fmt)

    @staticmethod
    def get_local_to_utc_offset(ts=None):
        ts = ts or time.time()
        return (dt.datetime.utcfromtimestamp(ts) - dt.datetime.fromtimestamp(ts)
                ).total_seconds()

    @staticmethod
    def get_utc_to_local_offset(ts=None):
        return - Moment.get_local_to_utc_offset(ts=ts)

    @staticmethod
    def get_local_timezone():
        return tzlocal.get_localzone()

    @staticmethod
    def get_utc_timezone():
        return pytz.timezone('UTC')

    @staticmethod
    def get_timezone(obj):
        return pytz.timezone(obj)

    @staticmethod
    def get_duration(st, et, unit='s'):
        st = Moment(st).to_datetime()
        et = Moment(et).to_datetime()
        if unit == 's':
            unit_len = dt.timedelta(seconds=1)
        elif unit == 'ms':
            unit_len = dt.timedelta(milliseconds=1)
        elif unit == 'us':
            unit_len = dt.timedelta(microseconds=1)
        elif unit == 'm':
            unit_len = dt.timedelta(minutes=1)
        return (et - st) / unit_len

    @staticmethod
    def get_durations(st, et, unit='s'):
        return [Moment.get_duration(s, e) for s, e in zip(st, et)]

    @staticmethod
    def transform(iterable):
        return [Moment(item) for item in iterable]

    @staticmethod
    def get_sequence(st, sr, N, endpoint_as_extra=True, tz=None, format='pandas'):
        if endpoint_as_extra:
            N = N + 1
        st = Moment(st).to_unix_timestamp()
        interval = 1 / float(sr)
        span = interval * (N - 1)
        et = st + span
        ts = np.linspace(start=st, stop=et, num=N, endpoint=True).tolist()
        if format == 'pandas':
            ts = [Moment(t).to_pandas_timestamp(tz=tz) for t in ts]
        elif format == 'posix':
            pass
        elif format == 'datetime':
            ts = [Moment(t).to_datetime(tz=tz) for t in ts]
        return ts

    @staticmethod
    def seq_to_unix_timestamp(seq, fmt=None):
        if type(seq[0]) in [int, float, np.float64, np.int64]:
            return seq
        elif type(seq[0]) == str:
            seq = pd.to_datetime(seq, format=fmt)
        elif type(seq[0]) == dt.datetime:
            local_ts = pd.to_datetime(seq).values.astype(float) / 10 ** 9
            local_ts = local_ts + Moment.get_local_to_utc_offset(local_ts[0])
            return local_ts
        elif type(seq[0]) == np.datetime64:
            local_ts = np.array(seq).astype(
                'datetime64[us]').astype(float) / 10 ** 6
            local_ts = local_ts + Moment.get_local_to_utc_offset(local_ts[0])
            return local_ts
        elif type(seq[0]) == pd.Timestamp:
            local_ts = pd.to_datetime(seq).values.astype(float) / 10 ** 9
            local_ts = local_ts + Moment.get_local_to_utc_offset(local_ts[0])
            return local_ts
        else:
            raise NotImplementedError('Input type is not supported')
