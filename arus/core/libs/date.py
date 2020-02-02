import numpy as np
import pandas as pd
import datetime as dt


def parse_timestamp(ts):
    if type(ts) == np.datetime64:
        result = pd.Timestamp(ts.astype('datetime64[ms]'))
    elif type(ts) == dt.datetime:
        result = pd.Timestamp(np.datetime64(ts, 'ms'))
    elif type(ts) == str:
        result = pd.Timestamp(ts)
    elif type(ts) == int or type(ts) == float:
        result = pd.Timestamp.fromtimestamp(ts)
    else:
        result = ts
    return result


def datetime2unix(ts):
    return (ts - dt.datetime(1970, 1, 1)) / dt.timedelta(seconds=1)


def compute_interval(st, et, unit='s'):
    if unit == 's':
        unit_len = dt.timedelta(seconds=1)
    elif unit == 'ms':
        unit_len = dt.timedelta(milliseconds=1)
    elif unit == 'us':
        unit_len = dt.timedelta(microseconds=1)
    elif unit == 'm':
        unit_len = dt.timedelta(minutes=1)
    return (et - st) / unit_len


def get_timestamp_sequence(st, sr, N):
    N = N + 1
    freq = str(int(1000000 / sr)) + 'us'
    ts = pd.date_range(start=st, periods=N, freq=freq)
    return ts
