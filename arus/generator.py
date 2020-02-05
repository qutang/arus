"""
generator functions that takes external data source and generate values in buffer_size (number of samples) using Python generator syntax.

Author: Qu Tang
Date: 01/28/2020
License: GNU v3
"""

from .core.libs import mhealth_format as mh
from . import moment
import datetime as dt
import pandas as pd
import numpy as np
import time


class Generator:
    def __init__(self, buffer_size=1800):
        self._buffer_size = buffer_size
        self._buffer = None

    def generate(self):
        pass

    def _buffering(self, data):
        if self._buffer is None and data.shape[0] == self._buffer_size:
            return data
        elif self._buffer is None and data.shape[0] < self._buffer_size:
            self._buffer = data
            return None
        elif self._buffer is not None:
            n = self._buffer_size - self._buffer.shape[0]
            result = pd.concat(
                (self._buffer, data.iloc[:n, :]), axis=0, sort=False)
            self._buffer = data.iloc[n:, :]
            return result


class MhealthSensorFileGenerator(Generator):
    def __init__(self, *filepaths, **kwargs):
        super().__init__(**kwargs)
        self._filepaths = filepaths

    def generate(self):
        for filepath in self._filepaths:
            reader = mh.io.read_data_csv(
                filepath, chunksize=self._buffer_size, iterator=True)
            for data in reader:
                result = self._buffering(data)
                if result is not None:
                    yield result


def generate_from_mhealth_sensor_files(*filepaths, buffer_size=1800):
    buffer = None
    for filepath in filepaths:
        reader = mh.io.read_data_csv(
            filepath, chunksize=buffer_size, iterator=True)
        for data in reader:
            buffer, result = _buffering(buffer, data, buffer_size)
            if result is not None:
                yield result


def generate_from_actigraph_csv_files(*filepaths, buffer_size=1800):
    for filepath in filepaths:
        reader, format_as_mhealth = mh.io.read_actigraph_csv(
            filepath, chunksize=buffer_size, iterator=True)
        for data in reader:
            data = format_as_mhealth(data)
            yield data


def generate_from_mhealth_annotation_files(*filepaths, buffer_size=1800):
    buffer = None
    for filepath in filepaths:
        reader = mh.io.read_data_csv(
            filepath, chunksize=buffer_size, iterator=True)
        for data in reader:
            buffer, result = _buffering(buffer, data, buffer_size)
            if result is not None:
                yield result


def generate_accel_from_normal_distribution(sr, buffer_size=1800, grange=8, start_time=None, sigma=1, max_samples=None):
    buffer_size = int(buffer_size)
    start_time = start_time or dt.datetime.now()
    counter = 0
    max_count = max_samples or 1
    while counter <= max_count:
        clock_start_time = time.time()
        data = np.random.standard_normal(size=(buffer_size, 3)) * sigma
        data[data > grange] = grange
        data[data < -grange] = -grange
        ts = moment.get_pandas_timestamp_sequence(start_time, sr, buffer_size)
        start_time = ts[-1]
        ts = ts[0:-1]
        result = mh.create_accel_dataframe(ts, data)
        delay = buffer_size / sr - (time.time() - clock_start_time)
        time.sleep(delay)
        counter += buffer_size
        if max_samples is None:
            max_count = counter + 1
        yield result


def generate_annotation_from_normal_distribution(duration_mu=5,
                                                 duration_sigma=5,
                                                 start_time=None,
                                                 num_mu=2,
                                                 num_sigma=1,
                                                 labels=['Sitting',
                                                         'Standing', 'Lying'],
                                                 max_samples=0):
    start_time = start_time or dt.datetime.now()
    counter = 0
    max_count = max_samples or 1
    while counter <= max_count:
        N = np.random.poisson(lam=num_mu)
        durations = np.random.standard_normal(
            size=N) * duration_sigma + duration_mu
        start_times = [start_time]
        stop_times = []
        for duration in durations:
            new_start_time = start_time + pd.Timedelta(duration, 'S')
            start_times.append(new_start_time)
            start_time = new_start_time
            stop_times.append(new_start_time)
        start_times = start_times[:-1]
        label_names = np.random.choice(labels, N)
        result = mh.create_annotation_dataframe(
            start_times, stop_times, label_names)
        counter += N
        if max_samples is None:
            max_count = counter + 1
        yield result


def _buffering(buffer, data, buffer_size):
    if buffer is None and data.shape[0] == buffer_size:
        return buffer, data
    elif buffer is None and data.shape[0] < buffer_size:
        buffer = data
        return buffer, None
    elif buffer is not None:
        n = buffer_size - buffer.shape[0]
        result = pd.concat(
            (buffer, data.iloc[:n, :]), axis=0, sort=False)
        buffer = data.iloc[n:, :]
        return buffer, result
