"""
generator functions that takes external data source and generate values in buffer_size (number of samples) using Python generator syntax.

Author: Qu Tang
Date: 01/28/2020
License: GNU v3
"""

from . import mhealth_format as mh
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

    def stop(self):
        pass

    def _buffering(self, data):
        if self._buffer_size is None:
            return data
        if self._buffer is None:
            if data.shape[0] == self._buffer_size:
                return data
            elif data.shape[0] < self._buffer_size:
                self._buffer = data
                return None
        else:
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
            reader = mh.MhealthFileReader(filepath)
            reader.read_csv(chunksize=self._buffer_size)
            for data in reader.get_data():
                result = self._buffering(data)
                if result is not None:
                    yield result


class MhealthAnnotationFileGenerator(Generator):
    def __init__(self, *filepaths, **kwargs):
        super().__init__(**kwargs)
        self._filepaths = filepaths

    def generate(self):
        for filepath in self._filepaths:
            reader = mh.MhealthFileReader(filepath)
            reader.read_csv(chunksize=self._buffer_size,
                            datetime_cols=[0, 1, 2])
            for data in reader.get_data():
                result = self._buffering(data)
                if result is not None:
                    yield result


class RandomAccelDataGenerator(Generator):
    def __init__(self, sr, grange=8, st=None, sigma=1, max_samples=None, **kwargs):
        super().__init__(**kwargs)
        self._sr = sr
        self._grange = grange
        self._st = st or dt.datetime.now()
        self._sigma = sigma
        self._max_samples = max_samples
        self._max_count = max_samples or 1

    def generate(self):
        counter = 0
        while counter <= self._max_count:
            data = np.random.standard_normal(
                size=(self._buffer_size, 3)) * self._sigma
            data[data > self._grange] = self._grange
            data[data < -self._grange] = -self._grange
            ts = moment.Moment.get_sequence(
                self._st, self._sr, self._buffer_size, format='pandas')
            self._st = ts[-1]
            ts = ts[0:-1]
            result = pd.DataFrame(index=ts, data=data).reset_index(drop=False)
            result.columns = [mh.TIMESTAMP_COL, 'X', 'Y', 'Z']
            time.sleep(0.2)
            counter += self._buffer_size
            if self._max_samples is None:
                self._max_count = counter + 1
            yield result


class RandomAnnotationDataGenerator(Generator):
    def __init__(self, labels, duration_mu=5, duration_sigma=5, st=None, num_mu=2, num_sigma=1, max_samples=None, **kwargs):
        super().__init__(**kwargs)
        self._labels = labels
        self._duration_mu = duration_mu
        self._duration_sigma = duration_sigma
        self._st = st or dt.datetime.now()
        self._num_mu = num_mu
        self._num_sigma = num_sigma
        self._max_samples = max_samples
        self._max_count = self._max_samples or 1

    def generate(self):
        counter = 0
        while counter <= self._max_count:
            N = np.random.poisson(lam=self._num_mu)
            durations = np.random.standard_normal(
                size=N) * self._duration_sigma + self._duration_mu
            start_times = [self._st]
            stop_times = []
            for duration in durations:
                new_start_time = self._st + pd.Timedelta(duration, 'S')
                start_times.append(new_start_time)
                self._st = new_start_time
                stop_times.append(new_start_time)
            start_times = start_times[:-1]
            label_names = np.random.choice(self._labels, N)
            result = pd.DataFrame.from_dict({
                mh.TIMESTAMP_COL: start_times,
                mh.START_TIME_COL: start_times,
                mh.STOP_TIME_COL: stop_times,
                mh.ANNOTATION_LABEL_COL: label_names
            })
            time.sleep(0.2)
            counter += N
            if self._max_samples is None:
                self._max_count = counter + 1
            yield result
