"""
generator functions that takes external data source and generate values in buffer_size (number of samples) using Python generator syntax.

Author: Qu Tang

Date: 01/28/2020

License: GNU v3
"""

from . import mhealth_format as mh
from . import moment
from . import operator
import datetime as dt
import pandas as pd
import numpy as np
import time


class Generator(operator.Operator):
    """Abstract class for instances that generate data streams.
    """

    def __init__(self, buffer_size: int = 1800):
        """Create generator instance.

        Arguments:
            buffer_size: the sample size for each burst of the streaming data.
        """
        super().__init__()
        self._buffer_size = buffer_size
        self._buffer = None
        self._stop = False

    def run(self, values=None, src=None, context={}):
        """Generate burst of streaming data.
        """
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
                self._buffer = data.iloc[self._buffer_size:, :]
                return data.iloc[:self._buffer_size, :]
        else:
            n = self._buffer_size - self._buffer.shape[0]
            self._buffer = pd.concat(
                (self._buffer, data.iloc[:n, :]), axis=0, sort=False)
            if self._buffer.shape[0] == self._buffer_size:
                result = self._buffer.copy()
                self._buffer = data.iloc[n:, :]
                return result
            else:
                return None


class MhealthSensorFileGenerator(Generator):
    """Generator class for sensor files stored in mhealth format.
    """

    def __init__(self, *filepaths: str, **kwargs: object):
        """Create MhealthSensorFileGenerator instance.

        Arguments:
            filepaths: the sensor file paths.
            kwargs: other keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._filepaths = filepaths

    def run(self, values=None, src=None, context={}):
        for filepath in self._filepaths:
            reader = mh.MhealthFileReader(filepath)
            reader.read_csv(chunksize=self._buffer_size)
            for data in reader.get_data():
                if self._stop:
                    break
                result = self._buffering(data)
                if result is not None:
                    self._result.put((result, self._context))
            if self._stop:
                break
        self._result.put((None, self._context))


class MhealthAnnotationFileGenerator(Generator):
    """Generator class for annotation files stored in mhealth format.
    """

    def __init__(self, *filepaths, **kwargs):
        """Create MhealthAnnotationFileGenerator instance.

        Arguments:
            filepaths: the sensor file paths.
            kwargs: other keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._filepaths = filepaths

    def run(self, values=None, src=None, context={}):
        for filepath in self._filepaths:
            reader = mh.MhealthFileReader(filepath)
            reader.read_csv(chunksize=self._buffer_size,
                            datetime_cols=[0, 1, 2])
            for data in reader.get_data():
                if self._stop:
                    break
                result = self._buffering(data)
                if result is not None:
                    self._result.put((result, self._context))
            if self._stop:
                break
        self._result.put((None, self._context))


class RandomAccelDataGenerator(Generator):
    """Generate random raw accelerometer data stream.
    """

    def __init__(self, sr: int, grange: int = 8, st: "str, datetime, numpy.datetime64, pandas.Timestamp" = None, sigma: float = 1, max_samples: int = None, **kwargs: object):
        """Create RandomAccelDataGenerator instance.

        Arguments:
            sr: sampling rate in Hz.
            grange: dynamic range in g.
            st: start time of timestamps.
            sigma: the standard deviation of the normal distribution to draw samples.
            max_samples: number of samples to be generated.
            kwargs: other keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._sr = sr
        self._grange = grange
        self._st = st or dt.datetime.now()
        self._sigma = sigma
        self._max_samples = max_samples
        self._max_count = max_samples or 1

    def run(self, values=None, src=None, context={}):
        counter = 0
        while counter <= self._max_count:
            if self._stop:
                break
            data = np.random.standard_normal(
                size=(self._buffer_size, 3)) * self._sigma
            data[data > self._grange] = self._grange
            data[data < -self._grange] = -self._grange
            ts = moment.Moment.get_sequence(
                self._st, self._sr, self._buffer_size, format='pandas')
            self._st = ts[-1]
            ts = ts[0:-1]
            result = pd.DataFrame(
                index=ts, data=data).reset_index(drop=False)
            result.columns = [mh.TIMESTAMP_COL, 'X', 'Y', 'Z']
            time.sleep(0.2)
            counter += self._buffer_size
            if self._max_samples is None:
                self._max_count = counter + 1
            self._result.put((result, self._context))
        self._result.put((None, self._context))


class RandomAnnotationDataGenerator(Generator):
    """Generate random annotation data.
    """

    def __init__(self, labels: list, duration_mu: float = 5, duration_sigma: float = 5, st: "str, datetime, numpy.datetime64, pandas.Timestamp" = None, num_mu: float = 2, num_sigma: float = 1, max_samples: int = None, **kwargs: object):
        """Create RandomAnnotationDataGenerator instance.

        Arguments:
            labels: annotation labels.
            duration_mu: the expected annotation duration.
            duration_sigma: the standard deviation of annotation duration.
            st: start time of timestamps.
            num_mu: the expected number of annotations.
            num_sigma: the standard deviation of number of annotations.
            max_samples: number of samples to be generated.
            kwargs: other keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._labels = labels
        self._duration_mu = duration_mu
        self._duration_sigma = duration_sigma
        self._st = st or dt.datetime.now()
        self._num_mu = num_mu
        self._num_sigma = num_sigma
        self._max_samples = max_samples
        self._max_count = self._max_samples or 1

    def run(self, values=None, src=None, context={}):
        counter = 0
        while counter <= self._max_count:
            if self._stop:
                break
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
            self._result.put((result, self._context))
        self._result.put((None, self._context))
