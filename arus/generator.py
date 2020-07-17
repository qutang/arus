"""ARUS dataflow: generators.

Generator functions can take external data source and generate values in buffer_size (number of samples) with Python generator pattern.

* Author: Qu Tang
* Date: 01/28/2020
* License: GNU v3
"""

import datetime as dt
import time

import numpy as np
import pandas as pd

from . import mhealth_format as mh
from . import moment, operator


class Generator(operator.Operator):
    """Base generator class.

    Note:
        This class should always be inherited and should not be called directly. Subclasses should override `run` method with its own data source.

    Args:
        buffer_size: The sample size for each burst of the streaming data.

    Example:
        Use generator classes in the following pattern.

        ```python
        # Replace Generator, args, kwargs with proper names for different generator classes
        gen = Generator(*args, **kwargs)
        # Start generator
        gen.start()
        # Get chunked data
        for data, context in gen.get_result():
            # handle data
            if data is None: # Some condition for early termination
                break
        # Stop generator
        gen.stop()
        ```
    """

    def __init__(self, buffer_size: int = 1000000):
        super().__init__()
        self._buffer_size = buffer_size
        self._buffer = None
        self._stop = False

    def start(self):
        """Generate burst of streaming data.

        Use this method instead of `run` when you are using generators directly instead of relying on `arus.Node`.
        """
        self.run()

    def run(self, values=None, src=None, context={}):
        """Implementation of data generation.

        **This method must be overrided by subclasses and developers should implement it to load data from data sources and generate chunks with the data.**


        Examples:
            Developers should implement with the following template.

            ```python
            # You can accept data source from the `__init__` method
            for data in self._load_data(self._data_sources):
                # Call this to buffer input data with `buffer_size`
                result = self._buffering(data)
                # Put the generated data into `self._result`. You should always attach the `self._context` so that it can be chained with other operators via `arus.Node`.
                self._result.put((result, self._context))
                # Implement stop condition
                if data is None or self._stop:
                    break
            ```

        Args:
            values: Not used.
            src: Not used.
            context: Not used.


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

    Note:
        The file paths should be sorted before loading. The generator will load data from the files one by one in order.

    Args:
        *filepaths: The sensor file paths as data sources.
        **kwargs: Other keyword arguments passed to parent class, which is `buffer_size`.

    Examples:
        Generate mhealth sensor data in chunks, with each chunk includes 10 samples.

        ```python
        gen = arus.MhealthSensorFileGenerator("path/to/sensor_file.csv", buffer_size=10)
        gen.start()
        for data, context in gen.get_result():
            print(data.shape[0]) # should be 10
            if data is None: # end condition
                break
        gen.stop()
        ```
    """

    def __init__(self, *filepaths: str, **kwargs: object):
        super().__init__(**kwargs)
        self._filepaths = filepaths

    def run(self, values=None, src=None, context={}):
        """Implementation of data generation (Hidden).
        """
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

    Note:
        The file paths should be sorted before loading. The generator will load data from the files one by one in order.

    Args:
        *filepaths: The annotation file paths as data sources.
        **kwargs: Other keyword arguments passed to parent class, which is `buffer_size`.

    Examples:
        Generate mhealth annotation data in chunks, with each chunk includes 10 rows of annotations.

        ```python
        gen = arus.MhealthAnnotationFileGenerator("path/to/annotation_file.csv", buffer_size=10)
        gen.start()
        for data, context in gen.get_result():
            print(data.shape[0]) # should be 10
            if data is None: # end condition
                break
        gen.stop()
        ```
    """

    def __init__(self, *filepaths, **kwargs):
        super().__init__(**kwargs)
        self._filepaths = filepaths

    def run(self, values=None, src=None, context={}):
        """Implementation of data generation (Hidden).
        """
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
    """Generator class for raw accelerometer data synthesized randomly.

    Args:
        sr: The sampling rate in Hz.
        grange: The dynamic range in g value.
        st: The start timestamp of the generated data. If `None`, it will be the current timestamp.
        sigma: The variance of the generated data sampled from Gaussian Distribution.
        max_samples: The maximum number of samples to be generated. 
        **kwargs: Other keyword arguments passed to parent class, which is `buffer_size`.

    Examples:
        Generate accelerometer data in chunks, with each chunk includes 10 samples for at most 100 samples (10 chunks).

        ```python
        gen = arus.RandomAccelDataGenerator(80, grange=8, st=datetime.datetime.now(), sigma=1.5, max_samples=100, buffer_size=10)
        gen.start()
        for data, context in gen.get_result():
            print(data.shape[0]) # should be 10
            # should end loop after 10 cycles
        gen.stop()
        ```
    """

    def __init__(self, sr: int, grange: int = 8, st: "str, datetime, numpy.datetime64, pandas.Timestamp" = None, sigma: float = 1, max_samples: int = None, **kwargs: object):
        super().__init__(**kwargs)
        self._sr = sr
        self._grange = grange
        self._st = st or dt.datetime.now()
        self._sigma = sigma
        self._max_samples = max_samples
        self._max_count = max_samples or 1

    def run(self, values=None, src=None, context={}):
        """Implementation of data generation (Hidden).
        """
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
    """Generator class for annotation data synthesized randomly.

    Args:
        labels: List of annotation labels to be randomly selected.
        duration_mu: The mean of the Gaussian distribution in seconds used to decide the annotation duration.
        duration_sigma: The standard deviation of the Gaussian distribution in seconds used to decide the annotation duration.
        num_mu: The mean of the Gaussian distribution used to decide the number of annotations for each generation.
        num_sigma: The standard deviation of the Gaussian distribution used to decide the number of annotations for each generation.
        st: The start timestamp of the generated data. If `None`, it will be the current timestamp.
        max_samples: The maximum number of rows of annotations to be generated.
        **kwargs: Other keyword arguments passed to parent class, which is `buffer_size`.

    Examples:
        Generate annotation data in chunks, with each chunk includes 10 samples for at most 100 samples (10 chunks).

        ```python
        gen = arus.RandomAnnotationDataGenerator(['Sit', 'Walk'], duration_mu=5, duration_sigma=5, st=st=datetime.datetime.now(), num_mu=3, num_sigma=1, max_samples=100, buffer_size=10)
        gen.start()
        for data, context in gen.get_result():
            print(data.shape[0]) # should be 10
            # should end loop after 10 cycles
        gen.stop()
        ```
    """

    def __init__(self, labels: list, duration_mu: float = 5, duration_sigma: float = 5, st: "str, datetime, numpy.datetime64, pandas.Timestamp" = None, num_mu: float = 2, num_sigma: float = 1, max_samples: int = None, **kwargs: object):
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
        """Implementation of data generation (Hidden).
        """
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
