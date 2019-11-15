"""Module includes classes that loads external data sources (e.g., file, network port, socket, user inputs and etc.) into a data queue using a separate thread.

## Usage of `arus.core.stream.SensorFileStream` 

### On mhealth sensor files

```python
.. include:: ../../examples/mhealth_stream.py
```

### On an Actigraph sensor file

```python
.. include:: ../../examples/actigraph_stream.py
```

### On mhealth sensor files with real-time delay

```python
.. include:: ../../examples/sensor_stream_simulated_reality.py
```

## Usage of `arus.core.stream.AnnotationFileStream`

```python
.. include:: ../../examples/annotation_stream.py
```

Author: Qu Tang
Date: 2019-11-15

.. include: ../../LICENSE
"""

import queue
import threading
from .libs.mhealth_format.io import read_data_csv
from .libs.mhealth_format.io import read_actigraph_csv
from .libs.mhealth_format import data as mh_data
from .libs.date import parse_timestamp
import pandas as pd
import numpy as np
import logging
import time


class Stream:
    """The base class for data stream

    Stream class is an abstraction of any data source that can be loaded into memory in arbitrary chunk size either asynchronously (currently only support threading) or synchronously.

    Subclass may implement loading mechanisms for different data sources. Such as files, large file, socket device, bluetooth device, remote server, and database.

    Returns:
        stream (Stream): an instance object of type `Stream`.
    """

    def __init__(self, data_source, window_size, start_time=None, name='default-stream', scheduler='thread'):
        """

        Args:
            data_source (object): An object that may be loaded into memory. The type of the object is decided by the implementation of subclass.
            window_size (float): Number of seconds. Each data in the queue would be a short chunk of data lasting `window_size` seconds loaded from the `data_source`.
            start_time (str or datetime or datetime64 or pandas.Timestamp, optional): The start time of data source. This is used to sync between multiple streams. If it is `None`, the default value would be extracted from the first sample of the loaded data.
            name (str, optional): The name of the data stream will also be used as the name of the sub-thread that is used to load data. Defaults to 'default-stream'.
            scheduler (str, optional): The scheduler used to load the data source. It can be either 'thread' or 'sync'. Defaults to 'thread'.
        """
        self._queue = queue.Queue()
        self._data_source = data_source
        self._window_size = window_size
        self._start_time = parse_timestamp(start_time)
        self.started = False
        self.name = name
        self._scheduler = scheduler

    @property
    def started(self):
        """Status of the stream

        Returns:
            started (bool): `True` if stream is running.
        """
        return self._started

    @started.setter
    def started(self, value):
        self._started = value

    @property
    def name(self):
        """The name of the data stream

        Returns:
            name (str): the name of the data stream
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def get_iterator(self):
        """Get a python iterator for the loaded data queue.

        Returns:
            data_queue (iterator): the iterator that can be looped to read loaded data.
        """
        stop_fun = self.stop
        q = self._queue

        class _data_iter:
            def __iter__(self):
                return self

            def __next__(self):
                data = q.get()
                if data is None:
                    # end of the stream, stop
                    stop_fun()
                    raise StopIteration
                return data

        return _data_iter()

    def next(self):
        """Manually get the next loaded data in data queue. Rarely used. Recommend to use the `Stream.get_iterator` method.

        Returns:
            data (object): the loaded data.
        """
        data = self._queue.get()
        if data is None:
            # end of the stream, stop
            self.stop()
        return data

    def start(self, scheduler=None):
        """Method to start loading data from the provided data source.
        Args:
            scheduler (str, optional): The scheduler used to load data. This will override the scheduler set in the constructor if the value is not `None`, otherwise it will fall back to the setting in the constructor.

        Raises:
            NotImplementedError: raised if the scheduler is not supported.
        """
        self.started = True
        self._scheduler = self._scheduler if scheduler is None else scheduler
        if self._scheduler == 'thread':
            self._loading_thread = self._get_thread_for_loading(
                self._data_source)
            self._loading_thread.daemon = True
            self._loading_thread.start()
        elif self._scheduler == 'sync':
            self.load_(self._data_source)
        else:
            raise NotImplementedError(
                'scheduler {} is not implemented'.format(scheduler))

    def _get_thread_for_loading(self, data_source):
        return threading.Thread(
            target=self.load_, name=self.name, args=(data_source,))

    def _put_data_in_queue(self, data):
        self._queue.put(data)

    def stop(self):
        """Method to stop the loading process
        """
        self.started = False

    def load_(self, data_source):
        """Implement this in the sub class.

        You may use `Stream._put_data_in_queue` method to put the loaded data into the queue. Must use `None` as stop signal for the data queue iterator.

        Raises:
            NotImplementedError: Must implement in subclass.
        """
        raise NotImplementedError('Sub class must implement this method')


class SensorFileStream(Stream):
    """Data stream to syncly or asyncly load sensor file or files with different storage formats.

    This class inherits `Stream` class to load data files.

    The stream will load a file or files in the `data_source` and separate them into chunks specified by `window_size` to be loaded in the data queue.

    Examples:
        1. Loading a list of files as 12.8s chunks asynchronously.

        ```python
        .. include:: ../../examples/mhealth_stream.py
        ```
    """

    def __init__(self, data_source, window_size, sr, start_time=None, buffer_size=1800, storage_format='mhealth', simulate_reality=False, name='mhealth-stream'):
        """
        Args:
            data_source (str or list): filepath or list of filepaths of mhealth sensor data
            sr (int): the sampling rate (Hz) for the given data
            buffer_size (float, optional): the buffer size for file reader in seconds
            storage_format (str, optional): the storage format of the files in `data_source`. It now supports `mhealth` and `actigraph`.
            simulate_reality (bool, optional): simulate real world time delay if `True`.
            name (str, optional): see `Stream.name`.
        """
        super().__init__(data_source=data_source,
                         window_size=window_size, start_time=start_time, name=name)
        self._sr = sr
        self._buffer_size = buffer_size
        self._storage_format = storage_format
        self._simulate_reality = simulate_reality

    def _load_files_into_chunks(self, filepaths):
        current_window = []
        current_window_st = None
        current_window_et = None
        current_clock = time.time()
        previous_window_st = None
        for filepath in filepaths:
            for data in self._load_file(filepath):
                if self.started:
                    chunks = self._extract_chunks_from_loaded_data(
                        data)
                    for chunk, window_st, window_et in chunks:
                        current_window_st = window_st if current_window_st is None else current_window_st
                        current_window_et = window_et if current_window_et is None else current_window_et
                        previous_window_st = window_st if previous_window_st is None else previous_window_st
                        if current_window_st == window_st and current_window_et == window_et:
                            current_window.append(chunk)
                        else:
                            current_window = pd.concat(
                                current_window, axis=0, sort=False)
                            current_clock = self._send_data(
                                current_window, current_clock, current_window_st, current_window_et, previous_window_st)
                            current_window = [chunk]
                            previous_window_st = current_window_st
                            current_window_st = window_st
                            current_window_et = window_et

    def _send_data(self, current_window, current_clock, current_window_st, current_window_et, previous_window_st):
        package = (current_window, current_window_st,
                   previous_window_st, self.name)
        if self._simulate_reality:
            delay = (current_window_st - previous_window_st) / \
                np.timedelta64(1, 's')
            logging.debug('Delay for ' + str(delay) +
                          ' seconds to simulate reality')
            time.sleep(max(current_clock + delay - time.time(), 0))
            self._put_data_in_queue(package)
            return time.time()
        else:
            self._put_data_in_queue(package)
            return current_clock

    def _load_file(self, filepath):
        chunksize = self._sr * self._buffer_size
        if self._storage_format == 'mhealth':
            reader = read_data_csv(
                filepath, chunksize=chunksize, iterator=True)
            for data in reader:
                yield data
        elif self._storage_format == 'actigraph':
            reader, format_as_mhealth = read_actigraph_csv(
                filepath, chunksize=chunksize, iterator=True)
            for data in reader:
                data = format_as_mhealth(data)
                yield data
        else:
            raise NotImplementedError(
                'The given storage format argument is not supported')

    def _extract_chunks_from_loaded_data(self, data):
        data_et = mh_data.get_end_time(data, 0)
        data_st = mh_data.get_start_time(data, 0)
        if self._start_time is None:
            self._start_time = data_st
        window_ts_marks = pd.date_range(start=self._start_time, end=data_et,
                                        freq=str(self._window_size * 1000) + 'ms')
        self._start_time = window_ts_marks[-1]
        chunks = []
        for window_st in window_ts_marks:
            window_et = window_st + \
                pd.Timedelta(self._window_size * 1000, unit='ms')
            chunk = mh_data.segment_sensor(
                data, start_time=window_st, stop_time=window_et)
            if chunk.empty:
                continue
            else:
                chunks.append((chunk, window_st, window_et))
        return chunks

    def load_(self, obj_toload):
        if isinstance(obj_toload, str):
            obj_toload = [obj_toload]
        self._load_files_into_chunks(obj_toload)
        self._put_data_in_queue(None)


class SensorGeneratorStream(Stream):
    """Data stream to output randomly simulated sensor data.

    This class inherits `Stream` class to generate simulated sensor data.

    The stream will generate a sensor data stream with the generator function defined in the `data_source` and separate them into chunks specified by `window_size` to be loaded in the data queue.
    """

    def __init__(self, data_source, window_size, sr, start_time=None, simulate_reality=False, name='sensor-generator-stream'):
        """
        Args:
            data_source (dict): a dict that stores a generator function for the simulated sensor data and its kwargs
            sr (int): the sampling rate (Hz) for the given data
            simulate_reality (bool, optional): simulate real world time delay if `True`.
            name (str, optional): see `Stream.name`.
        """
        super().__init__(data_source=data_source,
                         window_size=window_size, start_time=start_time, name=name)
        self._sr = sr
        self._simulate_reality = simulate_reality

    def _load_generator_into_chunks(self, config):
        current_window = []
        current_window_st = None
        current_window_et = None
        current_clock = time.time()
        previous_window_st = None
        generator = config['generator']
        kwargs = config['kwargs']
        for data in generator(sr=self._sr, **kwargs):
            if self.started:
                chunks = self._extract_chunks_from_loaded_data(
                    data)
                for chunk, window_st, window_et in chunks:
                    current_window_st = window_st if current_window_st is None else current_window_st
                    current_window_et = window_et if current_window_et is None else current_window_et
                    previous_window_st = window_st if previous_window_st is None else previous_window_st
                    if current_window_st == window_st and current_window_et == window_et:
                        current_window.append(chunk)
                    else:
                        current_window = pd.concat(
                            current_window, axis=0, sort=False)
                        current_clock = self._send_data(
                            current_window, current_clock, current_window_st, current_window_et, previous_window_st)
                        current_window = [chunk]
                        previous_window_st = current_window_st
                        current_window_st = window_st
                        current_window_et = window_et
            else:
                break

    def _send_data(self, current_window, current_clock, current_window_st, current_window_et, previous_window_st):
        package = (current_window, current_window_st,
                   previous_window_st, self.name)
        if self._simulate_reality:
            delay = (current_window_st - previous_window_st) / \
                np.timedelta64(1, 's')
            logging.debug('Delay for ' + str(delay) +
                          ' seconds to simulate reality')
            time.sleep(max(current_clock + delay - time.time(), 0))
            self._put_data_in_queue(package)
            return time.time()
        else:
            self._put_data_in_queue(package)
            return current_clock

    def _extract_chunks_from_loaded_data(self, data):
        data_et = mh_data.get_end_time(data, 0)
        data_st = mh_data.get_start_time(data, 0)
        if self._start_time is None:
            self._start_time = data_st
        window_ts_marks = pd.date_range(start=self._start_time, end=data_et,
                                        freq=str(self._window_size * 1000) + 'ms')
        self._start_time = window_ts_marks[-1]
        chunks = []
        for window_st in window_ts_marks:
            window_et = window_st + \
                pd.Timedelta(self._window_size * 1000, unit='ms')
            chunk = mh_data.segment_sensor(
                data, start_time=window_st, stop_time=window_et)
            if chunk.empty:
                continue
            else:
                chunks.append((chunk, window_st, window_et))
        return chunks

    def load_(self, obj_toload):
        self._load_generator_into_chunks(obj_toload)
        self._put_data_in_queue(None)


class AnnotationFileStream(Stream):
    """Stream to syncly or asyncly load annotation file or files.

    This class inherits `Stream` class to load annotation files.

    The stream will load a file or files in the `data_source` and separate them into chunks specified by `window_size` to be loaded in the data queue.

    Examples:
        1. Loading a list of files as 12.8s chunks asynchronously.

        ```python
        .. include:: ../../examples/annotation_stream.py
        ```
    """

    def __init__(self, data_source, window_size, start_time=None, storage_format='mhealth', simulate_reality=False, name='mhealth-stream'):
        """
        Args:
            data_source (str or list): filepath or list of filepaths of mhealth annotation data
            storage_format (str, optional): the storage format of the files in `data_source`. It now supports `mhealth`.
            simulate_reality (bool, optional): simulate real world time delay if `True`.
            name (str, optional): see `Stream.name`.
        """
        super().__init__(data_source=data_source,
                         window_size=window_size, start_time=start_time, name=name)
        self._storage_format = storage_format
        self._simulate_reality = simulate_reality

    def _load_file(self, filepath):
        if self._storage_format == 'mhealth':
            data = read_data_csv(
                filepath, chunksize=None, iterator=False)
            yield data
        else:
            raise NotImplementedError(
                'The given storage format argument is not supported')

    def _extract_chunks_from_loaded_data(self, data):
        data_et = mh_data.get_end_time(data, 2)
        data_st = mh_data.get_start_time(data, 1)
        if self._start_time is None:
            self._start_time = data_st
        window_ts_marks = pd.date_range(start=self._start_time, end=data_et,
                                        freq=str(self._window_size * 1000) + 'ms')
        self._start_time = window_ts_marks[-1]
        chunks = []
        for window_st in window_ts_marks:
            window_et = window_st + \
                pd.Timedelta(self._window_size * 1000, unit='ms')
            chunk = mh_data.segment_annotation(
                data, start_time=window_st, stop_time=window_et)
            if chunk.empty:
                chunk = pd.DataFrame(data={'HEADER_TIME_STAMP': [window_st], 'START_TIME': [
                                     window_st], 'STOP_TIME': [window_et], 'LABEL_NAME': ["Unknown"]})
                chunks.append((chunk, window_st, window_et))
            else:
                chunks.append((chunk, window_st, window_et))
        return chunks

    def _send_data(self, current_window, current_clock, current_window_st, current_window_et, previous_window_st):
        package = (current_window, current_window_st,
                   previous_window_st, self.name)
        if self._simulate_reality:
            delay = (current_window_st - previous_window_st) / \
                np.timedelta64(1, 's')
            logging.debug('Delay for ' + str(delay) +
                          ' seconds to simulate reality')
            time.sleep(max(current_clock + delay - time.time(), 0))
            self._put_data_in_queue(package)
            return time.time()
        else:
            self._put_data_in_queue(package)
            return current_clock

    def _load_files_into_chunks(self, filepaths):
        current_window = []
        current_window_st = None
        current_window_et = None
        current_clock = time.time()
        previous_window_st = None
        for filepath in filepaths:
            for data in self._load_file(filepath):
                if self.started:
                    chunks = self._extract_chunks_from_loaded_data(
                        data)
                    for chunk, window_st, window_et in chunks:
                        current_window_st = window_st if current_window_st is None else current_window_st
                        current_window_et = window_et if current_window_et is None else current_window_et
                        previous_window_st = window_st if previous_window_st is None else previous_window_st
                        if current_window_st == window_st and current_window_et == window_et:
                            current_window.append(chunk)
                        else:
                            current_window = pd.concat(
                                current_window, axis=0, sort=False)
                            current_clock = self._send_data(
                                current_window, current_clock, current_window_st, current_window_et, previous_window_st)
                            current_window = [chunk]
                            previous_window_st = current_window_st
                            current_window_st = window_st
                            current_window_et = window_et

    def load_(self, obj_toload):
        if isinstance(obj_toload, str):
            obj_toload = [obj_toload]
        self._load_files_into_chunks(obj_toload)
        self._put_data_in_queue(None)
