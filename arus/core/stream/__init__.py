"""Module includes classes that loads external data sources (e.g., file, network port, socket, user inputs and etc.) into a data queue using a separate thread.

## Usage of `arus.core.stream.SensorFileStream` 

### On mhealth sensor files

```python
.. include:: ../../../examples/mhealth_stream.py
```

### On an Actigraph sensor file

```python
.. include:: ../../../examples/actigraph_stream.py
```

### On mhealth sensor files with real-time delay

```python
.. include:: ../../../examples/sensor_stream_simulated_reality.py
```

## Usage of `arus.core.stream.AnnotationFileStream`

```python
.. include:: ../../../examples/annotation_stream.py
```

Author: Qu Tang

Date: 2019-11-15

License: see LICENSE file
"""

import queue
import threading
import pandas as pd
from ..libs.date import parse_timestamp
from ..libs import mhealth_format as mh
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

    def start(self):
        """Method to start loading data from the provided data source.
        """
        self.started = True
        self._loading_thread = self._get_thread_for_loading(
            self._data_source)
        self._loading_thread.daemon = True
        self._loading_thread.start()

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


class SlidingWindowStream(Stream):
    """Data stream to syncly or asyncly load sensor file or files with different storage formats.

    This class inherits `Stream` class to load data files.

    The stream will load a file or files in the `data_source` and separate them into chunks specified by `window_size` to be loaded in the data queue.

    Examples:
        1. Loading a list of files as 12.8s chunks asynchronously.

        ```python
        .. include:: ../../../examples/mhealth_stream.py
        ```
    """

    def __init__(self, data_source, window_size, start_time=None, buffer_size=1800, simulate_reality=False, start_time_col=0, stop_time_col=0, name='mhealth-stream'):
        """
        Args:
            data_source (str or list): filepath or list of filepaths of mhealth sensor data
            buffer_size (float, optional): the buffer size for file reader in seconds
            simulate_reality (bool, optional): simulate real world time delay if `True`.
            start_time_col (int, optional): the start time column index of the data. Defaults to 0.
            stop_time_col (int, optional): the stop time column index of the data. Defaults to 0.
            name (str, optional): see `Stream.name`.
        """
        super().__init__(data_source=data_source,
                         window_size=window_size, start_time=start_time, name=name)
        self._buffer_size = buffer_size
        self._simulate_reality = simulate_reality
        self._start_time_col = start_time_col
        self._stop_time_col = stop_time_col

    def load_data_source_(self, data_source):
        raise NotImplementedError("Sub class should implement this method and yield loaded data")

    def _extract_chunks(self, data):
        if data.empty:
            return []
        data_et = mh.data.get_end_time(data, self._stop_time_col)
        data_st = mh.data.get_start_time(data, self._start_time_col)
        if self._start_time is None:
            self._start_time = data_st
        window_ts_marks = pd.date_range(start=self._start_time, end=data_et,
                                        freq=str(self._window_size * 1000) + 'ms')
        self._start_time = window_ts_marks[-1]
        chunks = []
        for window_st in window_ts_marks:
            window_et = window_st + \
                pd.Timedelta(self._window_size * 1000, unit='ms')
            chunk = mh.data.segment(
                data, start_time=window_st, stop_time=window_et, start_time_col=self._start_time_col, stop_time_col=self._stop_time_col)
            if chunk.empty:
                chunks.append((chunk, window_st, window_et))
            else:
                chunks.append((chunk, window_st, window_et))
        return chunks

    def _load_data_source_into_chunks(self, data_source):
        current_window = []
        current_window_st = None
        current_window_et = None
        current_clock = time.time()
        previous_window_st = None
        previous_window_et = None
        for data in self.load_data_source_(data_source):
            if self.started:
                chunks = self._extract_chunks(
                    data)
                for chunk, window_st, window_et in chunks:
                    current_window_st = window_st if current_window_st is None else current_window_st
                    current_window_et = window_et if current_window_et is None else current_window_et
                    previous_window_st = window_st if previous_window_st is None else previous_window_st
                    previous_window_et = window_et if previous_window_et is None else previous_window_et
                    if current_window_st == window_st and current_window_et == window_et:
                        current_window.append(chunk)
                    else:
                        current_window = pd.concat(
                            current_window, axis=0, sort=False)
                        current_clock = self._send_data(
                            current_window, current_clock, current_window_st, current_window_et, previous_window_st, previous_window_et)
                        current_window = [chunk]
                        previous_window_st = current_window_st
                        previous_window_et = current_window_et
                        current_window_st = window_st
                        current_window_et = window_et
            else:
                break

    def _send_data(self, current_window, current_clock, current_window_st, current_window_et, previous_window_st, previous_window_et):
        package = (current_window, current_window_st, current_window_et, 
                   previous_window_st, previous_window_et, self.name)
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

    def load_(self, obj_toload):
        self._load_data_source_into_chunks(obj_toload)
        self._put_data_in_queue(None)