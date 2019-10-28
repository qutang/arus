"""Module that loads external data sources (e.g., file, network port, socket and etc.) into a data queue using separate thread or not.

Examples:

* Usage of `arus.core.stream.MhealthFileStream`

```python
.. include:: ../../examples/mhealth_stream.py
```

* Usage of `arus.core.stream.ActigraphFileStream`

```python
.. include:: ../../examples/actigraph_stream.py
```
"""

import queue
import threading
from .libs.mhealth_format.io import read_data_csv
from .libs.mhealth_format.io import read_actigraph_csv
from .libs.mhealth_format import data as mh_data
from .libs.mhealth_format.path import extract_file_type
from .libs.date import parse_timestamp
import pandas as pd
import logging


class Stream:
    """The base class for data stream

    Stream class is an abstraction of any data source that can be loaded into memory in arbitrary chunk size either asynchronously (currently only support threading) or synchronously.

    Subclass may implement loading mechanisms for different data sources. Such as files, large file, socket device, bluetooth device, remote server, and database.

    Returns:
        stream (Stream): an instance object of type `Stream`.
    """

    def __init__(self, data_source, chunk_size, start_time=None, name='default-stream', scheduler='thread'):
        """

        Args:
            data_source (object): An object that may be loaded into memory. The type of the object is decided by the implementation of subclass.
            chunk_size (float): Number of seconds. Each data in the queue would be a short chunk of data lasting `chunk_size` seconds loaded from the `data_source`.
            start_time (str or datetime or datetime64 or pandas.Timestamp): The start time of data source. This is used to sync between multiple streams. If it is `None`, the default value would be extracted from the first sample of the loaded data.
            name (str, optional): The name of the data stream will also be used as the name of the sub-thread that is used to load data. Defaults to 'default-stream'.
            scheduler (str, optional): The scheduler used to load the data source. It can be either 'thread' or 'sync'. Defaults to 'thread'.
        """
        self._queue = queue.Queue()
        self._data_source = data_source
        self._chunk_size = chunk_size
        self._start_time = start_time
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


class MhealthFileStream(Stream):
    """Data stream to syncly or asyncly load mhealth sensor file or files.

    This class inherits `Stream` class to load mhealth data files.

    The stream will load a file or files in the `data_source` and separate them into chunks specified by `chunk_size` to be loaded in the data queue.

    Examples:
        1. Loading a list of files as dataframe asynchronously and print out the head of each one.

        ```python
        stream = MhealthFileStream(
            data_source=files, chunk_size="file", sr=80, name='mhealth-stream')
        stream.start(scheduler='thread')
        for data in stream.get_iterator():
            print(data.head())
        ```
    """

    def __init__(self, data_source, chunk_size, sr, start_time=None, name='mhealth-stream'):
        """
        Args:
            data_source (str or list): filepath or list of filepaths of mhealth sensor data
            sr (int): the sampling rate (Hz) for the given data
            name (str, optional): see `Stream.name`.
        """
        super().__init__(data_source=data_source,
                         chunk_size=chunk_size, start_time=start_time, name=name)
        self._sr = sr

    def _load_files_into_chunks(self, filepaths):
        transition_chunk = None
        for filepath in filepaths:
            reader = read_data_csv(
                filepath, chunksize=self._sr * 3600, iterator=True)
            for data in reader:
                if self.started:
                    legacy_chunk, chunks, leftover_chunk = self._extract_chunks_from_loaded_data(
                        data)
                    if transition_chunk is None and legacy_chunk is not None:
                        self._put_data_in_queue(legacy_chunk)
                    elif transition_chunk is not None and legacy_chunk is not None:
                        transition_chunk = pd.concat([
                            transition_chunk, legacy_chunk], axis=0)
                        self._put_data_in_queue(transition_chunk)
                    for chunk in chunks:
                        self._put_data_in_queue(chunk)
                    if leftover_chunk is not None:
                        transition_chunk = leftover_chunk
        self._put_data_in_queue(transition_chunk)

    def _extract_chunks_from_loaded_data(self, data):
        data_et = mh_data.get_end_time(data, 0)
        data_st = mh_data.get_start_time(data, 0)
        if self._start_time is None:
            self._start_time = data_st
        chunk_ts_marks = pd.date_range(start=self._start_time, end=data_et,
                                       freq=str(self._chunk_size * 1000) + 'ms')
        self._start_time = chunk_ts_marks[-1]
        chunks = []
        legacy_chunk = None
        leftover_chunk = None
        for chunk_st in chunk_ts_marks:
            chunk = mh_data.segment_sensor(
                data, start_time=chunk_st, stop_time=chunk_st +
                pd.Timedelta(self._chunk_size * 1000, unit='ms'))
            if chunk.empty:
                continue
            elif mh_data.get_start_time(chunk, 0) + pd.Timedelta(self._chunk_size * 1000, unit='ms') > data_et:
                leftover_chunk = chunk
            elif mh_data.get_end_time(chunk, 0) - pd.Timedelta(self._chunk_size * 1000, unit='ms') < data_st:
                legacy_chunk = chunk
            else:
                chunks.append(chunk)
        return legacy_chunk, chunks, leftover_chunk

    def load_(self, obj_toload):
        if isinstance(obj_toload, str):
            obj_toload = [obj_toload]
        self._load_files_into_chunks(obj_toload)
        self._put_data_in_queue(None)


class ActigraphFileStream(Stream):
    """Data stream to syncly or asyncly load actigraph csv file.

    This class inherits `Stream` class to load actigraph csv file.

    The stream will load the actigraph csv file by chunks (each chunk is about an hour long) into mhealth dataframe. Therefore, you may get a series of one-hour long data from the data queue.

    Examples:
        1. Loading a large actigraph csv file as dataframes asynchronously and print out the head of each one.

        ```python
        stream = ActigraphFileStream(
            data_source=filepath, sr=80, name='actigraph-stream')
        stream.start(scheduler='thread')
        for data in stream.get_iterator():
            print(data.head())
        ```
    """

    def __init__(self, data_source, sr, name='actigraph-stream'):
        """
        Args:
            data_source (str): filepath of an actigraph csv file
            sr (int): the sampling rate (Hz) for the given data
            name (str, optional): see `Stream.name`.
        """
        super().__init__(data_source=data_source, name=name)
        self._sr = sr  # if it is a large file, load one hour data each time

    def _load_large_file(self, filepath):
        reader, format_to_mhealth = read_actigraph_csv(
            filepath, chunksize=self._sr * 3600, iterator=True)
        for data in reader:
            if self.started:
                try:
                    data = format_to_mhealth(data)
                except FileNotFoundError as e:
                    logging.error(e)
                    logging.debug(data)
                    raise FileNotFoundError
                self._put_data_in_queue(data)
            else:
                break
        self._put_data_in_queue(None)

    def load_(self, data_source):
        if isinstance(data_source, str):
            self._load_large_file(data_source)
        else:
            raise NotImplementedError('The data source type is unknown')
        self._put_data_in_queue(None)
