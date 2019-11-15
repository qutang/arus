"""
Module includes classes that accept single or multiple `arus.core.stream` instances, synchronize data from the streams, process the data using customized processor function, and output using the same iterator interface as `arus.core.stream`.

## Usage of `arus.core.pipeline.Pipeline`

### Single stream case

```python
.. include: ../../examples/single_stream_pipeline.py
```

### Multiple streams case

```python
.. include: ../../examples/multi_stream_pipeline.py
```

Author: Qu Tang
Date: 2019-11-15

.. include: ../../LICENSE
"""

from .stream import Stream
from pathos.pools import ProcessPool, ThreadPool
from pathos.helpers import cpu_count
import numpy as np
import queue
import pandas as pd
import logging
import threading
import time


class Pipeline:
    """The base class for a pipeline

    Pipeline class is a base class of any processor that accepts multiple `arus.core.stream` instances, sync and process them, and output the processed results.

    The base class provides the functionality to add, and sync multiple streams, as well as the functionality to add customized processor functions.

    Subclass may implement more complex logic such as saving the status of the processed results internally and reuse them with new data from the stream.

    Implementation details:
        Streams are served in threads. Data windows coming from the streams will be synced on a separate thread at first and the synced and merged data windows will be sent to process pools as process tasks for CPU intensive processing asynchronizedly. Another separate thread will wait for the completion of the process tasks and send the result to the result queue.

        This implementation, compared with previous version, makes the data processing most flexible considering that some processing needs to use data from more than one stream. Moreover, the customized process task can spawn other sub processes to further parallize the computational tasks for different combinations of stream data.

    Returns:
        pipeline (Pipeline): an instance object of type `Pipeline`.
    """

    def __init__(self, *, max_processes=None, scheduler='processes', name='default-pipeline'):
        """

        Args:
            max_processes (int, optional): the max number of sub processes to be spawned. Defaults to 'None'. If it is `None`, the max processes will be the number of cores - 2.
            scheduler (str, optional): scheduler to use, either 'processes' or 'threads'. Defaults to 'processes'.
            name (str, optional): the name of the pipeline. It will also be used as a prefix for all threads spawned by the class. Defaults to 'default-pipeline'.
        """
        self._scheduler = scheduler
        self._max_processes = max_processes
        self._process_tasks = queue.Queue()
        self._queue = queue.Queue()
        self.name = name
        self._streams = []
        self._chunks = dict()
        self._stream_pointer = dict()
        self.started = False
        self._stop_sender = False

    @property
    def name(self):
        """`name` property getter

        Returns:
            str: the name of the pipeline
        """
        return self._name

    @name.setter
    def name(self, value):
        """`name` property setter

        Args:
            value (str): the name of the pipeline
        """
        self._name = value

    @property
    def started(self):
        """The status of the pipeline. Getter.

        Returns:
            bool: the status of the pipeline. 'True' means the pipeline is running.
        """
        return self._started

    @started.setter
    def started(self, value):
        """The status of the pipeline. Setter.

        Args:
            value (bool): the status of the pipeline. 'True' means the pipeline is running.
        """
        self._started = value

    def finish_tasks_and_stop(self):
        """Gracefully shutdown the pipeline using the following procedure.

        1. It stops accepting incoming stream data.
        2. It allows sub-processes to finish existing tasks.
        3. It stops the sender thread.
        4. It stops the sub process pool to accept new tasks.
        5. It stops all incoming streams.
        6. It terminates the sub process pool.

        If any exceptions occur during the process, the function will return `False`.

        Returns:
            bool: `True` if the pipeline is shut down correctly.
        """
        try:
            self.started = False
            self._process_tasks.join()
            self._stop_sender = True
            if self._pool is not None:
                self._pool.close()
                self._pool.terminate()
            for stream in self._streams:
                stream.stop()
        except Exception as e:
            return False
        return True

    def _is_running(self):
        if self.started:
            logging.warning(
                'It is not allowed to modify streams while pipeline is running, please stop the pipeline at first')
            return True
        else:
            return False

    def get_stream(self, stream_name):
        """Get the stream instance by its name

        Args:
            stream_name (str): The name of the stream

        Returns:
            arus.core.Stream: The instance of the stream if it is found, otherwise return `None`.
        """
        found = list(filter(lambda s: s.name == stream_name, self._streams))
        return None if len(found) == 0 else found[0]

    def add_stream(self, stream):
        """Add a new stream to the pipeline

        The stream will only be added when the pipeline is stopped.

        Args:
            stream (arus.core.Stream): an instance of arus.core.Stream class.
        """
        if self._is_running():
            return
        if self.get_stream(stream.name) is None:
            self._streams.append(stream)

    def set_processor(self, processor, **kwargs):
        """Set the processor function and its arguments

        The processor function should accept the following arg in the first place:
            chunk_list (list): A list of tuple including chunked data (one window) from all streams. Each tuple includes the following items.
            1. data: the chunked data of the stream
            2. name: the name of the stream

        Args:
            processor (object): A function to process chunks from the streams.
            kwargs (dict): keyword arguments passed to processor.
        """
        self._processor = processor
        self._processor_kwargs = kwargs

    def remove_stream(self, stream_name):
        """Remove stream from the pipeline

        Remove only works when the pipeline is stopped.

        Args:
            stream_name (str): The name of the stream.
        """
        if self._is_running():
            return
        found_stream = self.get_stream(stream_name)
        if found_stream is not None:
            self._streams.remove(found_stream)

    def _sync_streams(self):
        """
        Implementation details:
            This method depends on the stream to make sure it will not block for a long time when providing the chunk.

            This method also asssumes that when one window of data is missing, the stream should provide a notification, such as `None` for that window.

            Therefore, in the case when a Bluetooth device disconnects, the stream that serves the data of the device should always output `None` or empty DataFrame for those windows.

            In real time, because stream runs on separate thread, as long as the data is successfully passed to the stream queue, it won't block for a long time.
        """
        num_of_processors = min(cpu_count() - 2, self._max_processes)
        if num_of_processors == 0:
            self._pool = None
        else:
            if self._scheduler == 'processes':
                self._pool = ProcessPool(nodes=num_of_processors)
            elif self._scheduler == 'threads':
                self._pool = ThreadPool(nodes=num_of_processors)
            else:
                raise NotImplementedError(
                    'This scheduler is not supported: {}'.format(self._scheduler))
        while self.started:
            for stream in self._streams:
                for data, st, prev_st, name in stream.get_iterator():
                    self._chunks[st.timestamp()] = [] if st.timestamp(
                    ) not in self._chunks else self._chunks[st.timestamp()]
                    self._chunks[st.timestamp()].append((data, name))
                    self._stream_pointer[name] = st.timestamp()
                    break
                self._process_synced_chunks(st, name)
        if num_of_processors > 0:
            self._pool.close()

    def _send_result(self):
        while not self._stop_sender:
            try:
                task = self._process_tasks.get(block=False, timeout=1)
                if self._max_processes == 0:
                    result = task
                else:
                    result = task.get()
                    self._process_tasks.task_done()
                self._put_result_in_queue(result)
            except queue.Empty:
                pass
        self._put_result_in_queue(None)

    def _process_synced_chunks(self, st, name):
        chunk_list = self._chunks[st.timestamp()]
        if len(chunk_list) == len(self._streams):
            # this is the last stream chunk for st
            if self._max_processes == 0:
                result = self._processor(chunk_list, **self._processor_kwargs)
                self._process_tasks.put(result)
                del self._chunks[st.timestamp()]
            else:
                try:
                    task = self._pool.apipe(self._processor, chunk_list,
                                            **self._processor_kwargs)
                    self._process_tasks.put(task)
                    del self._chunks[st.timestamp()]
                except ValueError as e:
                    return

    def _put_result_in_queue(self, result):
        self._queue.put(result)

    def start(self):
        """Start the pipeline

        Implementation details:
            The start procedure is as below,
                1. Start a thread for syncing stream chunks as daemon
                2. Start a thread for sending processed results as daemon
                3. Start streams one by one as daemon threads
                3. Set `started` being True
        """
        self._sync_thread = threading.Thread(
            target=self._sync_streams, name=self._name + '-sync-streams')
        self._sync_thread.daemon = True
        self._sender_thread = threading.Thread(
            target=self._send_result, name=self._name + '-send-result')
        self._sender_thread.daemon = True
        self._sync_thread.start()
        self._sender_thread.start()
        for stream in self._streams:
            stream.start(scheduler='thread')
        self.started = True

    def get_iterator(self):
        """Iterator for the processed results

        Raises:
            StopIteration: Raised when iterator ends

        Returns:
            Iteratable: an instance of a python iteratable for processed results
        """
        data_queue = self._queue

        class _result_iterator:
            def __iter__(self):
                return self

            def __next__(self):
                data = data_queue.get()
                if data is None:
                    # end of the stream, stop
                    raise StopIteration
                return data
        return _result_iterator()
