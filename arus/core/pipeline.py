"""
Module includes classes that accept single or multiple `arus.core.stream` instances, synchronize data from the streams, process the data using customized processor function, and output using the same iterator interface as `arus.core.stream`.

# Usage of `arus.core.pipeline.Pipeline`

# Single stream case

```python
.. include:: ../../examples/single_stream_pipeline.py
```

# Multiple streams case

```python
.. include:: ../../examples/multi_stream_pipeline.py
```

Author: Qu Tang

Date: 2019-11-15

License: see LICENSE file
"""

from .. import moment
import pathos.pools as ppools
import pathos.helpers as phelpers
import numpy as np
import queue
import pandas as pd
from loguru import logger
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

    def __init__(self, *, max_processes=None, scheduler='processes', preserve_status=False, name='default-pipeline'):
        """

        Args:
            max_processes (int, optional): the max number of sub processes to be spawned. Defaults to 'None'. If it is `None`, the max processes will be the number of cores - 2.
            scheduler (str, optional): scheduler to use, either 'processes' or 'threads'. Defaults to 'processes'.
            preserve_status (bool, optional): whether to preserve the previous input and output in the processor. If `True`, the second and third argument of processor function would be `prev_input` and `prev_output`. Defaults to `False`. When it is `True`, processors will run in sequence, meaning the next processor will start only when the previous one has completed.
            name (str, optional): the name of the pipeline. It will also be used as a prefix for all threads spawned by the class. Defaults to 'default-pipeline'.
        """
        self._started = False
        self._scheduler = scheduler
        self._max_processes = max_processes
        self._process_tasks = queue.Queue()
        self._queue = queue.Queue()
        self._prev_input = queue.Queue(1)
        self._prev_output = queue.Queue(1)
        self.name = name
        self._streams = []
        self._chunks = dict()
        self._stream_pointer = dict()
        self._streams_running = False
        self._process_cond = threading.Condition(threading.Lock())
        self._connected = False
        self._stop_sender = False
        self._preserve_status = preserve_status
        self._process_start_time = None
        self._pool = None

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

    def stop(self):
        return self.finish_tasks_and_stop()

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
            logger.info('Stop pipeline...')
            self._connected = False
            time.sleep(0.2)
            if self._started:
                logger.info('Wait for processing tasks...')
                self._process_tasks.join()
                self._process_tasks.queue.clear()
                time.sleep(0.2)
            self._stop_sender = True
            with self._process_cond:
                self._started = False
                self._process_cond.notify_all()
            time.sleep(0.2)
            if self._pool is not None:
                logger.info('Close processing pool...')
                self._pool.close()
                self._pool.join()
            logger.info('Stop input streams...')
            for stream in self._streams:
                stream.stop()
            logger.info('Clear result queue...')
            with self._queue.mutex:
                self._queue.queue.clear()
            logger.info('Stopped.')
        except Exception as e:
            logger.error(e)
            return False
        return True

    def pause(self):
        """Pause processing the incoming streams, yet pipeline can still receive data from streams. Data will be ignored and not stored.
        """
        with self._process_cond:
            self._started = False
            self._process_start_time = None
        with self._queue.mutex:
            self._queue.queue.clear()

    def _is_running(self):
        if self.started:
            logger.warning(
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
        found = list(filter(lambda s: s._name == stream_name, self._streams))
        return None if len(found) == 0 else found[0]

    def add_stream(self, stream):
        """Add a new stream to the pipeline

        The stream will only be added when the pipeline is stopped.

        Args:
            stream (arus.core.Stream): an instance of arus.core.Stream class.
        """
        if self._is_running():
            return
        if self.get_stream(stream._name) is None:
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
        num_of_processors = min(phelpers.cpu_count() - 2, self._max_processes)
        if num_of_processors == 0:
            self._pool = None
        else:
            if self._pool is None:
                if self._scheduler == 'processes':
                    self._pool = ppools.ProcessPool(nodes=num_of_processors)
                    self._pool.restart(force=True)
                elif self._scheduler == 'threads':
                    self._pool = ppools.ThreadPool(nodes=num_of_processors)
                    self._pool.restart(force=True)
                else:
                    raise NotImplementedError(
                        'This scheduler is not supported: {}'.format(self._scheduler))
            else:
                self._pool.restart(force=True)
        while self._connected:
            if self._started:
                for stream in self._streams:
                    if stream.get_status() == stream.Status.RUN:
                        for data, context, name in stream.generate():
                            st = context['start_time']
                            et = context['stop_time']
                            prev_st = context['prev_start_time']
                            prev_et = context['prev_stop_time']
                            if self._is_data_after_start_time(st):
                                logger.debug(
                                    'recieved data window from ' + name)
                                self._chunks[st.timestamp()] = [] if st.timestamp(
                                ) not in self._chunks else self._chunks[st.timestamp()]
                                self._chunks[st.timestamp()].append(
                                    (data, st, et, prev_st, prev_et, name))
                                self._stream_pointer[name] = st.timestamp()
                            break
                        if self._is_data_after_start_time(st):
                            self._process_synced_chunks(
                                st, et, prev_st, prev_et, self.name)
                        else:
                            logger.debug('Discard one stream window' + str(
                                st) + 'coming before process start time: ' + str(self._process_start_time))
                if np.all([stream.get_status() != stream.Status.RUN for stream in self._streams]):
                    self._streams_running = False
                    break
            else:
                with self._process_cond:
                    # ignore the incoming stream data, the thread will stand by
                    self._process_cond.wait()

    def _send_result(self):
        while not self._stop_sender:
            if self._started:
                try:
                    task, st, et, prev_st, prev_et, name = self._process_tasks.get(
                        block=True, timeout=0.1)
                    if self._max_processes == 0:
                        result = task
                    else:
                        result = task.get()
                        self._process_tasks.task_done()
                    if self._preserve_status:
                        self._prev_output.put(result)
                    logger.info('Sending processed results to queue...')
                    self._put_result_in_queue(
                        (result, st, et, prev_st, prev_et, name))
                except queue.Empty:
                    if not self._streams_running:
                        self._put_result_in_queue(None)
                        break
            else:
                with self._process_cond:
                    # ignore if the pipeline is just connected but not started
                    self._process_cond.wait()

    def _process_synced_chunks(self, st, et, prev_st, prev_et, name):
        if st.timestamp() not in self._chunks:
            return
        chunk_list = self._chunks[st.timestamp()]
        if len(chunk_list) == len(self._streams):
            # this is the last stream chunk for st
            if self._preserve_status:
                if prev_st is None:
                    # the first window
                    prev_input = None
                    prev_output = None
                else:
                    prev_input = self._prev_input.get()
                    prev_output = self._prev_output.get()
                if self._max_processes == 0:
                    result = self._processor(
                        chunk_list, prev_input, prev_output, **self._processor_kwargs)
                    self._process_tasks.put(result)
                    del self._chunks[st.timestamp()]
                else:
                    try:
                        task = self._pool.apipe(
                            self._processor, chunk_list, prev_input, prev_output, **self._processor_kwargs)
                        self._process_tasks.put(
                            (task, st, et, prev_st, prev_et, name))
                        del self._chunks[st.timestamp()]
                    except ValueError as e:
                        logger.error(e)
                        return
                if self._preserve_status:
                    self._prev_input.put(chunk_list)
            else:
                if self._max_processes == 0:
                    result = self._processor(
                        chunk_list, **self._processor_kwargs)
                    self._process_tasks.put(result)
                    del self._chunks[st.timestamp()]
                elif self._pool is not None:
                    try:
                        logger.info('Starting a processing task...')
                        task = self._pool.apipe(self._processor, chunk_list,
                                                **self._processor_kwargs)
                        logger.info('Started a processing task...')
                        self._process_tasks.put(
                            (task, st, et, prev_st, prev_et, name))
                        del self._chunks[st.timestamp()]
                    except Exception as e:
                        logger.error(e)
                        return

    def _put_result_in_queue(self, result):
        self._queue.put(result)

    def _is_data_after_start_time(self, st):
        if self._process_start_time is not None:
            return st.timestamp() >= self._process_start_time.timestamp()
        else:
            return True

    def connect(self, start_time=None):
        """Connect and start the streams but not processing them.

        This function will start the streams one by one on daemon threads, and will start the syncing thread and sender thread as daemons in stand-by status.

        The streams will output data but data will be ignored by the pipeline until `Pipeline.process` got called.
        """
        self._started = False
        self._stop_sender = False
        self._connected = True
        self._sync_thread = threading.Thread(
            target=self._sync_streams, name=self._name + '-sync-streams')
        self._sync_thread.daemon = True
        self._sender_thread = threading.Thread(
            target=self._send_result, name=self._name + '-send-result')
        self._sender_thread.daemon = True
        self._sync_thread.start()
        self._sender_thread.start()
        for stream in self._streams:
            stream.start(start_time=start_time)
        self._streams_running = True
        return self

    def process(self, start_time=None):
        """Start processing the connected incoming stream data.

        Args:
            start_time (str or datetime or np.datetime64 or pd.Timestamp, optional): The start time to start accepting the incoming stream windows. If it is `None`, the pipeline will always output any incoming data without checking `start_time`.
        """
        self._process_start_time = start_time
        if self._process_start_time is not None:
            self._process_start_time = moment.Moment(
                self._process_start_time).to_pandas_timestamp()
        with self._process_cond:
            self._started = True
            self._process_cond.notify_all()
        return True

    def start(self, start_time=None, process_start_time=None):
        """Connect and process the incoming streams in a row together.

        Args:
            start_time (str or datetime or np.datetime64 or pd.Timestamp, optional): When multiple streams are provided, users must provide a valid start_time as a reference to sync between streams. If it is `None`, it will use the first timestamp from the stream data as start_time. Default is None. 
            process_start_time (str or datetime or np.datetime64 or pd.Timestamp, optional): Any stream windows coming before this time will be ignored. If it is `None`, nothing will be ignored. Defaults to None.
        """
        self.connect(start_time=start_time)
        self.process(start_time=process_start_time)

    def get_iterator(self, timeout=None):
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
                try:
                    data = data_queue.get(timeout=timeout)
                    if data is None:
                        # end of the stream, stop
                        raise StopIteration
                    return data
                except queue.Empty:
                    return None, None, None, None, None, None
        return _result_iterator()
