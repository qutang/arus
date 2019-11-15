from .stream import Stream
from pathos.multiprocessing import ProcessPool
from pathos.helpers import cpu_count
import numpy as np
import queue
import pandas as pd
import logging
import threading
import time


class Pipeline:
    """

    Implementation details:

    Streams are served in threads. Data windows coming from the streams will be synced on a separate thread at first and the synced and merged data windows will be sent to process pools as process tasks for CPU intensive processing asynchronizedly. Another separate thread will wait for the completion of the process tasks and send the result to the result queue.

    This implementation, compared with previous version, makes the data processing most flexible considering that some processing needs to use data from more than one stream. Moreover, the customized process task can spawn other sub processes to further parallize the computational tasks for different combinations of stream data.
    """

    def __init__(self, *, name='default-pipeline'):
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
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def started(self):
        return self._started

    @started.setter
    def started(self, value):
        self._started = value

    def finish_tasks_and_stop(self):
        self.started = False
        self._process_tasks.join()
        self._stop_sender = True
        self._pool.close()
        for stream in self._streams:
            stream.stop()
        self._pool.terminate()

    def _is_running(self):
        if self.started:
            logging.warning(
                'It is not allowed to modify streams while pipeline is running, please stop the pipeline at first')
            return True
        else:
            return False

    def get_stream(self, stream_name):
        found = list(filter(lambda s: s.name == stream_name, self._streams))
        return None if len(found) == 0 else found[0]

    def add_stream(self, stream):
        if self._is_running():
            return
        if self.get_stream(stream.name) is None:
            self._streams.append(stream)

    def set_processor(self, processor, **kwargs):
        self._processor = processor
        self._processor_kwargs = kwargs

    def remove_stream(self, stream_name):
        if self._is_running():
            return
        found_stream = self.get_stream(stream_name)
        if found_stream is not None:
            self._streams.remove(found_stream)

    def _sync_streams(self):
        """

        This method depends on the stream to make sure it will not block for a long time when providing the chunk.

        This method also asssumes that when one window of data is missing, the stream should provide a notification, such as `None` for that window.

        Therefore, in the case when a Bluetooth device disconnects, the stream that serves the data of the device should always output `None` or empty DataFrame for those windows.

        In real time, because stream runs on separate thread, as long as the data is successfully passed to the stream queue, it won't block for a long time.
        """
        num_of_processors = cpu_count() - 2
        self._pool = ProcessPool(nodes=num_of_processors)
        while self.started:
            for stream in self._streams:
                for data, st, prev_st, name in stream.get_iterator():
                    self._chunks[st.timestamp()] = [] if st.timestamp(
                    ) not in self._chunks else self._chunks[st.timestamp()]
                    self._chunks[st.timestamp()].append((data, name))
                    self._stream_pointer[name] = st.timestamp()
                    break
                self._process_synced_chunks(st, name)
        self._pool.close()

    def _send_result(self):
        while not self._stop_sender:
            try:
                task = self._process_tasks.get_nowait()
                result = task.get()
                self._put_result_in_queue(result)
                self._process_tasks.task_done()
            except queue.Empty:
                pass
        self._put_result_in_queue(None)

    def _process_synced_chunks(self, st, name):
        chunk_list = self._chunks[st.timestamp()]
        if len(chunk_list) == len(self._streams):
            # this is the last stream chunk for st
            task = self._pool.apipe(self._processor, chunk_list,
                                    **self._processor_kwargs)
            self._process_tasks.put(task)
            del self._chunks[st.timestamp()]

    def _put_result_in_queue(self, result):
        self._queue.put(result)

    def start(self):
        self._sync_thread = threading.Thread(
            target=self._sync_streams, name=self._name + '-sync-streams')
        self._sync_thread.daemon = True
        self._sender_thread = threading.Thread(
            target=self._send_result, name=self._name + '-send-result')
        self._sender_thread.daemon = True
        self.started = True
        self._sync_thread.start()
        self._sender_thread.start()
        for stream in self._streams:
            stream.start(scheduler='thread')

    def get_iterator(self):
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
