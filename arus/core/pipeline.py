from .stream import Stream
from pathos.multiprocessing import ProcessPool
from pathos.helpers import cpu_count
import numpy as np
import queue
import pandas as pd
import logging


class Pipeline:
    def __init__(self, *, name='default-pipeline'):
        self._queue = queue.Queue()
        self.name = name
        self._streams = []
        self.started = False

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

    def get_stream(self, stream_name):
        found = list(filter(lambda s: s.name == stream_name, self._streams))
        return None if len(found) == 0 else found[0]

    def _is_running(self):
        if self.started:
            logging.warning(
                'It is not allowed to modify streams while pipeline is running, please stop the pipeline at first')
            return True
        else:
            return False

    def add_streams(self, *streams):
        if self._is_running():
            return
        for stream in streams:
            if self.get_stream(stream.name) is None:
                self._streams.append(stream)

    def remove_stream(self, stream_name):
        if self._is_running():
            return
        found_stream = self.get_stream(stream_name)
        if found_stream is not None:
            self._streams.remove(found_stream)

    def process_stream(self, stream):
        """Abstract method to be implemented by subclasses to process data in a data stream.

        Args:
            stream (Stream): an instance of data stream

        Raises:
            NotImplementedError: This method must be implemented by subclasses

        Examples:
            1. An example of implementation: computing mean of each data column of each data in the stream

            ```python
            def process_stream(self, stream):
                stream.start()
                all_data = [data.iloc[:, 1:4].mean(axis=0)
                            for data in stream.get_iterator()]
                return pd.concat(all_data, axis=1).transpose()
            ```
        """
        raise NotImplementedError("Subclass must implement this method")

    def process_stream_outputs(self, all_data):
        """Abstract method to be implemented by subclasses to process outputs of `Pipeline.process_stream` from all data streams.

        Results should be put into the queue using method `Pipeline._put_result_in_queue`.

        Args:
            all_data (object): This depends on the concrete implementation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses

        Examples:
            1. An example of implementation: concatenate outputs of each stream by columns

            ```python
            def process_stream_outputs(self, all_data):
                return pd.concat(all_data, axis=1)
            ```
        """
        raise NotImplementedError("Subclass must implement this method")

    def _put_result_in_queue(self, result):
        self._queue.put(result)

    def start(self, scheduler_each='process_pool', scheduler_together='thread', mode='online'):
        if scheduler_each == 'process_pool':
            results = self._run_streams_in_process_pool(self._streams)
        else:
            raise NotImplementedError('This given scheduler is not supported')
        final_result = self.process_stream_outputs(results)
        self._put_result_in_queue(final_result)

    def _run_streams_in_process_pool(self, streams):
        num_of_processors = min(len(streams), cpu_count())
        pool = ProcessPool(nodes=num_of_processors)
        results = pool.map(self.process_stream, streams)
        return results

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
