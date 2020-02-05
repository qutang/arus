import queue
from . import moment
import threading
import time


class Stream:
    """The base class for data stream

    Stream class is an abstraction of any data source that can be loaded into memory in arbitrary chunk size either asynchronously (currently only support threading) or synchronously.

    Subclass may implement loading mechanisms for different data sources. Such as files, large file, socket device, bluetooth device, remote server, and database.

    Returns:
        stream (Stream): an instance object of type `Stream`.
    """

    def __init__(self, generator, segmentor, name='default-stream', scheduler='thread'):
        """

        Args:
            data_source (object): An object that may be loaded into memory. The type of the object is decided by the implementation of subclass.
            window_size (float): Number of seconds. Each data in the queue would be a short chunk of data lasting `window_size` seconds loaded from the `data_source`.
            name (str, optional): The name of the data stream will also be used as the name of the sub-thread that is used to load data. Defaults to 'default-stream'.
            scheduler (str, optional): The scheduler used to load the data source. It can be either 'thread' or 'sync'. Defaults to 'thread'.
        """
        self._input_buffer = queue.Queue()
        self._output_buffer = queue.Queue()
        self._started = False
        self._name = name
        self._scheduler = scheduler
        self._generator = generator
        self._segmentor = segmentor

    def get_iterator(self):
        """Get a python iterator for the loaded data queue.

        Returns:
            data_queue (iterator): the iterator that can be looped to read loaded data.
        """
        stop_fun = self.stop
        buffer = self._output_buffer

        class _data_iter:
            def __iter__(self):
                return self

            def __next__(self):
                data = buffer.get()
                if data is None:
                    # end of the stream, stop
                    raise StopIteration
                return data

        return _data_iter()

    def next(self):
        """Manually get the next loaded data in data queue. Rarely used. Recommend to use the `Stream.get_iterator` method.

        Returns:
            data (object): the loaded data.
        """
        data = self._output_buffer.get()
        if data is None:
            # end of the stream, stop
            self.stop()
        return data

    def start(self, start_time=None):
        """Method to start loading data from the provided data source.

        start_time (str or datetime or datetime64 or pandas.Timestamp, optional): The start time of data source. This is used to sync between multiple streams. If it is `None`, the default value would be extracted from the first sample of the loaded data.
        """
        self._start_time = moment.to_pandas_timestamp(start_time)
        self._started = True
        self._loading_thread = self._get_thread_for_loading()
        self._loading_thread.daemon = True
        self._segment_thread = self._get_thread_for_chunking()
        self._segment_thread.daemon = True
        self._loading_thread.start()
        self._segment_thread.start()

    def _get_thread_for_loading(self):
        return threading.Thread(
            target=self._generate, name=self._name + '-loading')

    def _get_thread_for_chunking(self):
        return threading.Thread(
            target=self._segment, name=self._name + '-segmenting')

    def stop(self):
        """Method to stop the loading process
        """
        self._started = False
        time.sleep(0.1)
        self._segment_thread.join(timeout=1)
        time.sleep(0.1)
        self._loading_thread.join(timeout=1)
        self._segmentor.reset()
        with self._output_buffer.mutex:
            self._output_buffer.queue.clear()
        with self._input_buffer.mutex:
            self._input_buffer.queue.clear()
        self._start_time = None

    def _generate(self):
        for data in self._generator.generate():
            self._input_buffer.put(data)
        self._input_buffer.put(None)

    def _segment(self):
        while self._started:
            try:
                data = self._input_buffer.get(timeout=0.1)
                if data is None:
                    break
                for segment in self._segmentor.segment(data):
                    if not self._started:
                        break
                    self._output_buffer.put(segment + (self._name, ))
            except queue.Empty:
                continue
        self._output_buffer.put(None)