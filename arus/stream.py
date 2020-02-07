import queue
import threading
import time
import enum
import logging


class Stream:
    """The base class for data stream

    Stream class is an abstraction of any data source that can be loaded into memory in arbitrary chunk size either asynchronously (currently only support threading) or synchronously.

    Subclass may implement loading mechanisms for different data sources. Such as files, large file, socket device, bluetooth device, remote server, and database.

    Returns:
        stream (Stream): an instance object of type `Stream`.
    """

    class Status(enum.Enum):
        NOT_START = enum.auto()
        START = enum.auto()
        RUN = enum.auto()
        STOP = enum.auto()

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
        self._status = Stream.Status.NOT_START
        self._name = name
        self._scheduler = scheduler
        self._generator = generator
        self._segmentor = segmentor

    def generate(self):
        while True:
            data = self._output_buffer.get()
            if data is None:
                self._status = Stream.Status.STOP
                break
            yield data

    def start(self, start_time=None):
        """Method to start loading data from the provided data source.

        start_time (str or datetime or datetime64 or pandas.Timestamp, optional): The start time of data source. This is used to sync between multiple streams. If it is `None`, the default value would be extracted from the first sample of the loaded data.
        """
        self._status = Stream.Status.START
        logging.info('Stream is starting.')
        self._segmentor.set_ref_time(start_time)
        self._loading_thread = self._get_thread_for_loading()
        self._loading_thread.daemon = True
        self._segment_thread = self._get_thread_for_chunking()
        self._segment_thread.daemon = True
        self._loading_thread.start()
        self._segment_thread.start()
        while not self._loading_thread.isAlive() or not self._segment_thread.isAlive():
            time.sleep(0.1)
        logging.info('Stream started.')
        self._status = Stream.Status.RUN

    def stop(self):
        """Method to stop the loading process
        """
        logging.info('Stream is stopping.')
        self._status = Stream.Status.STOP
        self._segmentor.reset()
        self._generator.stop()
        time.sleep(0.1)
        self._segment_thread.join(timeout=1)
        logging.info('Segmentor thread stopped.')
        time.sleep(0.1)
        self._loading_thread.join(timeout=1)
        logging.info('Generator thread stopped.')
        with self._output_buffer.mutex:
            self._output_buffer.queue.clear()
        with self._input_buffer.mutex:
            self._input_buffer.queue.clear()
        self._status = Stream.Status.NOT_START
        logging.info('Stream stopped.')

    def get_status(self):
        return self._status

    def _get_thread_for_loading(self):
        return threading.Thread(
            target=self._generate, name=self._name + '-loading')

    def _get_thread_for_chunking(self):
        return threading.Thread(
            target=self._segment, name=self._name + '-segmenting')

    def _generate(self):
        logging.info('Generator thread started.')
        for data in self._generator.generate():
            if self._status == Stream.Status.STOP:
                break
            self._input_buffer.put(data)
        self._input_buffer.put(None)
        logging.info('Generator thread is stopping.')

    def _segment(self):
        logging.info('Segmentor thread started.')
        while True:
            try:
                data = self._input_buffer.get(timeout=0.1)
                if data is None:
                    break
                for segment in self._segmentor.segment(data):
                    if self._status == Stream.Status.STOP:
                        break
                    self._output_buffer.put(segment + (self._name, ))
            except queue.Empty:
                pass
            finally:
                if self._status == Stream.Status.STOP:
                    break
        self._output_buffer.put(None)
        logging.info('Segmentor thread is stopping.')
