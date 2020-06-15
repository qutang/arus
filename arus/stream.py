import queue
import threading
import time
import enum
from loguru import logger
import typing


class Stream:
    """
    The base class for data stream.

    Stream class is an abstraction of any data source that can be loaded into memory in arbitrary chunk size either asynchronously (currently only support threading) or synchronously.
    """

    class Status(enum.Enum):
        """Stream Status codes."""
        NOT_START = enum.auto()
        START = enum.auto()
        RUN = enum.auto()
        STOP = enum.auto()

    def __init__(self, generator: "arus.generator.Generator", segmentor: "arus.segmentor.Segmentor", name: typing.Optional[str] = 'default-stream', scheduler: typing.Optional[str] = 'thread'):
        """

        Arguments:
            generator: a Generator instance that can provide streaming data.
            segmentor: a Segmentor instance that is responsible for segmenting the streaming data into chunks.
            name: The name of the data stream will also be used as the name of the sub-thread that is used to load data. Defaults to 'default-stream'.
            scheduler: The scheduler used to load the data source. It can be either 'thread' or 'sync'. Defaults to 'thread'.
        """
        self._input_buffer = queue.Queue()
        self._output_buffer = queue.Queue()
        self._status = Stream.Status.NOT_START
        self._name = name
        self._scheduler = scheduler
        self._generator = generator
        self._segmentor = segmentor

    def generate(self) -> "pandas.Dataframe":
        """A python generator function to get the segmented streaming data.

        Returns:
            segmented streaming data.
        """
        while True:
            data = self._output_buffer.get()
            if data is None:
                self._status = Stream.Status.STOP
                break
            yield data

    def start(self, start_time: "str, datetime, numpy.datetime64, pandas.Timestamp" = None):
        """Method to start loading data from the provided data source.

        Arguments:
            start_time: The reference time for segmentation.

        Note:
            `start_time` is used to sync between multiple streams. If it is `None`, the default value would be extracted from the first sample of the loaded data.
        """
        self._status = Stream.Status.START
        logger.info('Stream is starting.')
        self._segmentor.set_ref_time(start_time)
        self._loading_thread = self._get_thread_for_loading()
        self._loading_thread.daemon = True
        self._segment_thread = self._get_thread_for_chunking()
        self._segment_thread.daemon = True
        self._loading_thread.start()
        self._segment_thread.start()
        while not self._loading_thread.isAlive() or not self._segment_thread.isAlive():
            time.sleep(0.1)
        logger.info('Stream started.')
        self._status = Stream.Status.RUN

    def stop(self):
        """Stop the loading process."""
        logger.info('Stream is stopping.')
        self._status = Stream.Status.STOP
        self._segmentor.stop()
        self._generator.stop()
        logger.info('Segmentor thread stopped.')
        time.sleep(0.1)
        self._loading_thread.join(timeout=1)
        logger.info('Generator thread stopped.')
        with self._output_buffer.mutex:
            self._output_buffer.queue.clear()
        with self._input_buffer.mutex:
            self._input_buffer.queue.clear()
        self._status = Stream.Status.NOT_START
        logger.info('Stream stopped.')

    def get_status(self) -> "Stream.Status":
        """Get the status code of the stream.

        Returns:
            The status code of the stream.
        """
        return self._status

    def _get_thread_for_loading(self):
        return threading.Thread(
            target=self._generate, name=self._name + '-loading')

    def _get_thread_for_chunking(self):
        return threading.Thread(
            target=self._segment, name=self._name + '-segmenting')

    def _generate(self):
        logger.info('Generator thread started.')
        self._generator.run()
        logger.info('Generator thread is stopping.')

    def _segment(self):
        logger.info('Segmentor thread started.')
        while True:
            try:
                data, _ = next(self._generator.get_result())
                if data is None:
                    break
                for segment in self._segmentor.segment(data):
                    if self._status == Stream.Status.STOP:
                        break
                    self._output_buffer.put(segment + (self._name, ))
            except:
                pass
            finally:
                if self._status == Stream.Status.STOP:
                    break
        self._output_buffer.put(None)
        logger.info('Segmentor thread is stopping.')
