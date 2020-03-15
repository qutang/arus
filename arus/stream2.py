import queue
import threading
import time
import enum
import logging
import typing

from . import o


class Stream:
    """
    The base class for data stream.

    Stream class is an abstraction of any data source that can be loaded into memory in arbitrary chunk size either asynchronously (currently only support threading) or synchronously.
    """

    def __init__(self, generator: "arus.generator.Generator", segmentor: "arus.segmentor.Segmentor", name: typing.Optional[str] = 'default-stream'):
        """

        Arguments:
            generator: a Generator instance that can provide streaming data.
            segmentor: a Segmentor instance that is responsible for segmenting the streaming data into chunks.
            name: The name of the data stream will also be used as the name of the sub-thread that is used to load data. Defaults to 'default-stream'.
        """
        self._name = name
        self._generator = o.O(op=generator, t=o.O.Type.INPUT,
                              name=self._name + '-generator')
        self._segmentor = o.O(op=segmentor, t=o.O.Type.PIPE,
                              name=self._name + '-segmentor')

    def start(self, start_time: "str, datetime, numpy.datetime64, pandas.Timestamp" = None):
        """Method to start loading data from the provided data source.

        Arguments:
            start_time: The reference time for segmentation.

        Note:
            `start_time` is used to sync between multiple streams. If it is `None`, the default value would be extracted from the first sample of the loaded data.
        """
        logging.info('Stream is starting.')
        self._segmentor.get_op().set_ref_time(start_time)
        self._segmentor.start()
        self._generator.start()
        logging.info('Stream started.')

    def stop(self):
        """Stop the loading process."""
        logging.info('Stream is stopping.')
        self._segmentor.stop()
        logging.info('Segmentor thread stopped.')
        time.sleep(0.1)
        self._generator.stop()
        logging.info('Generator thread stopped.')
        logging.info('Stream stopped.')

    def get_result(self):
        while True:
            data = next(self._generator.produce())
            self._segmentor.consume(data)
            data = next(self._segmentor.produce())
            yield data
