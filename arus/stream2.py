import queue
import threading
import time
import enum
from loguru import logger
import typing

from . import node
from . import operator


class Stream(operator.Operator):
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
        super().__init__()
        self._name = name
        self._generator = node.Node(op=generator, t=node.Node.Type.INPUT,
                                    name=self._name + '-generator')
        self._segmentor = node.Node(op=segmentor, t=node.Node.Type.PIPE,
                                    name=self._name + '-segmentor')
        self.set_essential_context(stream_id=name)

    def run(self, *, values=None, src=None, context={}):
        logger.info('Stream is starting.')
        self._segmentor.get_op().set_ref_time(self._context['ref_start_time'])
        self._generator.get_op().set_context(data_id=self._context['data_id'])
        self._segmentor.start()
        self._generator.start()
        logger.info('Stream started.')

    def set_essential_context(self, start_time=None, stream_id=None):
        self._context['ref_start_time'] = start_time
        self._context['data_id'] = stream_id

    def start(self, start_time: "str, datetime, numpy.datetime64, pandas.Timestamp" = None):
        """Method to start loading data from the provided data source.

        Arguments:
            start_time: The reference time for segmentation.

        Note:
            `start_time` is used to sync between multiple streams. If it is `None`, the default value would be extracted from the first sample of the loaded data.
        """
        self._context['ref_start_time'] = start_time
        self.run()

    def stop(self):
        """Stop the loading process."""
        logger.info('Stream is stopping.')
        self._segmentor.stop()
        logger.info('Segmentor thread stopped.')
        time.sleep(0.1)
        self._generator.stop()
        logger.info('Generator thread stopped.')
        super().stop()
        logger.info('Stream stopped.')

    def get_result(self):
        while True:
            if self._stop:
                break
            data = next(self._generator.produce())
            self._segmentor.consume(data)
            data = next(self._segmentor.produce())
            if data.signal == node.Node.Signal.WAIT:
                pass
            elif data.signal == node.Node.Signal.DATA:
                yield data.values, data.context
            else:
                pass
