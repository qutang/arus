import queue
import threading
import time
import enum
import logging
import typing

from . import o


class Pipeline(o.BaseOperator):
    """
    The base class for data processing pipeline for HAR.

    Stream class is an abstraction of any data source that can be loaded into memory in arbitrary chunk size either asynchronously (currently only support threading) or synchronously.
    """

    def __init__(self, *streams, synchronizer, processor,  name: typing.Optional[str] = 'default-pipeline'):
        """[summary]

        Arguments:
            synchronizer {[type]} -- [description]
            processor {[type]} -- [description]

        Keyword Arguments:
            name {typing.Optional[str]} -- [description] (default: {'default-pipeline'})
        """
        super().__init__()
        self._name = name
        self._streams = [o.O(op=stream, t=o.O.Type.INPUT,
                             name=self._name + stream._name) for stream in streams]
        self._synchronizer = o.O(op=synchronizer, t=o.O.Type.PIPE,
                                 name=self._name + '-synchronizer')
        self._processor = o.O(op=processor, t=o.O.Type.PIPE,
                              name=self._name + '-processor')

    def run(self, *, values=None, src=None, context={}):
        logging.info('Stream is starting.')
        for stream in self._streams:
            stream.start()
        self._synchronizer.start()
        self._processor.start()
        logging.info('Stream started.')

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
        super().stop()
        """Stop the loading process."""
        logging.info('Stream is stopping.')
        self._processor.stop()
        logging.info('Processor thread stopped.')
        time.sleep(0.1)
        self._synchronizer.stop()
        logging.info('Synchronizer thread stopped.')
        time.sleep(0.1)
        for stream in self._streams:
            stream.stop()
        logging.info('Stream threads stopped.')
        logging.info('Stream stopped.')

    def shutdown(self):
        self._processor.get_op().shutdown()

    def get_result(self):
        while True:
            if self._stop:
                break
            for stream in self._streams:
                data = next(stream.produce())
                self._synchronizer.consume(data)
                data = next(self._synchronizer.produce())
                self._processor.consume(data)
                data = next(self._processor.produce())
                if data.signal == o.O.Signal.WAIT:
                    pass
                elif data.signal == o.O.Signal.DATA:
                    yield data.values, data.context
                else:
                    pass
        yield None, {}
