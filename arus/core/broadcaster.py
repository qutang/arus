""" Broadcaster module that sends results or intermediate results to external pipe (e.g., file, network port, socket and etc.) using separate thread
"""

import queue
import threading
from ..libs.mhealth_format.io import write_data_csv
from ..libs.mhealth_format.logging import display_start_and_stop_time
import logging
from pathos.multiprocessing import ProcessPool
from pathos.helpers import cpu_count


class Broadcaster:
    def __init__(self, *, name='default-broadcaster'):
        self.name = name
        self._queue = queue.Queue()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def send_data(self, data_obj):
        self._queue.put(data_obj)

    def start(self, scheduler='thread', **kwargs):
        """Method to start broadcasting data from the queue.

        Args:
            scheduler (str, optional): Scheduler to run the broadcasting function in, could be 'thread' or 'process'. Defaults to 'thread'.
            kwargs (dict): keyword arguments that will be passed to the underlying broadcast function
        """
        if scheduler == 'thread':
            self._broadcasting_runner = self._get_thread_for_broadcasting(
                kwargs)
            self._broadcasting_runner.start()
        elif scheduler == 'process':
            raise NotImplementedError("This scheduler is not implemented")

    def _get_thread_for_broadcasting(self, kwargs):
        return threading.Thread(
            target=self.broadcast_, name=self.name, kwargs=kwargs)

    def broadcast_(self, **kwargs):
        raise NotImplementedError('Subclass must implement this method')

    def _get_iterator(self):
        """Get a python iterator for the broadcasting data queue.

        Returns:
            data_queue (iterator): the iterator that can be looped to process data to be broacasted.
        """
        q = self._queue

        class _data_iter:
            def __iter__(self):
                return self

            def __next__(self):
                data = q.get()
                if data is None:
                    # end of the stream, stop
                    raise StopIteration
                return data

        return _data_iter()


class MhealthFileBroadcaster(Broadcaster):
    def broadcast_(self, *, output_folder='./default-mhealth-dataset', **kwargs):
        for data_obj in self._get_iterator():
            write_data_csv(data_obj, output_folder, append=True, **kwargs)
            logging.info('Wrote {}'.format(
                display_start_and_stop_time(data_obj, file_type='sensor')))
