""" Module that sends results or intermediate results to external pipe (e.g., file, network port, socket and etc.) using separate thread or process.

Examples:

* Use broadcaster with stream

```python
.. include:: ../../examples/mhealth_broadcaster.py
```
"""

import queue
import threading
from .libs.mhealth_format.io import write_data_csv
from .libs.mhealth_format.logging import display_start_and_stop_time
import logging
from pathos.multiprocessing import ProcessPool
from pathos.helpers import cpu_count
from pathos.helpers import mp as pathos_mp


class Broadcaster:
    """Base class to define a broadcaster that can accept data and broadcast it to external pipes on a separate thread or process via a FIFO queue.

    One may use this base class to implement broadcasters to do various external tasks, such as file saving, sending to server, sending to socket.

    Returns:
        broadcaster (Broadcaster): An instance of Broadcaster class
    """

    def __init__(self, *, name='default-broadcaster'):
        """

        Args:
            name (str, optional): The name of the broadcaster. If using broadcaster in thread mode, this will be the name of the thread. Defaults to 'default-broadcaster'.
        """
        self.name = name
        self._queue = None

    @property
    def name(self):
        """Name of the broadcaster

        Returns:
            name (str): name of the broadcaster.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def send_data(self, data_obj):
        """Call this function to pass any data object to the broadcaster AFTER it starts.

        Args:
            data_obj (object): Can be any type of data. Subclass may restrict the type by overriding it.
        """
        self._queue.put(data_obj)

    def start(self, scheduler='thread', **kwargs):
        """Method to start broadcasting data from the queue.

        This method must be called before `Broadcaster.send_data` method.

        Args:
            scheduler (str, optional): Scheduler to run the broadcasting function in, could be 'thread' or 'process'. Defaults to 'thread'.
            kwargs (dict): keyword arguments that will be passed to the underlying broadcast function
        """
        self._scheduler = scheduler
        if scheduler == 'thread':
            self._broadcasting_runner = self._get_thread_for_broadcasting(
                kwargs)
            self._broadcasting_runner.start()
        elif scheduler == 'process':
            self._broadcasting_runner = self._get_process_for_broadcasting(
                kwargs)
        else:
            raise NotImplementedError("This scheduler is not implemented")

    def wait_to_finish(self):
        """Function to be called after sending all data to the broadcaster and wait for their completion.

        Returns:
            finished (bool): Always `True`.
        """
        self.send_data(None)
        if self._scheduler == 'thread':
            self._broadcasting_runner.join()
        elif self._scheduler == 'process':
            self._broadcasting_runner.get()
        return True

    def _get_thread_for_broadcasting(self, kwargs):
        self._queue = queue.Queue()
        return threading.Thread(
            target=self.broadcast_, name=self.name, kwargs=kwargs)

    def _get_process_for_broadcasting(self, kwargs):
        pool = ProcessPool(nodes=1)
        self._queue = pathos_mp.Manager().Queue()
        kwargs['queue'] = self._queue
        return pool.apipe(self.broadcast_, **kwargs)

    def broadcast_(self, **kwargs):
        """The broadcast function that outputs the received data to an external receiver

        For thread mode, use self._get_iterator() to get the data to be broadcasted.

        For process mode, the data queue can be accessed with name `queue` in the `kwargs` arguments. One can use static method `Broadcaster.make_iterator_from_queue` to create an iterator for this queue.

        Raises:
            NotImplementedError: raised when calling with an instance of the base class.
        """
        raise NotImplementedError('Subclass must implement this method')

    @staticmethod
    def make_iterator_from_queue(q):
        """Static utility method to turn any queue object into an iterator.

        The stop sign for the iterator is always `None`.

        Args:
            q (object): A queue object (could be any object implements Python's `queue` interface).

        Returns:
            iterator (iterator): An iterator wrapping around the queue object
        """
        class _data_iter:
            def __iter__(self):
                return self

            def __next__(self):
                data = q.get()
                if data is None:
                    logging.info('stop getting data from queue')
                    # end of the stream, stop
                    raise StopIteration
                return data
        return _data_iter()

    def _get_iterator(self):
        """Get a python iterator for the broadcasting data queue. This only works for thread mode.

        Returns:
            data_queue (iterator): the iterator that can be looped to process data to be broacasted.
        """
        q = self._queue
        return Broadcaster.make_iterator_from_queue(q)


class MhealthFileBroadcaster(Broadcaster):
    """A broadcaster that saves received data to files in mhealth format. See `arus.libs.mhealth_format.io.write_data_csv` for acceptable `kwargs` when starting the broadcaster.

    Examples:
        * Saving a list of sensor dataframes to mhealth sensor files in thread mode one by one.

        ```python
        list_of_data = [df1, df2, ..., dfn]
        broadcaster = MhealthFileBroadcaster(name='mhealth-sensor-saver')
        broadcaster.start(scheduler='thread', output_folder='./outputs/', pid='Participant_0', file_type='sensor',
                        sensor_or_annotation_type='ASpecialSensor',
                        data_type='AccelerationCalibrated',
                        version_code='NA',
                        sensor_or_annotator_id='SIDXXXXXX',
                        split_hours=True,
                        flat=True)
        for data in list_of_data:
            broadcaster.send_data(data)
        broadcaster.wait_to_finish()
        logging.info('finished')
        ```

        * Saving a list of sensor dataframes to mhealth sensor files in process mode one by one.
        ```python
        list_of_data = [df1, df2, ..., dfn]
        broadcaster = MhealthFileBroadcaster(name='mhealth-sensor-saver')
        broadcaster.start(scheduler='process', output_folder='./outputs/', pid='Participant_0', file_type='sensor',
                        sensor_or_annotation_type='ASpecialSensor',
                        data_type='AccelerationCalibrated',
                        version_code='NA',
                        sensor_or_annotator_id='SIDXXXXXX',
                        split_hours=True,
                        flat=True)
        for data in list_of_data:
            broadcaster.send_data(data)
        broadcaster.wait_to_finish()
        logging.info('finished')
        ```
        """

    def _write_to_file(self, data_obj, output_folder, **kwargs):
        logging.info('Writing {}'.format(
            display_start_and_stop_time(data_obj, file_type='sensor')))
        write_data_csv(data_obj, output_folder, append=True, **kwargs)
        logging.info('Wrote {}'.format(
            display_start_and_stop_time(data_obj, file_type='sensor')))

    def _broadcast_in_thread(self, *, output_folder='./default-mhealth-dataset', **kwargs):
        for data_obj in self._get_iterator():
            self._write_to_file(data_obj, output_folder, **kwargs)
        return 'Finished'

    def _broadcast_in_process(self, *, output_folder='./default-mhealth-dataset', **kwargs):
        q = kwargs['queue']
        del kwargs['queue']
        iterator = Broadcaster.make_iterator_from_queue(q)
        for data_obj in iterator:
            self._write_to_file(data_obj, output_folder, **kwargs)
        return 'Finished'

    def broadcast_(self, *, output_folder='./default-mhealth-dataset', **kwargs):
        """Implementation of the broadcasting mechanism for the class.

        Do not call this method directly!

        Args:
            output_folder (str, optional): The output folder to save the data to be broadcasted. Defaults to './default-mhealth-dataset'.
            kwargs (dict): See `arus.libs.mhealth_format.io.write_data_csv` for all required and acceptable arguments.
        """
        if 'queue' in kwargs:
            # broadcast with process mode
            return self._broadcast_in_process(output_folder=output_folder, **kwargs)
        else:
            return self._broadcast_in_thread(output_folder=output_folder, **kwargs)
