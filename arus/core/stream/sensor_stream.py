from . import SlidingWindowStream
from ..libs import mhealth_format as mh
import time
import logging


class SensorFileSlidingWindowStream(SlidingWindowStream):
    """Data stream to syncly or asyncly load sensor file or files with different storage formats.

    This class inherits `Stream` class to load data files.

    The stream will load a file or files in the `data_source` and separate them into chunks specified by `window_size` to be loaded in the data queue.

    Examples:
        1. Loading a list of files as 12.8s chunks asynchronously.

        ```python
        .. include:: ../../../examples/mhealth_stream.py
        ```
    """

    def __init__(self, data_source, window_size, sr, buffer_size=1800, storage_format='mhealth', simulate_reality=False, name='mhealth-stream'):
        """
        Args:
            data_source (str or list): filepath or list of filepaths of mhealth sensor data
            sr (int): the sampling rate (Hz) for the given data
            buffer_size (float, optional): the buffer size for file reader in seconds
            storage_format (str, optional): the storage format of the files in `data_source`. It now supports `mhealth` and `actigraph`.
            simulate_reality (bool, optional): simulate real world time delay if `True`.
            name (str, optional): see `Stream.name`.
        """
        super().__init__(data_source=data_source,
                         window_size=window_size, simulate_reality=simulate_reality, start_time_col=0, stop_time_col=0, name=name)
        self._buffer_size = buffer_size
        self._storage_format = storage_format
        self._sr = sr

    def load_data_source_(self, data_source):
        if isinstance(data_source, str):
            data_source = [data_source]
        chunksize = self._sr * self._buffer_size
        for filepath in data_source:
            if self._storage_format == 'mhealth':
                reader = mh.io.read_data_csv(
                    filepath, chunksize=chunksize, iterator=True)
                for data in reader:
                    self._buffer_data_source(data)
            elif self._storage_format == 'actigraph':
                reader, format_as_mhealth = mh.io.read_actigraph_csv(
                    filepath, chunksize=chunksize, iterator=True)
                for data in reader:
                    data = format_as_mhealth(data)
                    self._buffer_data_source(data)
            else:
                raise NotImplementedError(
                    'The given storage format argument is not supported')
        logging.info('Stop loading thread')
        self._buffer_data_source(None)
