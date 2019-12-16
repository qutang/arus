from . import SlidingWindowStream
from ..libs import mhealth_format as mh

class AnnotationFileSlidingWindowStream(SlidingWindowStream):
    """Stream to syncly or asyncly load annotation file or files.

    This class inherits `Stream` class to load annotation files.

    The stream will load a file or files in the `data_source` and separate them into chunks specified by `window_size` to be loaded in the data queue.

    Examples:
        1. Loading a list of files as 12.8s chunks asynchronously.

        ```python
        .. include:: ../../../examples/annotation_stream.py
        ```
    """

    def __init__(self, data_source, window_size, start_time=None, storage_format='mhealth', simulate_reality=False, name='mhealth-annotation-stream'):
        """
        Args:
            data_source (str or list): filepath or list of filepaths of mhealth annotation data
            storage_format (str, optional): the storage format of the files in `data_source`. It now supports `mhealth`.
            simulate_reality (bool, optional): simulate real world time delay if `True`.
            name (str, optional): see `Stream.name`.
        """
        super().__init__(data_source=data_source, 
                         window_size=window_size, start_time=start_time, buffer_size=None, simulate_reality=simulate_reality, start_time_col=1, stop_time_col=2, name=name)
        self._storage_format = storage_format

    def load_data_source_(self, data_source):
        if isinstance(data_source, str):
            data_source = [data_source]
        for filepath in data_source:
            if self._storage_format == 'mhealth':
                data = mh.io.read_data_csv(
                    filepath, chunksize=None, iterator=False)
                yield data
            else:
                raise NotImplementedError(
                    'The given storage format argument is not supported')