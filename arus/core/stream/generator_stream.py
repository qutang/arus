from . import SlidingWindowStream
import time
import pandas as pd
import logging
from ..libs.mhealth_format import data as mh_data
import numpy as np


class GeneratorSlidingWindowStream(SlidingWindowStream):
    """Data stream to output randomly simulated data.

    This class inherits `Stream` class to generate simulated data.

    The stream will generate a data stream with the generator function defined in the `data_source` and separate them into chunks specified by `window_size` to be loaded in the data queue.
    """

    def __init__(self, data_source, window_size, simulate_reality=False, start_time_col=0, stop_time_col=0, name='generator-stream'):
        """
        Args:
            data_source (dict): a dict that stores a generator function for the simulated data and its kwargs
            sr (int): the sampling rate (Hz) for the given data
            simulate_reality (bool, optional): simulate real world time delay if `True`.
            name (str, optional): see `Stream.name`.
        """
        super().__init__(data_source=data_source,
                         window_size=window_size, simulate_reality=simulate_reality, start_time_col=start_time_col, stop_time_col=stop_time_col, name=name)

    def load_data_source_(self, data_source):
        config = data_source
        generator = config['generator']
        kwargs = config['kwargs']
        for data in generator(start_time=self._start_time, **kwargs):
            if self.started:
                self._buffer_data_source(data)
            else:
                break
