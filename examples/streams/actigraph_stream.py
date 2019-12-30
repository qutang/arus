"""
Demonstration of the usage of arus.core.stream.SensorFileSlidingWindowStream
============================================================================

"""

from arus.core.stream.sensor_stream import SensorFileSlidingWindowStream
from arus.testing import load_test_data
from glob import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    window_size = 12.8
    files, sr = load_test_data(file_type='actigraph',
                               file_num='single', exception_type='consistent_sr')
    stream = SensorFileSlidingWindowStream(
        data_source=files, window_size=window_size, sr=sr, buffer_size=1800, storage_format='actigraph', name='spades_2')
    stream.start()
    chunk_sizes = []
    for data, _, _, _, _, name in stream.get_iterator():
        print("{},{},{},{}".format(name,
                                   data.iloc[0, 0], data.iloc[-1, 0], data.shape[0]))
        chunk_sizes.append(data.shape[0])
    stream.stop()
    pd.Series(chunk_sizes).plot(
        title='chunk sizes of the given stream with \nwindow size of ' + str(window_size) + ' seconds, sampling rate at ' + str(sr) + ' Hz')
    plt.hlines(y=sr * window_size, xmin=0,
               xmax=len(chunk_sizes), linestyles='dashed')
    plt.show()