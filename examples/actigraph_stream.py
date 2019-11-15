from arus.core.stream import SensorFileStream
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
    stream = SensorFileStream(
        data_source=files, window_size=window_size, start_time=None, sr=sr, buffer_size=1800, storage_format='actigraph', name='spades_2')
    stream.start(scheduler='thread')
    chunk_sizes = []
    for package in stream.get_iterator():
        data = package[0]
        name = package[3]
        print("{},{},{},{}".format(name,
                                   data.iloc[0, 0], data.iloc[-1, 0], data.shape[0]))
        chunk_sizes.append(data.shape[0])
    pd.Series(chunk_sizes).plot(
        title='chunk sizes of the given stream with \nwindow size of ' + str(window_size) + ' seconds, sampling rate at ' + str(sr) + ' Hz')
    plt.hlines(y=sr * window_size, xmin=0,
               xmax=len(chunk_sizes), linestyles='dashed')
    plt.show()
