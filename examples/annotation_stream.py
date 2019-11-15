from arus.core.stream import AnnotationFileStream
from arus.testing import load_test_data
from glob import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    window_size = 12.8
    files, sr = load_test_data(file_type='mhealth', sensor_type='annotation',
                               file_num='multiple', exception_type='no_missing')
    stream = AnnotationFileStream(
        data_source=files, window_size=window_size, start_time=None, storage_format='mhealth', name='annotation-stream')
    stream.start(scheduler='thread')
    chunk_sizes = []
    for package in stream.get_iterator():
        data = package[0]
        chunk_sizes.append(
            (data.iloc[-1, 2] - data.iloc[0, 1]) / pd.Timedelta(1, 's'))
    chunk_sizes
    pd.Series(chunk_sizes).plot(
        title='chunk sizes of the given stream with \nwindow size of ' + str(window_size) + ' seconds')
    plt.hlines(y=window_size, xmin=0,
               xmax=len(chunk_sizes), linestyles='dashed')
    plt.show()
