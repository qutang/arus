from ....testing import load_test_data
from ..sensor_stream import SensorFileSlidingWindowStream
import numpy as np
import pandas as pd


def test_SensorFileSlidingWindowStream():
    # with 12.8s window size, 900s buffer size
    window_size = 12.8
    buffer_size = 900
    # single mhealth stream, consistent sampling rate
    files, sr = load_test_data(file_type='mhealth',
                               file_num='single', exception_type='consistent_sr')
    stream = SensorFileSlidingWindowStream(
        data_source=files, window_size=window_size, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='single-mhealth-stream')
    stream.start()
    chunk_sizes = []
    for data, _, _, _, _, name in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
    stream.stop()
    assert np.all(np.array(chunk_sizes[1:-1]) == 1024)

    # multiple mhealth streams, consistent sampling rate
    files, sr = load_test_data(file_type='mhealth',
                               file_num='multiple', exception_type='consistent_sr')
    print(files)
    stream = SensorFileSlidingWindowStream(
        data_source=files, window_size=window_size, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='multiple-mhealth-stream')
    stream.start()
    chunk_sizes = []
    i = 0
    for data, _, _, _, _, name in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
        i = i + 1
        if i >= 10:
            break
    stream.stop()
    assert np.all(np.array(chunk_sizes[1:-1]) == 1024)

    # single mhealth stream, inconsistent sampling rate
    files, sr = load_test_data(file_type='mhealth',
                               file_num='single', exception_type='inconsistent_sr')
    stream = SensorFileSlidingWindowStream(
        data_source=files, window_size=window_size, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='single-mhealth-stream')
    stream.start()
    chunk_sizes = []
    for data, _, _, _, _, name in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
    stream.stop()
    result = np.unique(chunk_sizes, return_counts=True)
    np.testing.assert_array_equal(result[0], np.array(
        [334, 474, 501, 631, 632, 633, 634]))
    np.testing.assert_array_equal(result[1], np.array(
        [1, 1, 1, 11, 87, 46, 11]))

    # # multiple mhealth stream, inconsistent sampling rate
    files, sr = load_test_data(file_type='mhealth',
                               file_num='multiple', exception_type='inconsistent_sr')
    stream = SensorFileSlidingWindowStream(
        data_source=files, window_size=window_size, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='multiple-mhealth-stream')
    stream.start()
    chunk_sizes = []
    for data, _, _, _, _, name in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
    stream.stop()
    result = np.unique(chunk_sizes, return_counts=True)
    np.testing.assert_array_equal(result[0], np.array(
        [243, 450, 594, 640]))
    np.testing.assert_array_equal(result[1], np.array(
        [1, 1,   1,   122]))

    # # very short buffer size
    buffer_size = 10
    files, sr = load_test_data(file_type='mhealth',
                               file_num='single', exception_type='consistent_sr')
    stream = SensorFileSlidingWindowStream(
        data_source=files, window_size=window_size, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='single-mhealth-stream')
    stream.start()
    chunk_sizes = []
    for data, _, _, _, _, name in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
    stream.stop()
    assert np.all(np.array(chunk_sizes[1:-1]) == 1024)

    # # very short window size
    window_size = 2
    buffer_size = 900
    files, sr = load_test_data(file_type='mhealth',
                               file_num='single', exception_type='consistent_sr')
    stream = SensorFileSlidingWindowStream(
        data_source=files, window_size=window_size, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='single-mhealth-stream')
    stream.start()
    chunk_sizes = []
    for data, _, _, _, _, name in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
    stream.stop()
    assert np.all(np.array(chunk_sizes[1:-1]) == 160)
