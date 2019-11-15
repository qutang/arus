from ...testing import load_test_data
from ..stream import SensorFileStream, AnnotationFileStream, SensorGeneratorStream
from ...core.accelerometer import generator
import numpy as np
import pandas as pd


def test_SensorFileStream():
    # with 12.8s window size, 900s buffer size
    window_size = 12.8
    buffer_size = 900
    # single mhealth stream, consistent sampling rate
    files, sr = load_test_data(file_type='mhealth',
                               file_num='single', exception_type='consistent_sr')
    stream = SensorFileStream(
        data_source=files, window_size=window_size, start_time=None, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='single-mhealth-stream')
    stream.start(scheduler='thread')
    chunk_sizes = []
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
    assert np.all(np.array(chunk_sizes[1:-1]) == 1024)

    # multiple mhealth streams, consistent sampling rate
    files, sr = load_test_data(file_type='mhealth',
                               file_num='multiple', exception_type='consistent_sr')
    print(files)
    stream = SensorFileStream(
        data_source=files, window_size=window_size, start_time=None, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='multiple-mhealth-stream')
    stream.start(scheduler='sync')
    chunk_sizes = []
    i = 0
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
        i = i + 1
        if i >= 10:
            break
    stream._put_data_in_queue(None)
    assert np.all(np.array(chunk_sizes[1:-1]) == 1024)

    # single mhealth stream, inconsistent sampling rate
    files, sr = load_test_data(file_type='mhealth',
                               file_num='single', exception_type='inconsistent_sr')
    stream = SensorFileStream(
        data_source=files, window_size=window_size, start_time=None, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='single-mhealth-stream')
    stream.start(scheduler='thread')
    chunk_sizes = []
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
    result = np.unique(chunk_sizes, return_counts=True)
    np.testing.assert_array_equal(result[0], np.array(
        [334, 474, 501, 631, 632, 633, 634]))
    np.testing.assert_array_equal(result[1], np.array(
        [1, 1, 1, 11, 87, 46, 11]))

    # # multiple mhealth stream, inconsistent sampling rate
    files, sr = load_test_data(file_type='mhealth',
                               file_num='multiple', exception_type='inconsistent_sr')
    stream = SensorFileStream(
        data_source=files, window_size=window_size, start_time=None, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='multiple-mhealth-stream')
    stream.start(scheduler='thread')
    chunk_sizes = []
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
    result = np.unique(chunk_sizes, return_counts=True)
    np.testing.assert_array_equal(result[0], np.array(
        [243, 450, 594, 640]))
    np.testing.assert_array_equal(result[1], np.array(
        [1, 1,   1,   122]))

    # # very short buffer size
    buffer_size = 10
    files, sr = load_test_data(file_type='mhealth',
                               file_num='single', exception_type='consistent_sr')
    stream = SensorFileStream(
        data_source=files, window_size=window_size, start_time=None, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='single-mhealth-stream')
    stream.start(scheduler='thread')
    chunk_sizes = []
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
    assert np.all(np.array(chunk_sizes[1:-1]) == 1024)

    # # very short window size
    window_size = 2
    buffer_size = 900
    files, sr = load_test_data(file_type='mhealth',
                               file_num='single', exception_type='consistent_sr')
    stream = SensorFileStream(
        data_source=files, window_size=window_size, start_time=None, sr=sr, buffer_size=buffer_size, storage_format='mhealth', name='single-mhealth-stream')
    stream.start(scheduler='thread')
    chunk_sizes = []
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
    assert np.all(np.array(chunk_sizes[1:-1]) == 160)


def test_SensorGeneratorStream():
    window_size = 12.8
    sr = 50
    config = {
        'generator': generator.normal_dist,
        'kwargs': {
            "grange": 8,
            "start_time": None,
            "buffer_size": 100,
            "sleep_interval": 0,
            "sigma": 1
        }}
    stream = SensorGeneratorStream(
        data_source=config, sr=sr, window_size=window_size, start_time=None)
    stream.start(scheduler='thread')
    chunk_sizes = []
    chunk_means = []
    n = 5
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
        chunk_means.append(np.mean(data.values[:, 1:]))
        n = n - 1
        if n == 0:
            break
    assert np.all(np.array(chunk_sizes[1:-1]) == window_size * sr)
    np.testing.assert_array_almost_equal(
        np.array(chunk_means[1:-1]), 0, decimal=1)


def test_AnnotationFileStream():
    # with 12.8s window size
    window_size = 12.8
    # multiple annotation files, no blank periods
    files, sr = load_test_data(file_type='mhealth', sensor_type='annotation',
                               file_num='multiple', exception_type='no_missing')
    stream = AnnotationFileStream(
        data_source=files, window_size=window_size, start_time=None, storage_format='mhealth', name='annotation-stream')
    stream.start(scheduler='thread')
    chunk_sizes = []
    unknown_labels = 0
    valid_labels = 0
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(
            (data.iloc[-1, 2] - data.iloc[0, 1]) / pd.Timedelta(1, 's'))
        if data.iloc[0, 3] == 'Unknown':
            unknown_labels += 1
        else:
            valid_labels += 1
    result = np.unique(chunk_sizes, return_counts=True)
    assert np.max(result[0]) == 12.8
    assert np.max(result[1]) == 509
    assert unknown_labels == 73
    assert valid_labels == len(chunk_sizes) - unknown_labels
    # multiple annotation files, missing periods
    files, sr = load_test_data(file_type='mhealth', sensor_type='annotation',
                               file_num='multiple', exception_type='missing')
    stream = AnnotationFileStream(
        data_source=files, window_size=window_size, start_time=None, storage_format='mhealth', name='annotation-stream')
    stream.start(scheduler='thread')
    chunk_sizes = []
    unknown_labels = 0
    valid_labels = 0
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(
            (data.iloc[-1, 2] - data.iloc[0, 1]) / pd.Timedelta(1, 's'))
        if data.iloc[0, 3] == 'Unknown':
            unknown_labels += 1
        else:
            valid_labels += 1
    result = np.unique(chunk_sizes, return_counts=True)
    assert np.max(result[0]) == 12.8
    assert np.max(result[1]) == 507
    assert unknown_labels == 85
    assert valid_labels == len(chunk_sizes) - unknown_labels

    # single annotation file, no blank periods
    files, sr = load_test_data(file_type='mhealth', sensor_type='annotation',
                               file_num='single', exception_type='no_missing')
    stream = AnnotationFileStream(
        data_source=files, window_size=window_size, start_time=None, storage_format='mhealth', name='annotation-stream')
    stream.start(scheduler='thread')
    chunk_sizes = []
    unknown_labels = 0
    valid_labels = 0
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(
            (data.iloc[-1, 2] - data.iloc[0, 1]) / pd.Timedelta(1, 's'))
        if data.iloc[0, 3] == 'Unknown':
            unknown_labels += 1
        else:
            valid_labels += 1
    result = np.unique(chunk_sizes, return_counts=True)
    assert np.max(result[0]) == 12.8
    assert np.max(result[1]) == 153
    assert unknown_labels == 16
    assert valid_labels == len(chunk_sizes) - unknown_labels

    # single annotation file, missing periods
    files, sr = load_test_data(file_type='mhealth', sensor_type='annotation',
                               file_num='single', exception_type='missing')
    stream = AnnotationFileStream(
        data_source=files, window_size=window_size, start_time=None, storage_format='mhealth', name='annotation-stream')
    stream.start(scheduler='thread')
    chunk_sizes = []
    unknown_labels = 0
    valid_labels = 0
    for data, _, _, _ in stream.get_iterator():
        chunk_sizes.append(
            (data.iloc[-1, 2] - data.iloc[0, 1]) / pd.Timedelta(1, 's'))
        if data.iloc[0, 3] == 'Unknown':
            unknown_labels += 1
        else:
            valid_labels += 1
    result = np.unique(chunk_sizes, return_counts=True)
    assert np.max(result[0]) == 12.8
    assert np.max(result[1]) == 151
    assert unknown_labels == 22
    assert valid_labels == len(chunk_sizes) - unknown_labels
