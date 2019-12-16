from ....testing import load_test_data
from ..annotation_stream import AnnotationFileSlidingWindowStream
import numpy as np
import pandas as pd

def test_AnnotationFileSlidingWindowStream():
    # with 12.8s window size
    window_size = 12.8
    # multiple annotation files, no blank periods
    files, sr = load_test_data(file_type='mhealth', sensor_type='annotation',
                               file_num='multiple', exception_type='no_missing')
    stream = AnnotationFileSlidingWindowStream(
        data_source=files, window_size=window_size, start_time=None, storage_format='mhealth', name='annotation-stream')
    stream.start()
    chunk_sizes = []
    unknown_labels = 0
    valid_labels = 0
    for data, _, _, _, _, name in stream.get_iterator():
        if data.empty:
            unknown_labels += 1
        else:
            chunk_sizes.append(
                (data.iloc[-1, 2] - data.iloc[0, 1]) / pd.Timedelta(1, 's'))
            if data.iloc[0, 3] == 'Unknown':
                unknown_labels += 1
            else:
                valid_labels += 1
    result = np.unique(chunk_sizes, return_counts=True)
    assert np.max(result[0]) == 12.8
    assert np.max(result[1]) == 436
    assert unknown_labels == 73
    # multiple annotation files, missing periods
    files, sr = load_test_data(file_type='mhealth', sensor_type='annotation',
                               file_num='multiple', exception_type='missing')
    stream = AnnotationFileSlidingWindowStream(
        data_source=files, window_size=window_size, start_time=None, storage_format='mhealth', name='annotation-stream')
    stream.start()
    chunk_sizes = []
    unknown_labels = 0
    valid_labels = 0
    
    for data, _, _, _, _, name in stream.get_iterator():
        if data.empty:
            unknown_labels += 1
        else:
            chunk_sizes.append(
                (data.iloc[-1, 2] - data.iloc[0, 1]) / pd.Timedelta(1, 's'))
            if data.iloc[0, 3] == 'Unknown':
                unknown_labels += 1
            else:
                valid_labels += 1
    
    result = np.unique(chunk_sizes, return_counts=True)
    assert np.max(result[0]) == 12.8
    assert np.max(result[1]) == 422
    assert unknown_labels == 85

    # single annotation file, no blank periods
    files, sr = load_test_data(file_type='mhealth', sensor_type='annotation',
                               file_num='single', exception_type='no_missing')
    stream = AnnotationFileSlidingWindowStream(
        data_source=files, window_size=window_size, start_time=None, storage_format='mhealth', name='annotation-stream')
    stream.start()
    chunk_sizes = []
    unknown_labels = 0
    valid_labels = 0
    for data, _, _, _, _, name in stream.get_iterator():
        if data.empty:
            unknown_labels += 1
        else:
            chunk_sizes.append(
                (data.iloc[-1, 2] - data.iloc[0, 1]) / pd.Timedelta(1, 's'))
            if data.iloc[0, 3] == 'Unknown':
                unknown_labels += 1
            else:
                valid_labels += 1
    result = np.unique(chunk_sizes, return_counts=True)
    assert np.max(result[0]) == 12.8
    assert np.max(result[1]) == 137
    assert unknown_labels == 16

    # single annotation file, missing periods
    files, sr = load_test_data(file_type='mhealth', sensor_type='annotation',
                               file_num='single', exception_type='missing')
    stream = AnnotationFileSlidingWindowStream(
        data_source=files, window_size=window_size, start_time=None, storage_format='mhealth', name='annotation-stream')
    stream.start()
    chunk_sizes = []
    unknown_labels = 0
    valid_labels = 0
    for data, _, _, _, _, name in stream.get_iterator():
        if data.empty:
            unknown_labels += 1
        else:
            chunk_sizes.append(
                (data.iloc[-1, 2] - data.iloc[0, 1]) / pd.Timedelta(1, 's'))
            if data.iloc[0, 3] == 'Unknown':
                unknown_labels += 1
            else:
                valid_labels += 1
    result = np.unique(chunk_sizes, return_counts=True)
    assert np.max(result[0]) == 12.8
    assert np.max(result[1]) == 129
    assert unknown_labels == 22