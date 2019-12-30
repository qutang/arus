from ....testing import load_test_data
from ..generator_stream import GeneratorSlidingWindowStream
from ...accelerometer import generator as accel_generator
from ...annotation import generator as annot_generator
import numpy as np
import pandas as pd


def test_GeneratorSlidingWindowStream():
    window_size = 12.8
    config = {
        'generator': accel_generator.normal_dist,
        'kwargs': {
            "grange": 8,
            "buffer_size": 100,
            "sleep_interval": 0,
            "sigma": 1,
            "sr": 50
        }}
    stream = GeneratorSlidingWindowStream(
        data_source=config, window_size=window_size, start_time_col=0, stop_time_col=0, name='sensor-generator-stream')
    stream.start()
    chunk_sizes = []
    chunk_means = []
    n = 5
    for data, _, _, _, _, name in stream.get_iterator():
        chunk_sizes.append(data.shape[0])
        chunk_means.append(np.mean(data.values[:, 1:]))
        n = n - 1
        if n == 0:
            break
    stream.stop()
    assert np.all(np.array(chunk_sizes[1:-1])
                  == window_size * config['kwargs']['sr'])
    np.testing.assert_array_almost_equal(
        np.array(chunk_means[1:-1]), 0, decimal=1)

    # annotation generator stream
    window_size = 12.8
    config = {
        'generator': annot_generator.normal_dist,
        'kwargs': {
            "duration_mu": 8,
            "duration_sigma": 1,
            "sleep_interval": 0,
            "num_mu": 2,
        }}
    stream = GeneratorSlidingWindowStream(
        data_source=config, window_size=window_size, start_time_col=1, stop_time_col=2, name='annotation-generator-stream')
    stream.start()
    chunk_durations = []
    n = 5
    for data, _, _, _, _, name in stream.get_iterator():
        durations = ((data.iloc[:, 2] - data.iloc[:, 1]) /
                     pd.Timedelta(1, 'S')).values.tolist()
        chunk_durations += durations
        n = n - 1
        if n == 0:
            break
    stream.stop()
    np.testing.assert_array_equal(
        np.array(chunk_durations) <= window_size, True)
