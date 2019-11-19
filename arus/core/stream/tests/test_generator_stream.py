from ....testing import load_test_data
from ..generator_stream import GeneratorSlidingWindowStream
from ...accelerometer import generator
import numpy as np
import pandas as pd

def test_GeneratorSlidingWindowStream():
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
    stream = GeneratorSlidingWindowStream(
        data_source=config, sr=sr, window_size=window_size, start_time=None)
    stream.start()
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