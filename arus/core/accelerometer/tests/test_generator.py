
from .. import generator as gr
import numpy as np
import pandas as pd
import time


def test_generator():

    # default setting
    sr = 50
    grange = 4
    start_time = None
    buffer_size = 1800
    sleep_interval = 0
    sigma = 1
    n = 3
    for data in gr.normal_dist(sr=sr, grange=grange, start_time=start_time,
                               buffer_size=buffer_size, sleep_interval=sleep_interval, sigma=sigma):
        mean_data = np.mean(data.values[:, 1:], axis=0)
        std_data = np.std(data.iloc[:, 1:].values, axis=0)
        duration = (data.iloc[-1, 0] - data.iloc[0, 0]) / \
            pd.Timedelta(1, unit='seconds')
        np.testing.assert_array_almost_equal(
            mean_data, np.array([0, 0, 0]), decimal=2)
        np.testing.assert_array_almost_equal(
            std_data, np.array([sigma, sigma, sigma]),  decimal=2)
        np.testing.assert_almost_equal(duration, buffer_size, decimal=1)
        n = n - 1
        if n == 0:
            break

    # with sleep
    sr = 50
    grange = 4
    start_time = None
    buffer_size = 1800
    sleep_interval = 1
    sigma = 1
    ts = time.time()
    for data in gr.normal_dist(sr=sr, grange=grange, start_time=start_time,
                               buffer_size=buffer_size, sleep_interval=sleep_interval, sigma=sigma):
        delay = time.time() - ts
        ts = time.time()
        mean_data = np.mean(data.values[:, 1:], axis=0)
        std_data = np.std(data.iloc[:, 1:].values, axis=0)
        duration = (data.iloc[-1, 0] - data.iloc[0, 0]) / \
            pd.Timedelta(1, unit='seconds')
        np.testing.assert_array_almost_equal(
            mean_data, np.array([0, 0, 0]), decimal=2)
        np.testing.assert_array_almost_equal(
            std_data, np.array([sigma, sigma, sigma]),  decimal=2)
        np.testing.assert_almost_equal(duration, buffer_size, decimal=1)
        np.testing.assert_almost_equal(delay, sleep_interval, decimal=2)
        break

    # with max to grange
    sr = 50
    grange = 2
    start_time = None
    buffer_size = 1800
    sleep_interval = 0
    sigma = 4
    for data in gr.normal_dist(sr=sr, grange=grange, start_time=start_time,
                               buffer_size=buffer_size, sleep_interval=sleep_interval, sigma=sigma):
        mean_data = np.mean(data.values[:, 1:], axis=0)
        std_data = np.std(data.iloc[:, 1:].values, axis=0)
        duration = (data.iloc[-1, 0] - data.iloc[0, 0]) / \
            pd.Timedelta(1, unit='seconds')
        max_counts = np.sum(np.abs(data.values[:, 1:]) == grange)
        np.testing.assert_array_almost_equal(
            mean_data, np.array([0, 0, 0]), decimal=2)
        np.testing.assert_array_almost_equal(
            std_data, np.array([1.72, 1.72, 1.72]),  decimal=2)
        np.testing.assert_almost_equal(duration, buffer_size, decimal=1)
        assert max_counts > grange / sigma * sr * buffer_size * 3
        break
