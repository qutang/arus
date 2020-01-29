
from .. import generator as gr
import numpy as np
import pandas as pd
import time


def test_generate_accel_from_normal_distribution():
    # default setting
    sr = 3600
    grange = 4
    start_time = None
    buffer_size = 3600 / 2
    sigma = 1
    max_samples = buffer_size * 2
    for data in gr.generate_accel_from_normal_distribution(
            sr=sr,
            grange=grange,
            start_time=start_time,
            buffer_size=buffer_size,
            sigma=sigma,
            max_samples=max_samples):
        mean_data = np.mean(data.values[:, 1:], axis=0)
        std_data = np.std(data.iloc[:, 1:].values, axis=0)
        duration = (data.iloc[-1, 0] - data.iloc[0, 0]) / \
            pd.Timedelta(1, unit='seconds')
        np.testing.assert_array_almost_equal(
            mean_data, np.array([0, 0, 0]), decimal=1)
        np.testing.assert_array_almost_equal(
            std_data, np.array([sigma, sigma, sigma]),  decimal=1)
        np.testing.assert_almost_equal(duration, buffer_size / sr, decimal=1)


def test_generate_annotation_from_normal_distribution():
    # default setting
    duration_mu = 5
    duration_sigma = 1
    start_time = None
    num_mu = 3
    labels = ['Sitting', 'Standing', 'Lying']
    max_samples = 50

    durations = []
    rows = []
    for data in gr.generate_annotation_from_normal_distribution(duration_mu=duration_mu, duration_sigma=duration_sigma, start_time=start_time, num_mu=num_mu, labels=labels, max_samples=max_samples):
        durations += ((data['STOP_TIME'] - data['START_TIME']
                       )/pd.Timedelta(1, 'S')).values.tolist()
        rows.append(data.shape[0])
    duration_mean = np.mean(durations)
    rows_mean = np.mean(rows)
    np.testing.assert_almost_equal(duration_mean, duration_mu, decimal=0)
    np.testing.assert_almost_equal(rows_mean, num_mu, decimal=0)
