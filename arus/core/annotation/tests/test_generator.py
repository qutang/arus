
from .. import generator as gr
import numpy as np
import pandas as pd
import time


def test_generator():
    # default setting
    duration_mu = 5
    duration_sigma = 1
    start_time = None
    num_mu = 3
    labels = ['Sitting', 'Standing', 'Lying']
    sleep_interval = 0
    
    durations = []
    rows = []
    n = 50
    for data in gr.normal_dist(duration_mu=duration_mu, duration_sigma=duration_sigma, start_time=start_time, num_mu=num_mu, labels=labels, sleep_interval=sleep_interval):
        durations += ((data['STOP_TIME'] - data['START_TIME'])/pd.Timedelta(1, 'S')).values.tolist()
        rows.append(data.shape[0])
        n -= 1
        if n == 0:
            break
    duration_mean = np.mean(durations)
    rows_mean = np.mean(rows)
    np.testing.assert_almost_equal(duration_mean, duration_mu, decimal=0)
    np.testing.assert_almost_equal(rows_mean, num_mu, decimal=0)
    
    
