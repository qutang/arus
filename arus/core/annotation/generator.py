import pandas as pd
import numpy as np
from datetime import datetime
import time

def _create_dataframe(start_times, stop_times, label_names):
    result = pd.DataFrame.from_dict({
        'HEADER_TIME_STAMP': start_times,
        'START_TIME': start_times,
        'STOP_TIME': stop_times,
        'LABEL_NAME': label_names
    })
    return result


def normal_dist(duration_mu=5, duration_sigma=5, start_time=None, num_mu=2, num_sigma=1, labels=['Sitting', 'Standing', 'Lying'], sleep_interval=0):
    start_time = pd.Timestamp(datetime.now()) if start_time is None else start_time
    while True:
        N = np.random.poisson(lam=num_mu)
        durations = np.random.standard_normal(size=N) * duration_sigma + duration_mu
        start_times = [start_time]
        stop_times = []
        for duration in durations:
            new_start_time = start_time + pd.Timedelta(duration, 'S')
            start_times.append(new_start_time)
            start_time = new_start_time
            stop_times.append(new_start_time)
        start_times = start_times[:-1]
        label_names = np.random.choice(labels, N)
        result = _create_dataframe(start_times, stop_times, label_names)
        yield result
