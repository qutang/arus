import pandas as pd
import numpy as np
from datetime import datetime
import time


def _generate_ts(start_time, sr, buffer_size):
    N = buffer_size * sr + 1
    freq = str(int(1000 / sr)) + 'ms'
    ts = pd.date_range(start=start_time, periods=N, freq=freq)
    return ts[0:-1], ts[-1]


def _create_dataframe(ts, data):
    result = pd.DataFrame(index=ts, data=data, columns=['X', 'Y', 'Z'])
    result = result.reset_index(drop=False)
    result = result.rename(columns={'index': 'HEADER_TIME_STAMP'})
    return result


def normal_dist(sr, grange=8, start_time=None, buffer_size=1800, sleep_interval=0, sigma=1):
    start_time = datetime.now() if start_time is None else start_time
    st = time.time()
    while True:
        N = buffer_size * sr
        data = np.random.standard_normal(size=(N, 3)) * sigma
        data[data > grange] = grange
        data[data < -grange] = -grange
        ts, start_time = _generate_ts(start_time, sr, buffer_size)
        result = _create_dataframe(ts, data)
        delay = max([sleep_interval - (time.time() - st), 0])
        time.sleep(delay)
        st = time.time()
        yield result
