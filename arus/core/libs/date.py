import numpy as np
import pandas as pd
import datetime as dt

def parse_timestamp(ts):
    if type(ts) == np.datetime64:
        result = pd.Timestamp(ts.astype('datetime64[ms]'))
    elif type(ts) == dt.datetime:
        result = pd.Timestamp(np.datetime64(ts, 'ms'))
    elif type(ts) == str:
        result = pd.Timestamp(ts)
    elif type(ts) == int or type(ts) == float:
        result = pd.Timestamp.fromtimestamp(ts)
    else:
        result = ts
    return result


def datetime2unix(ts):
    return (ts - dt.datetime(1970, 1, 1)) / dt.timedelta(seconds=1)
