from .. import moment
import numpy as np
import pandas as pd
import datetime as dt


def test_to_pandas_timestamp():
    ts = float(1580780607)

    now = dt.datetime.utcfromtimestamp(ts)
    np.testing.assert_almost_equal(
        ts, moment.to_pandas_timestamp(now).timestamp())

    now = ts
    np.testing.assert_almost_equal(
        ts, moment.to_pandas_timestamp(now).timestamp())

    now = pd.Timestamp.utcfromtimestamp(ts)
    np.testing.assert_almost_equal(
        ts, moment.to_pandas_timestamp(now).timestamp())

    now = np.datetime64(dt.datetime.utcfromtimestamp(ts))
    np.testing.assert_almost_equal(
        ts, moment.to_pandas_timestamp(now).timestamp())
