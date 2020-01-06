"""
Stream using mhealth sensor files
================================================================================

This example demonstrates a stream that uses data files stored in mhealth specification as data source.
"""

# %%
# Imports
# --------
import logging
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arus.core.stream import SensorFileSlidingWindowStream
from arus.testing import load_test_data

# %%
# Turn on logging information
# ---------------------------
logging.basicConfig(
    level=logging.DEBUG, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')

# %%
# Get the test mhealth data files
# -------------------------------

files, sr = load_test_data(file_type='mhealth',
                           file_num='multiple', exception_type='inconsistent_sr')

# %%
# Setup stream
# --------------
window_size = 12.8
stream = SensorFileSlidingWindowStream(data_source=files,
                                       window_size=window_size,
                                       sr=sr,
                                       buffer_size=900,
                                       storage_format='mhealth',
                                       name='spades_2')

# %%
# Start stream and read in data
# -----------------------------
stream.start()
chunk_sizes = []
for data, _, _, _, _, name in stream.get_iterator():
    print("{},{},{}".format(
        data.iloc[0, 0], data.iloc[-1, 0], data.shape[0]))
    chunk_sizes.append(data.shape[0])

# %%
# Stop stream
# ------------
stream.stop()

# %%
# Plot the stats of the received data
# ------------------------------------
#
# The plot shows there are two places where the sampling rate of the data drops. This is because the data files loaded have missing data at those moments. The test data is manipulated to include missing moments for test purpose.
pd.Series(chunk_sizes).plot(
    title='chunk sizes of the given stream with \nwindow size of ' + str(window_size) + ' seconds, sampling rate at ' + str(sr) + ' Hz')
fig = plt.hlines(y=sr * window_size,
                 xmin=0,
                 xmax=len(chunk_sizes),
                 linestyles='dashed')
