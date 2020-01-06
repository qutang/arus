"""
Stream using mhealth data files with real-time delay
================================================================================

This example shows how to simulate real-time sensor streaming using existing data files.
"""

# %%
# Imports
# --------
import logging
import os
from glob import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arus.core.stream import SensorFileSlidingWindowStream
from arus.testing import load_test_data

# %%
# Turn on logging info
# -----------------------
logging.basicConfig(
    level=logging.DEBUG, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')

# %%
# Load test mhealth data files
# -----------------------------
files, sr = load_test_data(file_type='mhealth',
                           file_num='multiple',
                           exception_type='consistent_sr')

# %%
# Setup stream
# --------------
# To simulate real-time streaming delay, set `simulate_reality` to be `True`.
window_size = 1
stream = SensorFileSlidingWindowStream(data_source=files,
                                       window_size=window_size,
                                       sr=sr,
                                       buffer_size=900,
                                       storage_format='mhealth',
                                       simulate_reality=True,
                                       name='spades_2')

# %%
# Start stream and read in data
# ------------------------------
# To save time, only run for three windows, in total 3 seconds.
st = time.time()
stream.start()
chunk_sizes = []
for data, _, _, _, _, name in stream.get_iterator():
    print("{},{},{}".format(
        data.iloc[0, 0], data.iloc[-1, 0], data.shape[0]))
    chunk_sizes.append(data.shape[0])
    if len(chunk_sizes) == 3:
        break
lapsed_seconds = time.time() - st
print('Stream has run for ' + str(lapsed_seconds) + ' seconds.')

# %%
# Stop stream
# -------------
stream.stop()

# %%
# Plot stats of received data
# -----------------------------
pd.Series(chunk_sizes).plot(
    title='chunk sizes of the given stream with \nwindow size of ' + str(window_size) + ' seconds, sampling rate at ' + str(sr) + ' Hz')
fig = plt.hlines(y=sr * window_size,
                 xmin=0,
                 xmax=len(chunk_sizes),
                 linestyles='dashed')
