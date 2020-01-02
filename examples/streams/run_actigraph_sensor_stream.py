"""
Stream using Actigraph sensor files
============================================================================

"""

# %%
# Imports
# -----------
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arus.core.stream import SensorFileSlidingWindowStream
from arus.testing import load_test_data

# %%
# Load test Actigraph sensor files
# ---------------------------------
files, sr = load_test_data(file_type='actigraph',
                           file_num='single',
                           exception_type='consistent_sr')

# %%
# Setup stream
# --------------
window_size = 12.8
stream = SensorFileSlidingWindowStream(data_source=files,
                                       window_size=window_size,
                                       sr=sr,
                                       buffer_size=1800,
                                       storage_format='actigraph',
                                       name='spades_2')

# %%
# Start stream and read in data
# ------------------------------
stream.start()
chunk_sizes = []
for data, _, _, _, _, name in stream.get_iterator():
    print("{},{},{},{}".format(name,
                               data.iloc[0, 0], data.iloc[-1, 0], data.shape[0]))
    chunk_sizes.append(data.shape[0])

# %%
# Stop stream
# --------------
stream.stop()

# %%
# Plot stats of recieved data
# ----------------------------
pd.Series(chunk_sizes).plot(
    title='chunk sizes of the given stream with \nwindow size of ' + str(window_size) + ' seconds, sampling rate at ' + str(sr) + ' Hz')
fig = plt.hlines(y=sr * window_size,
                 xmin=0,
                 xmax=len(chunk_sizes),
                 linestyles='dashed')
