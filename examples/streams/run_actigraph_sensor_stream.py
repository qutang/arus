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
import arus
from arus.testing import load_test_data

# %%
# Load test Actigraph sensor files
# ---------------------------------
sensor_file, sr = load_test_data(file_type='actigraph',
                                 file_num='single',
                                 exception_type='consistent_sr')

# %%
# Setup stream
# --------------
window_size = 12.8
gr = arus.generator.ActigraphSensorFileGenerator(sensor_file)
seg = arus.segmentor.SlidingWindowSegmentor(window_size)
stream = arus.Stream(gr, seg, name='spades-2')

# %%
# Start stream and read in data
# ------------------------------
stream.start()
chunk_sizes = []
for data, _, _, _, _, name in stream.generate():
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
