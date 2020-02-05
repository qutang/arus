"""
New Stream using mhealth sensor files
================================================================================

This example demonstrates a stream that generates stream segments from data files stored in mhealth specification using sliding window method.
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

import arus

# %%
# Turn on logging information
# ---------------------------
arus.dev.set_default_logging()

# %%
# Get the test mhealth data files
# -------------------------------

spades_lab = arus.dataset.load_dataset('spades_lab')
sensor_files = spades_lab['subjects']['SPADES_22']['sensors']['DA']

# %%
# Setup stream
# --------------
window_size = 12.8
generator = arus.generator.MhealthSensorFileGenerator(
    *sensor_files, buffer_size=1800)
segmentor = arus.segmentor.SlidingWindowSegmentor(window_size=window_size)

stream = arus.Stream(generator, segmentor,
                     name='mhealth-stream', scheduler='thread')

# %%
# Start stream and read in data
# -----------------------------
stream.start()
chunk_sizes = []
for data, st, et, _, _, name in stream.get_iterator():
    print("{},{},{}".format(
        st, et, data.shape[0]))
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
    title='chunk sizes of the given stream with \nwindow size of ' + str(window_size) + ' seconds')
fig = plt.hlines(y=80 * window_size,
                 xmin=0,
                 xmax=len(chunk_sizes),
                 linestyles='dashed')

if __name__ == "__main__":
    plt.show()
