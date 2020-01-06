"""
Stream using mhealth annotation files
================================================================================

"""

# %%
# Imports
# ----------
from arus.testing import load_test_data
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arus.core.stream import AnnotationFileSlidingWindowStream

# %%
# Load test annotation files
# ---------------------------------
# `files` includes more than one file path.

files, sr = load_test_data(file_type='mhealth',
                           sensor_type='annotation',
                           file_num='multiple',
                           exception_type='no_missing')

# %%
# Setup stream
# ---------------
# Stream can accept multiple files as the data source and will read them one by one, so users should ensure the files are sorted in order.
window_size = 12.8
stream = AnnotationFileSlidingWindowStream(data_source=files,
                                           window_size=window_size,
                                           storage_format='mhealth',
                                           name='annotation-stream')

# %%
# Start stream and read in data
# ---------------------------------
stream.start()
chunk_sizes = []
for data, _, _, _, _, name in stream.get_iterator():
    if not data.empty:
        chunk_sizes.append(
            (data.iloc[-1, 2] - data.iloc[0, 1]) / pd.Timedelta(1, 's'))

# %%
# Stop stream
# --------------
stream.stop()

# %% 
# Plot the stats of the received data
# -------------------------------------
# The plot shows at many places, the duration of the annotation windows are not as long as the window size. This is normal, because annotations may not fill up the entire window and there are moments covered with no annotations.
pd.Series(chunk_sizes).plot(
    title='annotation durations of each window in the given stream with \nwindow size of ' + str(window_size) + ' seconds')
fig = plt.hlines(y=window_size,
                 xmin=0,
                 xmax=len(chunk_sizes),
                 linestyles='dashed')
