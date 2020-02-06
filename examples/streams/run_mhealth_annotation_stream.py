"""
Stream using mhealth annotation files
================================================================================

"""

# %%
# Imports
# ----------
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
# Get the test mhealth annotation files
# -------------------------------

spades_lab = arus.dataset.load_dataset('spades_lab')
annotation_files = spades_lab['subjects']['SPADES_22']['annotations']['SPADESInLab']

# %%
# Setup stream
# --------------
window_size = 12.8
generator = arus.generator.MhealthAnnotationFileGenerator(
    *annotation_files, buffer_size=10)
segmentor = arus.segmentor.SlidingWindowSegmentor(
    window_size=window_size, st_col=1, et_col=2)

stream = arus.Stream(generator, segmentor,
                     name='mhealth-annotation-stream', scheduler='thread')

# %%
# Start stream and read in data
# ---------------------------------
stream.start()
chunk_sizes = []
for data, st, et, _, _, name in stream.generate():
    print("{},{},{}".format(
        st, et, data.shape[0]))
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

if __name__ == "__main__":
    plt.show()
