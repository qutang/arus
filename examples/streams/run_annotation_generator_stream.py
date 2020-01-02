"""
Stream using annotation generator
=================================
"""

# %%
# Imports
# --------
from arus.core.annotation.generator import normal_dist
from arus.core.stream import GeneratorSlidingWindowStream

# %%
# Set up annotation generator
# ---------------------------
#
# `kwargs` will be passed to `normal_dist` when generating the stream
# The generated stream data will be stored as pandas DataFrame with mHealth Specification.
data_source = {
    "generator": normal_dist,
    "kwargs": {
        "duration_mu": 5,
        "duration_sigma": 1,
        "num_mu": 3,
        "labels": ['Sitting', 'Standing', 'Lying'],
        "sleep_interval": 0
    }
}

# %%
# Setup stream
# -------------------------------------------------
stream = GeneratorSlidingWindowStream(data_source,
                                      window_size=12.8,
                                      simulate_reality=False,
                                      start_time_col=1,
                                      stop_time_col=2, 
                                      name='annotator-generator-stream')

# %%
# Start stream and read data from the stream
# ------------------------------------------
#
# You may use `stream.get_iterator()` to return a generator to read data from a stream.
stream.start()
n = 5
for window, st, et, prev_st, prev_et, name in stream.get_iterator():
    print("{}-{}-{}".format(window.iloc[0, 0],
                            window.iloc[-1, 0], '-'.join(window.iloc[:, 3])))
    n -= 1
    if n == 0:
        break


# %%
# Stop the stream
# ------------------------------------------
stream.stop()
