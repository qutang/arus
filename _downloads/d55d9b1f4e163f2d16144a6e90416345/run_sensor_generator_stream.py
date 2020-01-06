"""
Stream using accelerometer generator
================================================================================

"""

# %%
# Imports
# ---------
from arus.core.stream import GeneratorSlidingWindowStream
from arus.core.accelerometer.generator import normal_dist

# %%
# Setup accelerometer generator
# -----------------------------
#
# `kwargs` will be passed to `normal_dist` when generating the stream
# The generated stream data will be stored as pandas DataFrame with mHealth Specification.
data_source = {'generator': normal_dist, 'kwargs': {
    "grange": 4,
    "buffer_size": 1800,
    "sleep_interval": 1,
    "sigma": 1,
    "sr": 80
}}

# %%
# Setup stream
# -------------
stream = GeneratorSlidingWindowStream(data_source,
                                      window_size=12.8,
                                      simulate_reality=False,
                                      start_time_col=0,
                                      stop_time_col=0,
                                      name='sensor-generator-stream')

# %%
# Start stream and get data from the stream
# ------------------------------------------
#
# You may use `stream.get_iterator()` to return a generator to read data from a stream.
stream.start()
n = 5
for window, st, et, prev_st, prev_et, name in stream.get_iterator():
    print(
        "{}-{}-{}".format(window.iloc[0, 0], window.iloc[-1, 0], window.shape))
    n -= 1
    if n == 0:
        break

# %%
# Stop stream
# ------------------------------------------
stream.stop()
