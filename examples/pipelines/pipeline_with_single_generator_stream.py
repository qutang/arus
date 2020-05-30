"""
Pipeline with single sensor generator stream
=====================================================================

This example demonstrates using pipeline with a single sensor generator stream.
"""

# %%
# Imports
# ----------

import pandas as pd
import multiprocessing

from arus.core.pipeline import Pipeline
import arus
import datetime as dt

multiprocessing.freeze_support()

# %%
# pipeline processor test function
# ---------------------------------


def _pipeline_test_processor(chunk_list, **kwargs):
    import pandas as pd
    result = {'NAME': [],
              'START_TIME': [], 'STOP_TIME': []}
    for data, st, et, prev_st, prev_et, name in chunk_list:
        result['NAME'].append(name)
        result['START_TIME'].append(data.iloc[0, 0])
        result['STOP_TIME'].append(data.iloc[-1, 0])
    result = pd.DataFrame.from_dict(result)
    return result


# %%
# Setup stream
# --------------
window_size = 12.8
gr1 = arus.generator.RandomAccelDataGenerator(
    sr=80, grange=8, sigma=1, buffer_size=100)
seg1 = arus.segmentor.SlidingWindowSegmentor(window_size)
stream1 = arus.Stream(gr1, seg1, name='sensor-stream')

# %%
# Setup pipeline
# ----------------
# Here we use `threads` scheduler for demostration due to limitation of sphinx_gallery. In practice, you should better use `processes` scheduler to get benefit from multi-core processing.
pipeline = Pipeline(max_processes=1, scheduler='threads')
pipeline.add_stream(stream1)
pipeline.set_processor(_pipeline_test_processor)

# %%
# Start pipeline and read in processed data
# -------------------------------------------
pipeline.start()
results = []
for result, st, et, prev_st, prev_et, name in pipeline.get_iterator():
    result['PREV_WINDOW_ST'] = prev_st
    results.append(result)
    if len(results) == 10:
        break

# %%
# Stop pipeline
# ----------------
success = pipeline.stop()

# %%
# Output of the processed data
# ------------------------------
output = pd.concat(results, axis=0, sort=False)
output
