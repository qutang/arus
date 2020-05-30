"""
Demonstration of the usage of arus.core.Pipeline (1)
====================================================

The pipeline uses one annotation stream and one sensor stream as input.
"""

from arus.core.pipeline import Pipeline
import arus
import datetime as dt
import pandas as pd
import numpy as np


def _pipeline_test_processor(chunk_list, **kwargs):
    import pandas as pd
    import numpy as np
    result = {'NAME': [],
              'START_TIME': [], 'STOP_TIME': [], 'VALUE': []}
    for data, st, et, prev_st, prev_et, name in chunk_list:
        if name == 'sensor-stream':
            result['NAME'].append(name)
            result['START_TIME'].append(data.iloc[0, 0])
            result['STOP_TIME'].append(data.iloc[-1, 0])
            result['VALUE'].append(np.mean(data.iloc[:, 1:].values))
        elif name == 'annotation-stream':
            result['NAME'].append(name)
            result['START_TIME'].append(data.iloc[0, 1])
            result['STOP_TIME'].append(data.iloc[-1, 2])
            result['VALUE'].append('-'.join(data.iloc[:, 3].values))
    result = pd.DataFrame.from_dict(result)
    return result


if __name__ == "__main__":
    window_size = 12.8
    sr = 80

    gr1 = arus.generator.RandomAccelDataGenerator(
        sr=sr, grange=8, sigma=1, buffer_size=100)
    seg1 = arus.segmentor.SlidingWindowSegmentor(window_size)
    stream1 = arus.Stream(gr1, seg1, name='sensor-stream')

    gr2 = arus.generator.RandomAnnotationDataGenerator(
        labels=["Sitting", 'Standing', 'Lying', 'Walking', 'Running'], duration_mu=8, duration_sigma=2, num_mu=3, buffer_size=100)
    seg2 = arus.segmentor.SlidingWindowSegmentor(
        window_size, st_col=1, et_col=2)
    stream2 = arus.Stream(gr2, seg2, name='annotation-stream')

    start_time = dt.datetime.now()
    pipeline = Pipeline(max_processes=2)
    pipeline.add_stream(stream1)
    pipeline.add_stream(stream2)
    pipeline.set_processor(_pipeline_test_processor)
    pipeline.start(start_time=start_time)
    results = []
    for result, st, et, prev_st, prev_et, name in pipeline.get_iterator():
        print(st)
        result['WINDOW_ST'] = st
        result['WINDOW_ET'] = et
        result['PREV_WINDOW_ST'] = prev_st
        result['PREV_WINDOW_ET'] = prev_et
        result['STREAM_NAME'] = name
        results.append(result)
        if len(results) == 10:
            break
    pipeline.stop()
    print(pd.concat(results, axis=0, sort=False))
