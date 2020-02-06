"""
Demonstration of the usage of arus.core.pipeline
=====================================================================

This example shows how to pass previous output and input to processor for a pipeline.
"""

from arus.core.pipeline import Pipeline
import arus
import datetime as dt
import pandas as pd


def _pipeline_test_processor(chunk_list, prev_input=None, prev_output=None, **kwargs):
    import pandas as pd
    result = {'NAME': [],
              'START_TIME': [], 'STOP_TIME': []}
    for data, st, et, prev_st, prev_et, name in chunk_list:
        result['NAME'].append(name)
        result['START_TIME'].append(data.iloc[0, 0])
        result['STOP_TIME'].append(data.iloc[-1, 0])
    result = pd.DataFrame.from_dict(result)
    if prev_output is not None:
        result['PREV_START_TIME'] = prev_output['START_TIME']
        result['PREV_STOP_TIME'] = prev_output['STOP_TIME']
    else:
        result['PREV_START_TIME'] = None
        result['PREV_STOP_TIME'] = None
    return result


if __name__ == "__main__":
    window_size = 12.8
    gr1 = arus.generator.RandomAccelDataGenerator(
        sr=80, grange=8, sigma=1, buffer_size=100)
    seg1 = arus.segmentor.SlidingWindowSegmentor(window_size)
    stream1 = arus.Stream(gr1, seg1, name='stream-1')

    pipeline = Pipeline(
        max_processes=2, scheduler='processes', preserve_status=True)
    pipeline.add_stream(stream1)
    pipeline.set_processor(_pipeline_test_processor)
    pipeline.start()
    results = []
    prev_results = []
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
