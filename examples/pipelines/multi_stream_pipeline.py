"""
Demonstration of the usage of arus.core.Pipeline (3)
====================================================

The pipeline uses multiple sensor streams.
"""

from arus.core.pipeline import Pipeline
import arus
import datetime as dt
import pandas as pd


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


if __name__ == "__main__":

    window_size = 12.8
    start_time = dt.datetime.now()

    gr1 = arus.generator.RandomAccelDataGenerator(
        sr=80, grange=8, sigma=1, buffer_size=100)
    seg1 = arus.segmentor.SlidingWindowSegmentor(window_size)
    stream1 = arus.Stream(gr1, seg1, name='stream-1')

    gr2 = arus.generator.RandomAccelDataGenerator(
        sr=50, grange=4, sigma=2, buffer_size=100)
    seg2 = arus.segmentor.SlidingWindowSegmentor(window_size)
    stream2 = arus.Stream(gr2, seg2, name='stream-2')

    pipeline = Pipeline(max_processes=2, scheduler='threads')
    pipeline.add_stream(stream1)
    pipeline.add_stream(stream2)
    pipeline.set_processor(_pipeline_test_processor)
    # for multiple streams, we have to serve start_time to sync streams
    pipeline.start(start_time=start_time)
    results = []
    for result, st, et, prev_st, prev_et, name in pipeline.get_iterator():
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
