from ...core.pipeline import Pipeline
from ...core.stream.generator_stream import GeneratorSlidingWindowStream
from ...core.accelerometer import generator
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import pytest


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


@pytest.mark.skipif(sys.platform == 'linux', reason="does not run on linux")
def test_Pipeline():
    # test on a single stream
    stream_config = {
        "generator": generator.normal_dist,
        'kwargs': {
            "grange": 8,
            "start_time": None,
            "buffer_size": 100,
            "sleep_interval": 0,
            "sigma": 1,
            "sr": 80
        }
    }

    window_size = 12.8
    start_time = datetime.now()
    stream = GeneratorSlidingWindowStream(
        stream_config, window_size=window_size, start_time=start_time, name='stream-1')

    pipeline = Pipeline(max_processes=2, scheduler='threads',
                        name='single-stream-pipeline')
    pipeline.add_stream(stream)
    pipeline.set_processor(_pipeline_test_processor)
    pipeline.start()

    results = []
    for result, st, et, prev_st, prev_et, name in pipeline.get_iterator():
        result['WINDOW_ST'] = st
        result['WINDOW_ET'] = et
        result['PREV_WINDOW_ST'] = prev_st
        result['PREV_WINDOW_ET'] = prev_et
        result['STREAM_NAME'] = name
        results.append(result)
        if len(results) == 5:
            break
    pipeline.finish_tasks_and_stop()
    results = pd.concat(results, axis=0, sort=False)
    durations = (results['WINDOW_ET'] -
                 results['WINDOW_ST']) / pd.Timedelta(1, unit='S')
    np.testing.assert_array_almost_equal(durations, window_size, decimal=1)

    # test on three streams, using the same pipeline but change the parameter of the first stream a little bit
    stream2_config = {
        "generator": generator.normal_dist,
        'kwargs': {
            "grange": 4,
            "start_time": None,
            "buffer_size": 400,
            "sleep_interval": 1,
            "sigma": 2,
            "sr": 50
        }
    }

    stream3_config = {
        "generator": generator.normal_dist,
        'kwargs': {
            "grange": 4,
            "start_time": None,
            "buffer_size": 800,
            "sleep_interval": 1,
            "sigma": 2,
            "sr": 80
        }
    }

    start_time = datetime.now()
    stream2 = GeneratorSlidingWindowStream(
        stream2_config, window_size=window_size, start_time=start_time, name='stream-2')
    stream3 = GeneratorSlidingWindowStream(
        stream3_config, window_size=window_size, start_time=start_time, name='stream-3')
    pipeline.get_stream('stream-1')._start_time = start_time
    pipeline.add_stream(stream2)
    pipeline.add_stream(stream3)
    pipeline.start()

    results = []
    for result, st, et, prev_st, prev_et, name in pipeline.get_iterator():
        result['WINDOW_ST'] = st
        result['WINDOW_ET'] = et
        result['PREV_WINDOW_ST'] = prev_st
        result['PREV_WINDOW_ET'] = prev_et
        result['STREAM_NAME'] = name
        np.testing.assert_array_equal(
            result['START_TIME'].values, result['START_TIME'].values[0])
        results.append(result)
        if len(results) == 5:
            break
    pipeline.finish_tasks_and_stop()
    results = pd.concat(results, axis=0, sort=False)
    durations = (results['WINDOW_ET'] -
                 results['WINDOW_ST']) / pd.Timedelta(1, unit='S')
    np.testing.assert_array_almost_equal(durations, window_size, decimal=1)
