"""
Demonstration of the usage of arus.core.Pipeline (1)
====================================================

The pipeline uses one annotation stream and one sensor stream as input.
"""

from arus.core.pipeline import Pipeline
from arus.core.stream.generator_stream import GeneratorSlidingWindowStream
from arus.core.accelerometer import generator as accel_generator
from arus.core.annotation import generator as annot_generator
from datetime import datetime
import pandas as pd
import numpy as np
import logging


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
    logging.basicConfig(
        level=logging.DEBUG, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')
    # test on multiple streams
    stream1_config = {
        "generator": accel_generator.normal_dist,
        'kwargs': {
            "grange": 8,
            "buffer_size": 100,
            "sleep_interval": 0,
            "sigma": 1,
            "sr": 80
        }
    }

    stream2_config = {
        "generator": annot_generator.normal_dist,
        'kwargs': {
            "duration_mu": 8,
            "duration_sigma": 2,
            "sleep_interval": 1,
            "num_mu": 3,
            "labels": ["Sitting", 'Standing', 'Lying', 'Walking', 'Running']
        }
    }

    window_size = 12.8
    sr = 80
    start_time = datetime.now()
    stream1 = GeneratorSlidingWindowStream(
        stream1_config, window_size=window_size, start_time_col=0, stop_time_col=0, name='sensor-stream')
    stream2 = GeneratorSlidingWindowStream(
        stream2_config, window_size=window_size, start_time_col=1, stop_time_col=2, name='annotation-stream')

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
