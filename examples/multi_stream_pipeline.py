from arus.core.pipeline import Pipeline
from arus.core.stream.generator_stream import GeneratorSlidingWindowStream
from arus.core.accelerometer import generator
from datetime import datetime
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
    # test on multiple streams
    stream1_config = {
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

    window_size = 12.8
    start_time = datetime.now()
    stream1 = GeneratorSlidingWindowStream(
        stream1_config, window_size=window_size, start_time=start_time, start_time_col=0, stop_time_col=0, name='stream-1')
    stream2 = GeneratorSlidingWindowStream(
        stream2_config, window_size=window_size, start_time=start_time, start_time_col=0, stop_time_col=0, name='stream-2')

    pipeline = Pipeline(max_processes=2)
    pipeline.add_stream(stream1)
    pipeline.add_stream(stream2)
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
        if len(results) == 10:
            break
    pipeline.finish_tasks_and_stop()
    print(pd.concat(results, axis=0, sort=False))
