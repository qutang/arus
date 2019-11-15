from arus.core.pipeline import Pipeline
from arus.core.stream import SensorGeneratorStream
from arus.core.accelerometer import generator
from datetime import datetime
import pandas as pd


def _pipeline_test_processor(chunk_list, **kwargs):
    import pandas as pd
    result = {'NAME': [],
              'START_TIME': [], 'STOP_TIME': []}
    for data, name in chunk_list:
        result['NAME'].append(name)
        result['START_TIME'].append(data.iloc[0, 0])
        result['STOP_TIME'].append(data.iloc[-1, 0])
    result = pd.DataFrame.from_dict(result)
    return result


if __name__ == "__main__":
    # test on a single stream
    stream1_config = {
        "generator": generator.normal_dist,
        'kwargs': {
            "grange": 8,
            "start_time": None,
            "buffer_size": 100,
            "sleep_interval": 0,
            "sigma": 1
        }
    }

    window_size = 12.8
    sr = 80
    start_time = datetime.now()
    stream1 = SensorGeneratorStream(
        stream1_config, window_size=window_size, sr=sr, start_time=start_time, name='stream-1')

    pipeline = Pipeline(max_processes=2)
    pipeline.add_stream(stream1)
    pipeline.set_processor(_pipeline_test_processor)
    pipeline.start()
    results = []
    for result in pipeline.get_iterator():
        results.append(result)
        if len(results) == 10:
            break
    pipeline.finish_tasks_and_stop()
    print(pd.concat(results, axis=0, sort=False))
