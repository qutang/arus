from arus.core.pipeline import Pipeline
from arus.core.stream.generator_stream import GeneratorSlidingWindowStream
from arus.core.accelerometer import generator as accel_generator
from arus.core.annotation import generator as annot_generator
from datetime import datetime
import pandas as pd
import numpy as np


def _master_pipeline_processor(chunk_list, **kwargs):
    import pandas as pd
    import numpy as np
    result = {'HEADER_TIME_STAMP': [], 'START_TIME': [],
              'STOP_TIME': [], 'VALUE': [], 'ANNOTATION': []}
    for data, st, et, prev_st, prev_et, name in chunk_list:
        if len(result['START_TIME']) == 0 or result['START_TIME'][-1] != st:
            result['HEADER_TIME_STAMP'].append(st)
            result['START_TIME'].append(st)
            result['STOP_TIME'].append(et)
        if name == 'annotation-stream':
            result['ANNOTATION'].append('-'.join(data.iloc[:, 3].values))
        elif name == 'feature-pipeline':
            result['VALUE'].append(data.iloc[0, 3])
    result = pd.DataFrame.from_dict(result)

    return result


def _feature_pipeline_processor(chunk_list, **kwargs):
    import pandas as pd
    import numpy as np
    from arus.core.accelerometer.features import stats as accel_stats
    from arus.core.accelerometer.transformation import vector_magnitude
    result = {'HEADER_TIME_STAMP': [],
              'START_TIME': [], 'STOP_TIME': [], 'VALUE': []}
    for data, st, et, prev_st, prev_et, name in chunk_list:
        result['HEADER_TIME_STAMP'].append(st)
        result['START_TIME'].append(st)
        result['STOP_TIME'].append(et)
        vm_values = vector_magnitude(data.iloc[:, 1:].values)
        values, name = accel_stats.mean(vm_values)
        result['VALUE'].append(values[0, 0])
    result = pd.DataFrame.from_dict(result)
    return result


if __name__ == "__main__":
    # test on one annotation stream and one sensor stream + feature pipeline
    stream1_config = {
        "generator": accel_generator.normal_dist,
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
        "generator": annot_generator.normal_dist,
        'kwargs': {
            "duration_mu": 8,
            "start_time": None,
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
        stream1_config, window_size=window_size, start_time=start_time, start_time_col=0, stop_time_col=0, name='sensor-stream')
    stream2 = GeneratorSlidingWindowStream(
        stream2_config, window_size=window_size, start_time=start_time, buffer_size=None, start_time_col=1, stop_time_col=2, name='annotation-stream')

    feat_pipeline = Pipeline(
        max_processes=2, scheduler='processes', name='feature-pipeline')
    feat_pipeline.add_stream(stream1)
    feat_pipeline.set_processor(_feature_pipeline_processor)
    master_pipeline = Pipeline(
        max_processes=2, scheduler='processes', name='master-pipeline')
    master_pipeline.add_stream(stream2)
    master_pipeline.add_stream(feat_pipeline)
    master_pipeline.set_processor(_master_pipeline_processor)
    master_pipeline.start()
    results = []
    for result, st, et, prev_st, prev_et, name in master_pipeline.get_iterator():
        results.append(result)
        if len(results) == 10:
            break
        print(len(results))
    print(master_pipeline.finish_tasks_and_stop())
    print(pd.concat(results, axis=0, sort=False))
