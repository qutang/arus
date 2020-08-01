"""
Demonstration of the usage of arus.core.Pipeline (2)
====================================================

The pipeline uses one annotation stream and one sensor stream as input.
"""

from arus.core.pipeline import Pipeline
import arus
import datetime as dt
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
    import arus
    result = {'HEADER_TIME_STAMP': [],
              'START_TIME': [], 'STOP_TIME': [], 'VALUE': []}
    for data, st, et, prev_st, prev_et, name in chunk_list:
        result['HEADER_TIME_STAMP'].append(st)
        result['START_TIME'].append(st)
        result['STOP_TIME'].append(et)
        vm_values = arus.ext.numpy.vector_magnitude(data.iloc[:, 1:].values)
        values, name = arus.accel.mean(vm_values)
        result['VALUE'].append(values[0, 0])
    result = pd.DataFrame.from_dict(result)
    return result


if __name__ == "__main__":
    # test on one annotation stream and one sensor stream + feature pipeline

    window_size = 12.8
    sr = 80
    start_time = dt.datetime.now()
    gr1 = arus.generator.RandomAccelDataGenerator(
        sr=sr, grange=8, sigma=1, buffer_size=100)
    seg1 = arus.segmentor.SlidingWindowSegmentor(window_size)
    stream1 = arus.Stream(gr1, seg1, name='sensor-stream')

    gr2 = arus.generator.RandomAnnotationDataGenerator(
        labels=["Sitting", 'Standing', 'Lying', 'Walking', 'Running'], duration_mu=8, duration_sigma=2, num_mu=3, buffer_size=100)
    seg2 = arus.segmentor.SlidingWindowSegmentor(
        window_size, st_col=1, et_col=2)
    stream2 = arus.Stream(gr2, seg2, name='annotation-stream')

    feat_pipeline = Pipeline(
        max_processes=2, scheduler='processes', name='feature-pipeline')

    feat_pipeline.add_stream(stream1)
    feat_pipeline.set_processor(_feature_pipeline_processor)
    master_pipeline = Pipeline(
        max_processes=2, scheduler='processes', name='master-pipeline')
    master_pipeline.add_stream(stream2)
    master_pipeline.add_stream(feat_pipeline)
    master_pipeline.set_processor(_master_pipeline_processor)
    master_pipeline.start(start_time=start_time)
    results = []
    for result, st, et, prev_st, prev_et, name in master_pipeline.get_iterator():
        results.append(result)
        if len(results) == 10:
            break
        print(len(results))
    print(master_pipeline.stop())
    print(pd.concat(results, axis=0, sort=False))
