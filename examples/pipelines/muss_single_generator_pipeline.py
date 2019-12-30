"""
Demonstration of the usage of arus.models.muss.get_inference_pipeline
=====================================================================

The pipeline uses a single sensor generator stream.
"""

from arus.models.muss import MUSSModel
from arus.testing import load_test_data
import pandas as pd
from arus.core.stream.generator_stream import GeneratorSlidingWindowStream
from arus.core.accelerometer import generator
from datetime import datetime


def train_test_classifier(muss):
    feature_filepath, _ = load_test_data(file_type='mhealth', sensor_type='feature',
                                         file_num='single', exception_type='single_placement')
    class_filepath, _ = load_test_data(file_type='mhealth', sensor_type='class_labels',
                                       file_num='single', exception_type='multi_tasks')
    feature_set = pd.read_csv(
        feature_filepath, infer_datetime_format=True, parse_dates=[0, 1, 2])
    feature_set = feature_set[['HEADER_TIME_STAMP',
                               'START_TIME', 'STOP_TIME'] + muss.get_feature_names()]
    class_set = pd.read_csv(
        class_filepath, infer_datetime_format=True, parse_dates=[0, 1, 2])
    class_set = class_set[['HEADER_TIME_STAMP',
                           'START_TIME', 'STOP_TIME', 'MUSS_3_POSTURES']]
    input_feature, input_class = muss.sync_feature_and_class(
        feature_set, class_set)
    filtered_feature, filtered_class = muss.remove_classes(
        input_feature, input_class, class_col='MUSS_3_POSTURES', classes_to_remove=['Unknown', 'Transition'])
    model = muss.train_classifier(
        filtered_feature, filtered_class, class_col='MUSS_3_POSTURES', feature_names=muss.get_feature_names(), placement_names=['DW'])
    return model


def prepare_streams():
    stream1_config = {
        "generator": generator.normal_dist,
        'kwargs': {
            "grange": 8,
            "start_time": None,
            "buffer_size": 100,
            "sleep_interval": 0,
            "sigma": 1,
            "sr": 50
        }
    }

    window_size = 12.8
    start_time = datetime.now()
    stream = GeneratorSlidingWindowStream(
        stream1_config, window_size=window_size, start_time=start_time, start_time_col=0, stop_time_col=0, name='DW')
    return stream


if __name__ == "__main__":
    muss = MUSSModel()
    model = train_test_classifier(muss)
    stream = prepare_streams()
    muss_pipeline = muss.get_inference_pipeline(
        stream, model=model, scheduler='processes', max_processes=2, DW={'sr': 50})
    muss_pipeline.start()
    i = 0
    for data, _, _, _, _, name in muss_pipeline.get_iterator():
        i = i + 1
        print(data)
        if i == 5:
            break
    muss_pipeline.stop()