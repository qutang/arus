"""
Demonstration of the usage of arus.models.muss.get_inference_pipeline
=====================================================================

The pipeline uses multiple sensor generator streams.
"""

from arus.models.muss import MUSSModel
from arus.testing import load_test_data
import pandas as pd
import arus
import datetime as dt


def train_test_classifier(muss):
    feature_filepath, _ = load_test_data(file_type='mhealth', sensor_type='feature',
                                         file_num='single', exception_type='multi_placements')
    class_filepath, _ = load_test_data(file_type='mhealth', sensor_type='class_labels',
                                       file_num='single', exception_type='multi_tasks')
    feature_set = pd.read_csv(
        feature_filepath, infer_datetime_format=True, parse_dates=[0, 1, 2])
    dw_feature_set = feature_set.loc[feature_set['SENSOR_PLACEMENT'] == 'DW', [
        'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + muss.get_feature_names()]
    da_feature_set = feature_set.loc[feature_set['SENSOR_PLACEMENT'] == 'DA', [
        'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + muss.get_feature_names()]
    combined_feature_set, feature_names = muss.combine_features(
        dw_feature_set, da_feature_set, placement_names=['DW', 'DA'], group_col=None)
    class_set = pd.read_csv(
        class_filepath, infer_datetime_format=True, parse_dates=[0, 1, 2])
    class_set = class_set[['HEADER_TIME_STAMP',
                           'START_TIME', 'STOP_TIME', 'MUSS_3_POSTURES']]
    input_feature, input_class = muss.sync_feature_and_class(
        combined_feature_set, class_set)
    filtered_feature, filtered_class = muss.remove_classes(
        input_feature, input_class, class_col='MUSS_3_POSTURES', classes_to_remove=['Unknown', 'Transition'])
    model = muss.train_classifier(
        filtered_feature, filtered_class, class_col='MUSS_3_POSTURES', feature_names=feature_names, placement_names=['DW', 'DA'])
    return model


def prepare_streams():
    window_size = 12.8
    gr1 = arus.generator.RandomAccelDataGenerator(
        sr=50, grange=8, sigma=1, buffer_size=100)
    seg1 = arus.segmentor.SlidingWindowSegmentor(window_size)
    stream1 = arus.Stream(gr1, seg1, name='DW')
    gr2 = arus.generator.RandomAccelDataGenerator(
        sr=50, grange=4, sigma=2, buffer_size=100)
    seg2 = arus.segmentor.SlidingWindowSegmentor(window_size)
    stream2 = arus.Stream(gr2, seg2, name='DA')
    return stream1, stream2


if __name__ == "__main__":
    start_time = dt.datetime.now()
    muss = MUSSModel()
    model = train_test_classifier(muss)
    stream1, stream2 = prepare_streams()
    muss_pipeline = muss.get_inference_pipeline(
        stream1, stream2, model=model, scheduler='processes', max_processes=2, DW={'sr': 50}, DA={'sr': 50})
    muss_pipeline.start(start_time=start_time)
    i = 0
    for data, _, _, _, _, name in muss_pipeline.get_iterator():
        i = i + 1
        print(data)
        if i == 5:
            break
    muss_pipeline.stop()
