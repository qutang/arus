from arus.models.muss import MUSSModel
from arus.testing import load_test_data
import pandas as pd
from arus.plugins.metawear.stream import MetaWearSlidingWindowStream
from datetime import datetime
import logging


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
    stream = MetaWearSlidingWindowStream("D2:C6:AF:2B:DB:22", sr=50, grange=8,
                                         window_size=12.8, start_time=datetime.now(), name='DW')
    return stream


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')
    muss = MUSSModel()
    model = train_test_classifier(muss)
    stream = prepare_streams()
    muss_pipeline = muss.get_inference_pipeline(
        stream, model=model, sr=50, scheduler='processes', max_processes=2)
    muss_pipeline.start()
    i = 0
    for data, _, _, _, _, name in muss_pipeline.get_iterator():
        print(data)
        i = i + 1
        if i == 5:
            break
    muss_pipeline.stop()
