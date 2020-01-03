from arus.testing import load_test_data
import pandas as pd
from arus.models.muss import MUSSModel
from pathos import pools
from arus.plugins.metawear import MetaWearSlidingWindowStream, MetaWearScanner
from arus.core.libs import mhealth_format as arus_mh
from playsound import playsound
import datetime as dt
import os


def load_initial_data():
    class_filepath, _ = load_test_data(file_type='mhealth', sensor_type='class_labels',
                                       file_num='single', exception_type='multi_tasks')
    feature_filepath, _ = load_test_data(file_type='mhealth', sensor_type='feature',
                                         file_num='single', exception_type='multi_placements')
    class_df = pd.read_csv(class_filepath, parse_dates=[
        0, 1, 2], infer_datetime_format=True)
    feature_df = pd.read_csv(feature_filepath, parse_dates=[
        0, 1, 2], infer_datetime_format=True)
    return feature_df, class_df


def train_initial_model(training_labels, feature_df, class_df, pool):
    muss = MUSSModel()
    feature_set = feature_df
    class_set = class_df
    yield 'Extracting training data for DW...'
    dw_features = feature_set.loc[feature_set['SENSOR_PLACEMENT'] == 'DW', [
        'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + muss.get_feature_names()]
    yield 'Extracting training data for DA...'
    da_features = feature_set.loc[feature_set['SENSOR_PLACEMENT'] == 'DA', [
        'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + muss.get_feature_names()]
    yield 'Combining training data together...'
    combined_feature_set, combined_feature_names = muss.combine_features(
        dw_features, da_features, placement_names=['DW', 'DA'])
    cleared_class_set = class_set[['HEADER_TIME_STAMP',
                                   'START_TIME', 'STOP_TIME', 'MUSS_22_ACTIVITY_ABBRS']]
    yield 'Synchronizing training data and class labels...'
    synced_feature, synced_class = muss.sync_feature_and_class(
        combined_feature_set, cleared_class_set)
    # only use training labels
    yield 'Filtering out unused class labels...'
    filter_condition = synced_class['MUSS_22_ACTIVITY_ABBRS'].isin(
        training_labels)
    input_feature = synced_feature.loc[filter_condition, :]
    input_class = synced_class.loc[filter_condition, :]

    yield 'Training SVM classifier...'
    task = pool.apipe(muss.train_classifier, input_feature,
                      input_class, class_col='MUSS_22_ACTIVITY_ABBRS', feature_names=combined_feature_names, placement_names=['DW', 'DA'])
    yield task


def validate_initial_model(model, feature_df, class_df, pool):
    muss = MUSSModel()
    feature_set = feature_df
    class_set = class_df
    yield 'Extracting validation data for DW...'
    dw_features = feature_set.loc[feature_set['SENSOR_PLACEMENT'] == 'DW', [
        'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME', 'PID'] + muss.get_feature_names()]
    yield 'Extracting validation data for DA...'
    da_features = feature_set.loc[feature_set['SENSOR_PLACEMENT'] == 'DA', [
        'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME', 'PID'] + muss.get_feature_names()]
    yield 'Combining validation data together...'
    combined_feature_set, combined_feature_names = muss.combine_features(
        dw_features, da_features, placement_names=['DW', 'DA'], group_col='PID')
    cleared_class_set = class_set[['HEADER_TIME_STAMP',
                                   'START_TIME', 'STOP_TIME', 'PID', 'MUSS_22_ACTIVITY_ABBRS']]
    yield 'Synchronizing training data and class labels...'
    synced_feature, synced_class = muss.sync_feature_and_class(
        combined_feature_set, cleared_class_set, group_col='PID')
    # only use training labels
    yield 'Filtering out unused class labels...'
    training_labels = model[0].classes_
    filter_condition = synced_class['MUSS_22_ACTIVITY_ABBRS'].isin(
        training_labels)
    input_feature = synced_feature.loc[filter_condition, :]
    input_class = synced_class.loc[filter_condition, :]

    yield 'Validating SVM classifier...'
    task = pool.apipe(muss.validate_classifier, input_feature,
                      input_class, class_col='MUSS_22_ACTIVITY_ABBRS', feature_names=combined_feature_names, placement_names=['DW', 'DA'], group_col='PID')
    yield task


def get_confusion_matrix_figure(validation_result):
    muss = MUSSModel()
    labels = validation_result[-2]
    fig = muss.get_confusion_matrix(
        validation_result[0], validation_result[1], labels=labels, graph=True)
    return fig


def get_classification_report_table(validation_result):
    muss = MUSSModel()
    report = muss.get_classification_report(
        validation_result[0], validation_result[1], labels=validation_result[-2])
    report_table = []
    for key, values in report.items():
        report_row = []
        if key in validation_result[-2]:
            report_row.append(key)
            for metric_name, metric_value in values.items():
                if metric_name == 'precision':
                    report_row.append(round(metric_value, 2))
                elif metric_name == 'recall':
                    report_row.append(round(metric_value, 2))
                elif metric_name == 'f1-score':
                    report_row.append(round(metric_value, 2))
            report_table.append(report_row)
    return report_table


def test_initial_model(devices, model):
    muss = MUSSModel()
    device_addrs = devices
    streams = []
    start_time = dt.datetime.now()
    for addr, placement in zip(device_addrs, ['DW', 'DA']):
        stream = MetaWearSlidingWindowStream(
            addr, window_size=4, sr=50, grange=8, name=placement)
        streams.append(stream)
    pipeline = muss.get_inference_pipeline(
        *streams, name='muss-pipeline', model=model, DA={'sr': 50}, DW={'sr': 50}, max_processes=2, scheduler='processes')
    pipeline.connect(start_time=start_time)
    return pipeline


def collect_data(devices, output_folder, pid, model=None):
    muss = MUSSModel()
    device_addrs = devices
    streams = []
    window_size = 4
    sr = 50
    output_folder = os.path.join(output_folder, 'data')
    start_time = dt.datetime.now()
    for addr, placement in zip(device_addrs, ['DW', 'DA']):
        stream = MetaWearSlidingWindowStream(
            addr, window_size=window_size, sr=sr, grange=8, name=placement)
        streams.append(stream)
    pipeline = muss.get_data_collection_pipeline(
        *streams, model=model, DA={'sr': sr}, DW={'sr': sr}, max_processes=2, scheduler='processes', output_folder=output_folder, pid=pid)
    pipeline.connect(start_time=start_time)
    return pipeline


def save_current_annotation(current_annotation, output_folder, pid, pool, active=False):
    output_folder = os.path.join(output_folder, 'data')
    df = pd.DataFrame.from_dict(current_annotation)
    task = pool.apipe(arus_mh.write_data_csv, df, output_folder=output_folder,                  pid=pid, file_type='annotation',
                      sensor_or_annotation_type='ActiveSession' if active else 'PassiveSession', sensor_or_annotator_id='ARUS', split_hours=False, flat=True, append=True)
    return task


def play_sound(text, pool):
    return pool.apipe(playsound, text + '.mp3', block=True)


def get_nearby_devices(pool):
    scanner = MetaWearScanner()
    task = pool.apipe(scanner.get_nearby_devices, max_devices=2)
    return task
