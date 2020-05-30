from arus.testing import load_test_data
import pandas as pd
from arus.models.muss import MUSSModel, Strategy
from pathos import pools
import arus
from playsound import playsound
import datetime as dt
import os
import enum
import queue
import numpy as np
from loguru import logger

muss = MUSSModel()


class PROCESSOR_MODE(enum.Enum):
    TEST_ONLY = enum.auto()
    TEST_AND_SAVE = enum.auto()
    COLLECT_ONLY = enum.auto()
    TEST_AND_COLLECT = enum.auto()
    ACTIVE_COLLECT = enum.auto()


def load_origin_dataset():
    class_filepath, _ = load_test_data(file_type='mhealth', sensor_type='class_labels',
                                       file_num='single', exception_type='multi_tasks')
    feature_filepath, _ = load_test_data(file_type='mhealth', sensor_type='feature',
                                         file_num='single', exception_type='multi_placements')
    class_df = pd.read_csv(class_filepath, parse_dates=[
        0, 1, 2], infer_datetime_format=True)
    feature_df = pd.read_csv(feature_filepath, parse_dates=[
        0, 1, 2], infer_datetime_format=True)
    return feature_df, class_df


def get_class_label_candidates(dataset):
    class_df = dataset[1]
    return class_df['MUSS_22_ACTIVITY_ABBRS'].unique().tolist()


def get_model_summary(model=None):
    if model is not None:
        name = ','.join(model[0].classes_)
        acc = round(model[2], 2)
        summary = 'Classes:\n' + name + '\n' + \
            'Training accuracy:\n' + str(acc)
    else:
        summary = 'No model is available'
    return summary


def get_dataset_summary(dataset=None):
    if dataset is not None:
        labels = dataset['GT_LABEL'].unique().tolist()
        num_of_windows = dataset.shape[0]
        summary = 'Classes:\n' + \
            str(labels) + '\n' + 'Total windows:\n' + str(num_of_windows)
    else:
        summary = 'No data is available'
    return summary


def get_data_summary_table(dataset=None):
    def _summarize(df):
        if 'PREDICTION' not in df:
            windows = df.shape[0]
            accuracy = np.nan
        else:
            correct = np.sum(df['PREDICTION'] == df.name)
            incorrect = np.sum(df['PREDICTION'] != df.name)
            accuracy = correct / (correct + incorrect)
            windows = correct + incorrect
        return pd.DataFrame(data={'a': [windows], 'b': [accuracy]})

    if dataset is None:
        return None
    else:
        result = dataset.groupby(by=['GT_LABEL']).apply(
            _summarize).reset_index(drop=False)
        result = result[['GT_LABEL', 'a', 'b']]
        if 'PREDICTION' in dataset:
            accuracy = np.sum(dataset['PREDICTION'] ==
                              dataset['GT_LABEL']) / dataset.shape[0]
        else:
            accuracy = np.nan
        result = result.append(
            {'GT_LABEL': 'Total', 'a': dataset.shape[0], 'b': accuracy}, ignore_index=True)
        return result.values.tolist()


def extract_placement_features(feature_df, placement_names, group_col=[]):
    placement_features = []
    for placement in placement_names:
        placement_feature = feature_df.loc[
            feature_df['SENSOR_PLACEMENT'] == placement,
            ['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + group_col +
            muss.get_feature_names()
        ]
        placement_features.append(placement_feature)
    return placement_features


def train_model(origin_labels=None,
                origin_dataset=None,
                origin_model=None,
                new_labels=None,
                new_dataset=None,
                progress_queue=None,
                placement_names=['DW', 'DA', 'DT'], class_col='MUSS_22_ACTIVITY_ABBRS',
                strategy=Strategy.USE_ORIGIN_ONLY,
                pool=None):
    progress_queue = progress_queue or queue.Queue()

    pool = pool or pools.ProcessPool(nodes=1)
    pool.restart(force=True)

    if origin_model is not None:
        placement_names = origin_model[-2]
        origin_labels = origin_labels or origin_model[0].classes_

    if origin_dataset is not None:
        origin_feature, origin_class, origin_feature_names = prepare_origin_dataset(
            origin_dataset,
            origin_labels,
            placement_names,
            class_col,
            progress_queue
        )

    if new_dataset is not None:
        new_feature, new_class, new_feature_names = prepare_new_dataset(
            new_dataset,
            new_labels,
            class_col,
            progress_queue
        )

    progress_queue.put('Select training strategy...')
    if strategy == Strategy.USE_ORIGIN_ONLY:
        input_feature = origin_feature
        input_class = origin_class
        feature_names = origin_feature_names
    elif strategy == Strategy.USE_NEW_ONLY:
        input_feature = new_feature
        input_class = new_class
        feature_names = new_feature_names
    elif strategy == Strategy.REPLACE_ORIGIN:
        input_feature, input_class = replace_original_data(
            origin_feature,
            origin_class,
            origin_labels,
            new_feature,
            new_class,
            new_labels
        )
        feature_names = origin_feature_names
    elif strategy == Strategy.COMBINE_ORIGIN:
        input_feature, input_class = combine_original_data(
            origin_feature,
            origin_class,
            origin_labels,
            new_feature,
            new_class,
            new_labels
        )
        feature_names = origin_feature_names

    progress_queue.put('Training SVM classifier...')
    task = pool.apipe(muss.train_classifier, input_feature,
                      input_class, class_col=class_col, feature_names=feature_names, placement_names=placement_names)
    progress_queue.put(task)


def prepare_origin_dataset(dataset, labels, placement_names, class_col, progress_queue, group_col=[]):
    origin_feature = dataset[0]
    origin_class = dataset[1]

    progress_queue.put('Extracting training data for each placement...')
    placement_features = extract_placement_features(
        origin_feature, placement_names)

    progress_queue.put('Combining training data together...')
    combined_feature_set, combined_feature_names = muss.combine_features(
        *placement_features, placement_names=placement_names)

    cleared_class_set = origin_class[['HEADER_TIME_STAMP',
                                      'START_TIME', 'STOP_TIME'] + group_col + [class_col]]

    progress_queue.put('Synchronizing training data and class labels...')
    synced_feature, synced_class = muss.sync_feature_and_class(
        combined_feature_set, cleared_class_set)

    # only use training labels
    progress_queue.put('Filtering out unused class labels...')
    filter_condition = synced_class[class_col].isin(
        labels)
    input_feature = synced_feature.loc[filter_condition, :]
    input_class = synced_class.loc[filter_condition, :]
    return input_feature, input_class, combined_feature_names


def prepare_new_dataset(dataset, labels, class_col, progress_queue):
    progress_queue.put(
        'Preparing feature and classes for the new collected data...')
    new_feature = dataset.iloc[:, :-2]
    new_class = dataset.iloc[:, [0, 1, 2, -2]]
    new_class = new_class.rename(
        columns={'GT_LABEL': class_col})
    progress_queue.put('Filtering out unused class labels...')
    filter_condition = new_class[class_col].isin(
        labels)
    input_feature = new_feature.loc[filter_condition, :]
    input_class = new_class.loc[filter_condition, :]
    combined_feature_names = new_feature.columns[3:].values.tolist()
    return input_feature, input_class, combined_feature_names


def validate_model(origin_labels=None,
                   origin_dataset=None,
                   origin_model=None,
                   new_labels=None,
                   new_dataset=None,
                   progress_queue=None,
                   placement_names=['DW', 'DA'], class_col='MUSS_22_ACTIVITY_ABBRS',
                   strategy=Strategy.USE_ORIGIN_ONLY,
                   pool=None):
    progress_queue = progress_queue or queue.Queue()

    pool = pool or pools.ProcessPool(nodes=1)
    pool.restart(force=True)
    feature_names = None
    if origin_model is not None:
        placement_names = origin_model[-2]
        origin_labels = origin_labels or origin_model[0].classes_

    if origin_dataset is not None:
        origin_feature, origin_class, origin_feature_names = prepare_origin_dataset(
            origin_dataset,
            origin_labels,
            placement_names,
            class_col,
            progress_queue,
            group_col=['PID']
        )
        feature_names = origin_feature_names

    new_feature = None
    new_class = None
    if new_dataset is not None:
        new_feature, new_class, new_feature_names = prepare_new_dataset(
            new_dataset,
            new_labels,
            class_col,
            progress_queue
        )
        if feature_names is None:
            feature_names = new_feature_names
        new_feature = new_feature[
            ['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + feature_names]

    progress_queue.put('Validating SVM classifier...')

    task = pool.apipe(muss.validate_classifier, origin_feature,
                      origin_class, class_col=class_col, feature_names=feature_names, placement_names=placement_names, new_input_feature=new_feature, new_input_class=new_class, strategy=strategy, group_col='PID')
    progress_queue.put(task)


def replace_original_data(origin_feature, origin_class, origin_labels, new_feature, new_class, new_labels):
    labels_keep_in_origin = list(filter(
        lambda l: l not in new_labels, origin_labels))
    filter_condition = origin_class['MUSS_22_ACTIVITY_ABBRS'].isin(
        labels_keep_in_origin)
    origin_feature = origin_feature.loc[filter_condition, :]
    origin_class = origin_class.loc[filter_condition, :]

    input_feature = pd.concat(
        [origin_feature, new_feature], sort=False, join='outer')
    input_class = pd.concat([origin_class, new_class],
                            sort=False, join='outer')
    return input_feature, input_class


def combine_original_data(origin_feature, origin_class, origin_labels, new_feature, new_class, new_labels):
    labels_keep_in_origin = origin_labels
    filter_condition = origin_class['MUSS_22_ACTIVITY_ABBRS'].isin(
        labels_keep_in_origin)
    origin_feature = origin_feature.loc[filter_condition, :]
    origin_class = origin_class.loc[filter_condition, :]

    new_filter_condition = new_class['MUSS_22_ACTIVITY_ABBRS'].isin(new_labels)
    new_feature = new_feature.loc[new_filter_condition, :]
    new_class = new_class.loc[new_filter_condition, :]

    input_feature = pd.concat(
        [origin_feature, new_feature], sort=False, join='outer')
    input_class = pd.concat([origin_class, new_class],
                            sort=False, join='outer')
    return input_feature, input_class


def merge_features(feature_set, placement_names, muss, group_col=None):
    placement_features = []
    if group_col is None:
        selected_cols = [
            'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'] + muss.get_feature_names()
    else:
        selected_cols = [
            'HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME', group_col] + muss.get_feature_names()
    for placement in placement_names:
        place_features = feature_set.loc[feature_set['SENSOR_PLACEMENT']
                                         == placement, selected_cols]
        placement_features.append(place_features)
    combined_feature_set, combined_feature_names = muss.combine_features(
        *placement_features, placement_names=placement_names, group_col=group_col)
    return combined_feature_set, combined_feature_names


def get_confusion_matrix_figure(validation_result, fig=None):
    muss = MUSSModel()
    labels = validation_result[-2]
    fig = muss.get_confusion_matrix(
        validation_result[0], validation_result[1], labels=labels, graph=True, fig=fig)
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


def connect_devices(devices, model, placement_names=['DW', 'DA'], mode=PROCESSOR_MODE.TEST_ONLY, output_folder=None, pid=None, pool=None):
    logger.info('device addrs: ' + str(devices))
    logger.info('device placements: ' + str(placement_names))
    pool = pool or pools.ThreadPool(nodes=1)
    pool.restart(force=True)
    device_addrs = devices
    streams = []
    start_time = dt.datetime.now()
    output_folder = os.path.join(output_folder, 'data')
    if model is not None:
        placement_names = model[-2]
    kwargs = {}
    for addr, placement in zip(device_addrs, placement_names):
        generator = arus.plugins.metawear.MetaWearAccelDataGenerator(
            addr, sr=50, grange=8, buffer_size=100)
        segmentor = arus.segmentor.SlidingWindowSegmentor(window_size=4)
        stream = arus.Stream(generator, segmentor, name=placement)
        streams.append(stream)
        kwargs[placement] = {'sr': 50}
    if mode == PROCESSOR_MODE.TEST_ONLY:
        pipeline = muss.get_inference_pipeline(
            *streams, name='muss-pipeline', model=model, max_processes=2, scheduler='processes', **kwargs)
    else:
        pipeline = muss.get_data_collection_pipeline(
            *streams, name='muss-pipeline', model=model, max_processes=2, scheduler='processes', output_folder=output_folder, pid=pid, **kwargs
        )
    task = pool.apipe(pipeline.connect, start_time=start_time)
    return task


def disconnect_devices(pipeline, pool=None):
    pool = pool or pools.ThreadPool(nodes=1)
    pool.restart(force=True)
    task = pool.apipe(pipeline.stop)
    return task


def start_test_model(pipeline, pool=None):
    start_time = dt.datetime.now()
    pool = pool or pools.ThreadPool(nodes=1)
    pool.restart(force=True)
    task = pool.apipe(pipeline.process, start_time=start_time)
    return task


def stop_test_model(pipeline, pool=None):
    pool = pool or pools.ThreadPool(nodes=1)
    pool.restart(force=True)
    task = pool.apipe(pipeline.pause)
    return task


def save_annotation(current_annotation, output_folder, pid, session_name, pool=None):
    output_folder = os.path.join(output_folder, 'data')
    writer = arus.mh.MhealthFileWriter(
        output_folder, pid, hourly=False, date_folders=False)
    writer.set_for_annotation(session_name, 'ARUS')
    df = pd.DataFrame.from_dict(current_annotation)
    task = writer.write_csv(df, append=True, block=False)[0]
    return task


def play_sound(text, pool=None):
    pool = pool or pools.ThreadPool(nodes=1)
    pool.restart(force=True)
    return pool.apipe(playsound, text + '.mp3', block=True)


def get_nearby_devices(pool=None):
    pool = pool or pools.ThreadPool(nodes=1)
    pool.restart(force=True)
    scanner = arus.plugins.metawear.MetaWearScanner()
    task = pool.apipe(scanner.get_nearby_devices, max_devices=3)
    return task
