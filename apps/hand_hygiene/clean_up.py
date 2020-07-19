

import arus
import glob
import os
from alive_progress import alive_bar
from loguru import logger
import pandas as pd
import datetime
import shutil
import sync_sensor as sync

HAND_CLAPPING_TIME_OFFSETS = [
    (datetime.timedelta(seconds=1, milliseconds=260),
     datetime.timedelta(seconds=1, milliseconds=480)),
    (datetime.timedelta(seconds=2, milliseconds=40),
     datetime.timedelta(seconds=2, milliseconds=380)),
    (datetime.timedelta(seconds=2, milliseconds=860),
     datetime.timedelta(seconds=3, milliseconds=200)),
    (datetime.timedelta(seconds=3, milliseconds=680),
     datetime.timedelta(seconds=3, milliseconds=960)),
    (datetime.timedelta(seconds=4, milliseconds=500),
     datetime.timedelta(seconds=4, milliseconds=800)),
    (datetime.timedelta(seconds=5, milliseconds=380),
     datetime.timedelta(seconds=5, milliseconds=660))
]


def convert_to_mhealth(root, pid, skip_sync=False, remove_exists=True):
    master_folder = os.path.join(root, pid, arus.mh.MASTER_FOLDER)
    if remove_exists and os.path.exists(master_folder):
        logger.info(f'Remove existing {master_folder}')
        shutil.rmtree(master_folder)
    annot_df, task_annot_df = _convert_annotations(root, pid)
    _convert_sensors(root, pid, data_type='AccelerometerCalibrated',
                     skip_sync=skip_sync, annot_df=annot_df, task_annot_df=task_annot_df)
    _convert_sensors(root, pid, data_type='IMUTenAxes', skip_sync=skip_sync,
                     annot_df=annot_df, task_annot_df=task_annot_df)


def _convert_sensors(root, pid, data_type, skip_sync=True, annot_df=None, task_annot_df=None):
    logger.info(
        f"Convert {data_type} data to mhealth format for hand hygiene raw dataset")

    if data_type == 'AccelerometerCalibrated':
        filename_pattern = '*RAW.csv'
    elif data_type == 'IMUTenAxes':
        filename_pattern = '*IMU.csv'
    else:
        raise NotImplementedError(
            f'The data type {data_type} is not supported')

    sensor_files = glob.glob(os.path.join(
        root, pid, "OriginalRaw", filename_pattern), recursive=True)

    for sensor_file in sensor_files:
        logger.info('Convert {} to mhealth'.format(sensor_file))

        sensor_df, meta = _read_raw_actigraph(sensor_file)
        if not skip_sync and annot_df is not None and task_annot_df is not None:
            logger.info(
                "Start synchronizing this sensor to annotations based on hand clappings")
            sensor_df = sync._sync(sensor_df, annot_df, task_annot_df)
        else:
            logger.warning("Skip synchronization")

        if data_type == 'IMUTenAxes':
            _write_to_mhealth(root, pid, sensor_df, meta,
                              'IMUAccelerometerCalibrated')
            _write_to_mhealth(root, pid, sensor_df, meta,
                              'IMUTemperature')
            _write_to_mhealth(root, pid, sensor_df, meta,
                              'IMUGyroscope')
            _write_to_mhealth(root, pid, sensor_df, meta,
                              'IMUMagnetometer')
        else:
            _write_to_mhealth(root, pid, sensor_df, meta, data_type)


def _read_raw_actigraph(sensor_file):
    reader = arus.plugins.actigraph.ActigraphReader(sensor_file)
    read_iterator = reader.read(chunksize=None)
    meta = reader.get_meta()
    sensor_df = next(read_iterator.get_data())
    return sensor_df, meta


def _write_to_mhealth(root, pid, sensor_df, meta, data_type):
    writer = arus.mh.MhealthFileWriter(
        root, pid, hourly=True, date_folders=True)
    writer.set_for_sensor("ActigraphGT9X", data_type,
                          meta['SENSOR_ID'], version_code=meta['VERSION_CODE'].replace('.', ''))
    col_names = arus.mh.parse_column_names_from_data_type(data_type)
    chunk_with_selected_cols = sensor_df.loc[:, [
        arus.mh.TIMESTAMP_COL] + col_names]
    writer.write_csv(chunk_with_selected_cols, append=False, block=True)


def _convert_annotations(root, pid):
    logger.info(
        "Convert annotation data to mhealth format for hand hygiene raw dataset")
    raw_annotation_files = glob.glob(os.path.join(
        root, pid, "OriginalRaw", "**", "*annotations.csv"), recursive=True)
    annot_dfs = []
    task_annot_dfs = []
    with alive_bar(len(raw_annotation_files)) as bar:
        for raw_annotation_file in raw_annotation_files:
            bar.text('Convert {} to mhealth'.format(raw_annotation_file))
            annot_df, task_annot_df = _read_raw_annotation_file(
                raw_annotation_file)
            annot_dfs.append(annot_df)
            task_annot_dfs.append(task_annot_df)
    if len(annot_dfs) > 0:
        annot_df = pd.concat(annot_dfs, axis=0, ignore_index=True)
        annot_df.sort_values(by=annot_df.columns[0], inplace=True)
    else:
        annot_df = None
    if len(task_annot_dfs) > 0:
        task_annot_df = pd.concat(task_annot_dfs, axis=0, ignore_index=True)
        task_annot_df.sort_values(by=task_annot_df.columns[0], inplace=True)
    else:
        task_annot_df = None

    if annot_df is not None:
        writer = arus.mh.MhealthFileWriter(
            root, pid, hourly=True, date_folders=True)
        writer.set_for_annotation("HandHygiene", "App")
        writer.write_csv(annot_df, append=False, block=True)
    if task_annot_df is not None:
        writer = arus.mh.MhealthFileWriter(
            root, pid, hourly=True, date_folders=True)
        writer.set_for_annotation("HandHygieneTasks", "App")
        writer.write_csv(task_annot_df, append=False, block=True)

    return annot_df, task_annot_df


def _split_hand_clapping_annotations(start_time):
    start_time = pd.Timestamp(start_time).to_pydatetime()
    sts = []
    ets = []
    for offset_st, offset_et in HAND_CLAPPING_TIME_OFFSETS:
        st = start_time + offset_st
        et = start_time + offset_et
        sts.append(st)
        ets.append(et)
    return sts, ets


def _assemble_annotation_df(raw_annotations):
    label_names = raw_annotations['LABEL_NAME'].unique().tolist()
    dfs = []
    for label_name in label_names:
        start_times = []
        stop_times = []
        if '6 times' in label_name:
            old_start_times = raw_annotations.loc[(raw_annotations['LABEL_NAME'] == label_name) & (
                raw_annotations['EVENT_TYPE'] == "START"), 'HEADER_TIME_STAMP'].values
            # Only use the first hand clapping for syncing because the second hand clapping video has been changed between different sessions
            old_start_times = old_start_times[::2]
            for st in old_start_times:
                sts, ets = _split_hand_clapping_annotations(st)
                start_times = start_times + sts
                stop_times = stop_times + ets
        else:
            start_times = raw_annotations.loc[(raw_annotations['LABEL_NAME'] == label_name) & (
                raw_annotations['EVENT_TYPE'] == "START"), 'HEADER_TIME_STAMP'].values
            stop_times = raw_annotations.loc[(raw_annotations['LABEL_NAME'] == label_name) & (
                raw_annotations['EVENT_TYPE'] == "STOP"), 'HEADER_TIME_STAMP'].values
        if ":" in label_name:
            pruned_label_name = label_name.split(':')[1]
        else:
            pruned_label_name = label_name.split(' ')[1]
        label_df = pd.DataFrame(data={'HEADER_TIME_STAMP': start_times, 'START_TIME': start_times,
                                      'STOP_TIME': stop_times, 'LABEL_NAME': [pruned_label_name]*len(start_times)})
        dfs.append(label_df)
    if len(dfs) > 0:
        result_df = pd.concat(dfs, axis=0).sort_values(
            by=['HEADER_TIME_STAMP'])
    else:
        result_df = None
    return result_df


def _read_raw_annotation_file(filepath):
    raw_df = pd.read_csv(filepath, header=None,
                         infer_datetime_format=True, parse_dates=[0])
    raw_df.columns = ['HEADER_TIME_STAMP', 'LABEL_NAME', 'EVENT_TYPE']
    filter_condition = (raw_df['LABEL_NAME'].str.contains(
        'Collect ')) & (raw_df['LABEL_NAME'].str.contains(':'))
    task_filter_condition = (raw_df['LABEL_NAME'].str.contains(
        'Collect ')) & (~raw_df['LABEL_NAME'].str.contains(':'))
    raw_annotations = raw_df.loc[filter_condition, :]
    raw_task_annotations = raw_df.loc[task_filter_condition, :]
    logger.debug(raw_annotations)
    annot_df = _assemble_annotation_df(raw_annotations)
    task_annot_df = _assemble_annotation_df(raw_task_annotations)
    return annot_df, task_annot_df


if __name__ == "__main__":
    convert_to_mhealth('D:/datasets/hand_hygiene', 'P1-1',
                       skip_sync=False, remove_exists=True)
    arus.cli.convert_to_signaligner_both(
        'D:/datasets/hand_hygiene', 'P1-1', 80)  # , date_range=['2020-06-09'])
