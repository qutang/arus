

import arus
import glob
import os
from alive_progress import alive_bar
from loguru import logger
import pandas as pd
import datetime
import shutil

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


def convert_to_mhealth(root, pid):
    _convert_sensors(root, pid)
    _convert_annotations(root, pid)


def _convert_sensors(root, pid):
    logger.info(
        "Convert sensor data to mhealth format for hand hygiene raw dataset")
    actigraph_files = glob.glob(os.path.join(
        root, pid, "OriginalRaw", "*RAW.csv"), recursive=True)

    master_folder = os.path.join(root, pid, arus.mh.MASTER_FOLDER)
    logger.info('Empty existing master synced folder')
    shutil.rmtree(master_folder)

    with alive_bar(len(actigraph_files)) as bar:
        for actigraph_file in actigraph_files:
            bar('Convert {} to mhealth'.format(actigraph_file))
            writer = arus.mh.MhealthFileWriter(
                root, pid, hourly=True, date_folders=True)

            reader = arus.plugins.actigraph.ActigraphReader(actigraph_file)
            reader.read_meta()
            meta = reader.get_meta()
            writer.set_for_sensor("ActigraphGT9X", "AccelerometerCalibrated",
                                  meta['SENSOR_ID'], version_code=meta['VERSION_CODE'].replace('.', ''))
            read_iterator = reader.read_csv(chunksize=None)
            for chunk in read_iterator.get_data():
                writer.write_csv(chunk, append=False, block=True)


def _convert_annotations(root, pid):
    logger.info(
        "Convert annotation data to mhealth format for hand hygiene raw dataset")
    raw_annotation_files = glob.glob(os.path.join(
        root, pid, "OriginalRaw", "**", "*annotations.csv"), recursive=True)
    with alive_bar(len(raw_annotation_files)) as bar:
        for raw_annotation_file in raw_annotation_files:
            bar('Convert {} to mhealth'.format(raw_annotation_file))
            result_df = _read_raw_annotation_file(raw_annotation_file)
            if result_df is None:
                continue
            writer = arus.mh.MhealthFileWriter(
                root, pid, hourly=True, date_folders=True)
            writer.set_for_annotation("HandHygiene", "App")
            writer.write_csv(result_df, append=False, block=True)


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


def _read_raw_annotation_file(filepath):
    raw_df = pd.read_csv(filepath, header=None,
                         infer_datetime_format=True, parse_dates=[0])
    raw_df.columns = ['HEADER_TIME_STAMP', 'LABEL_NAME', 'EVENT_TYPE']
    filter_condition = (raw_df['LABEL_NAME'].str.contains(
        'Collect ')) & (raw_df['LABEL_NAME'].str.contains(':'))
    raw_annotations = raw_df.loc[filter_condition, :]
    logger.debug(raw_annotations)
    label_names = raw_annotations['LABEL_NAME'].unique().tolist()
    dfs = []
    for label_name in label_names:
        start_times = []
        stop_times = []
        if '6 times' in label_name:
            old_start_times = raw_annotations.loc[(raw_annotations['LABEL_NAME'] == label_name) & (
                raw_annotations['EVENT_TYPE'] == "START"), 'HEADER_TIME_STAMP'].values
            for st in old_start_times:
                sts, ets = _split_hand_clapping_annotations(st)
                start_times = start_times + sts
                stop_times = stop_times + ets
        else:
            start_times = raw_annotations.loc[(raw_annotations['LABEL_NAME'] == label_name) & (
                raw_annotations['EVENT_TYPE'] == "START"), 'HEADER_TIME_STAMP'].values
            stop_times = raw_annotations.loc[(raw_annotations['LABEL_NAME'] == label_name) & (
                raw_annotations['EVENT_TYPE'] == "STOP"), 'HEADER_TIME_STAMP'].values
        pruned_label_name = label_name.split(':')[1]
        label_df = pd.DataFrame(data={'HEADER_TIME_STAMP': start_times, 'START_TIME': start_times,
                                      'STOP_TIME': stop_times, 'LABEL_NAME': [pruned_label_name]*len(start_times)})
        dfs.append(label_df)
    if len(dfs) > 0:
        result_df = pd.concat(dfs, axis=0).sort_values(
            by=['HEADER_TIME_STAMP'])
    else:
        result_df = None
    return result_df


if __name__ == "__main__":
    convert_to_mhealth('D:/datasets/hand_hygiene', 'P1')
    arus.cli.convert_to_signaligner_both(
        'D:/datasets/hand_hygiene', 'P1', 80, date_range=['2020-06-09'])
