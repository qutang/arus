

import arus
import glob
import os
from alive_progress import alive_bar
from loguru import logger
import pandas as pd


def convert_to_mhealth(root, pid):
    # _convert_sensors(root, pid)
    _convert_annotations(root, pid)


def _convert_sensors(root, pid):
    logger.info(
        "Convert sensor data to mhealth format for hand hygiene raw dataset")
    actigraph_files = glob.glob(os.path.join(
        root, pid, "OriginalRaw", "*RAW.csv"), recursive=True)
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
            writer = arus.mh.MhealthFileWriter(
                root, pid, hourly=True, date_folders=True)
            writer.set_for_annotation("HandHygiene", "App")
            writer.write_csv(result_df, append=False, block=True)


def _read_raw_annotation_file(filepath):
    raw_df = pd.read_csv(filepath, header=None,
                         infer_datetime_format=True, parse_dates=[0])
    raw_df.columns = ['HEADER_TIME_STAMP', 'LABEL_NAME', 'EVENT_TYPE']
    filter_condition = raw_df['LABEL_NAME'].str.contains('Collect ')
    raw_annotations = raw_df.loc[filter_condition, :]
    logger.debug(raw_annotations)
    label_names = raw_annotations['LABEL_NAME'].unique().tolist()
    dfs = []
    for label_name in label_names:
        start_times = []
        stop_times = []
        start_times = raw_annotations.loc[(raw_annotations['LABEL_NAME'] == label_name) & (
            raw_annotations['EVENT_TYPE'] == "START"), 'HEADER_TIME_STAMP'].values
        stop_times = raw_annotations.loc[(raw_annotations['LABEL_NAME'] == label_name) & (
            raw_annotations['EVENT_TYPE'] == "STOP"), 'HEADER_TIME_STAMP'].values
        pruned_label_name = label_name.split(':')[1]
        label_df = pd.DataFrame(data={'HEADER_TIME_STAMP': start_times, 'START_TIME': start_times,
                                      'STOP_TIME': stop_times, 'LABEL_NAME': [pruned_label_name]*len(start_times)})
        dfs.append(label_df)
    result_df = pd.concat(dfs, axis=0).sort_values(by=['HEADER_TIME_STAMP'])
    return result_df


if __name__ == "__main__":
    # convert_to_mhealth('D:/datasets/hand_hygiene', 'P1')
    arus.cli.convert_to_signaligner_both(
        'D:/datasets/hand_hygiene', 'P1', 80, None)
