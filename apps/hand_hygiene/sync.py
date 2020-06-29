

import arus
import glob
from alive_progress import alive_bar
from loguru import logger
import pandas as pd
from scipy import signal
import numpy as np
import datetime as dt
import os


def get_sync_periods(annot_df):
    condition = annot_df.iloc[:, 3] == 'SYNC_COLLECT'
    sync_annotations = annot_df.loc[condition, :].iloc[:, [1, 2]]
    sync_annotations['START_TIME'] = sync_annotations['START_TIME'] - \
        pd.Timedelta(np.timedelta64(3, 's'))
    sync_annotations['STOP_TIME'] = sync_annotations['STOP_TIME'] + \
        pd.Timedelta(np.timedelta64(3, 's'))
    return sync_annotations


def detect_claps(sensor_df):
    ts = sensor_df.iloc[:, 0]
    z_values = sensor_df.iloc[:, -1].values
    peak_indices = signal.find_peaks(z_values, 1.8, distance=40)
    logger.info(f'{len(peak_indices[0])} clap peaks found')
    while len(peak_indices[0]) != 12:
        logger.warning(
            "The number of peaks seem not correct, the peak finder setting may need to be adjusted!!!")
        height = input("Try a new peak height (g). Default is None.\n:")
        height = float(height) if height != "" else None
        height_threshold = input(
            "Try a new peak relative height (g). Default is None.\n:")
        height_threshold = float(
            height_threshold) if height_threshold != "" else None
        distance = input(
            "Try a new inter-peak distance (# of samples). Default is None.\n:")
        distance = int(distance) if distance != "" else None
        width = input(
            "Try a new peak width (# of samples). Default is None.\n:")
        width = int(width) if width != "" else None
        plateau_size = input(
            "Try a new peak plateau size (# of samples). Default is None.\n:")
        plateau_size = int(plateau_size) if plateau_size != "" else None
        peak_indices = signal.find_peaks(
            z_values, height=height, threshold=height_threshold, distance=distance, width=width, plateau_size=plateau_size)
        logger.info(f'{len(peak_indices[0])} clap peaks found')

    peak_ts = ts.iloc[peak_indices[0]]
    st = peak_ts - pd.Timedelta(np.timedelta64(50, 'ms'))
    et = peak_ts + pd.Timedelta(np.timedelta64(50, 'ms'))
    clap_df = pd.DataFrame(data={
        'HEADER_TIME_STAMP': st.tolist(),
        'START_TIME': st.tolist(),
        'STOP_TIME': et.tolist(),
        'LABEL_NAME': "Sync peaks"
    }, index=range(len(peak_ts)))
    return clap_df


def sync(root, pid):
    logger.info("Start synchronizing sensors and annotations")
    files = arus.mh.get_annotation_files(
        pid, dataset_path=root, annotation_type='HandHygieneTasks')
    logger.debug(files)
    annot_df = pd.concat([pd.read_csv(f, parse_dates=[0, 1, 2],
                                      infer_datetime_format=True) for f in files], axis=0)
    logger.debug(annot_df)
    sync_annotations = get_sync_periods(annot_df)
    for row in sync_annotations.itertuples(index=False):
        st = row.START_TIME
        et = row.STOP_TIME
        sensor_files = arus.mh.get_sensor_files(
            pid, root, given_date=st.to_pydatetime(), data_type='AccelerometerCalibrated')
        for sensor_file in sensor_files:
            logger.info(
                f"Analyze accelerometer data of {os.path.basename(sensor_file)}")
            reader = arus.mh.MhealthFileReader(sensor_file)
            reader.read_csv()
            sensor_df = next(reader.get_data())
            chunk = arus.ext.pandas.segment_by_time(
                sensor_df, seg_st=st, seg_et=et)
            clap_df = detect_claps(chunk)
            data_id = arus.mh.parse_data_id_from_filepath(sensor_file)
            writer = arus.mh.MhealthFileWriter(
                root, pid, hourly=True, date_folders=True)
            writer.set_for_annotation('HandClapPeaks', data_id)
            writer.write_csv(clap_df, append=False)


if __name__ == "__main__":
    sync('D:/datasets/hand_hygiene', pid='P2')
