

import arus
import glob
from loguru import logger
import pandas as pd
from scipy import signal
import numpy as np
import datetime as dt
import os


def _get_sync_periods(annot_df):
    condition = annot_df.iloc[:, 3] == 'SYNC_COLLECT'
    sync_annotations = annot_df.loc[condition, :].iloc[:, [1, 2]]
    sync_annotations['START_TIME'] = sync_annotations['START_TIME'] - \
        pd.Timedelta(np.timedelta64(3, 's'))
    sync_annotations['STOP_TIME'] = sync_annotations['STOP_TIME'] + \
        pd.Timedelta(np.timedelta64(3, 's'))
    return sync_annotations


def _detect_claps(sync_sensor_df):
    ts = sync_sensor_df.iloc[:, 0]
    vm_values = arus.ext.numpy.vector_magnitude(
        sync_sensor_df.iloc[:, 1:4])[:, 0]
    height = 3
    distance = 40
    height_threshold = None
    width = None
    plateau_size = None
    peak_indices = signal.find_peaks(vm_values, height, distance=distance)
    logger.info(f'{len(peak_indices[0])} clap peaks found')
    while len(peak_indices[0]) != 12:
        logger.warning(
            "The number of peaks seem not correct, the peak finder setting may need to be adjusted!!!")
        logger.warning(
            f'Current peak finding settings: height={height}, distance={distance}, threshold={height_threshold}, width={width}, plateau_size={plateau_size}')
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
            vm_values, height=height, threshold=height_threshold, distance=distance, width=width, plateau_size=plateau_size)
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


def _get_average_sensor_offset(annot_df, peak_annot_df):
    # Only use the last three peaks for the first hand clapping activities during sync task
    peaks_from_sensor = peak_annot_df.iloc[[2, 3, 4], :]
    peaks_from_annot = annot_df.loc[annot_df['LABEL_NAME'].str.contains(
        '6 times'), :].iloc[[2, 3, 4], :]

    peaks_ts_from_sensor = peaks_from_sensor['START_TIME'] + (
        peaks_from_sensor['STOP_TIME'] - peaks_from_sensor['START_TIME']) / 2.0
    peaks_ts_from_annot = peaks_from_annot['START_TIME']
    peaks_ts_from_sensor.reset_index(drop=True, inplace=True)
    peaks_ts_from_annot.reset_index(drop=True, inplace=True)
    average_offset = (peaks_ts_from_annot - peaks_ts_from_sensor).mean()
    logger.info(f'Timestamp offset for this sensor is: {average_offset}')
    return average_offset


def _sync_sensor_to_annotations(sensor_df, average_offset):
    sensor_df['HEADER_TIME_STAMP'] = sensor_df['HEADER_TIME_STAMP'] + \
        average_offset
    return sensor_df


def _sync(sensor_df, annot_df, task_annot_df):
    sync_annots = _get_sync_periods(task_annot_df)
    average_offsets = []
    logger.info(
        f'Found {sync_annots.shape[0]} synchronization markers. Analyzing them...')
    for row in sync_annots.itertuples(index=False):
        st = row.START_TIME
        et = row.STOP_TIME
        sync_sensor_df = arus.ext.pandas.segment_by_time(
            sensor_df, seg_st=st, seg_et=et)
        sync_annot_df = arus.ext.pandas.segment_by_time(
            annot_df, seg_st=st, seg_et=et, st_col=1, et_col=2)
        if sync_sensor_df.empty:
            logger.warning(
                "Did not find corresponding sensor data for the current synchronization markers.")
        else:
            sync_peak_df = _detect_claps(sync_sensor_df)
            average_offset = _get_average_sensor_offset(
                sync_annot_df, sync_peak_df)
            average_offsets.append(average_offset)
    average_offset = np.mean(average_offsets)
    logger.info(f'The sensor offsets are: {average_offsets}')
    logger.info(f'The average sensor offset is: {average_offset}')
    sensor_df = _sync_sensor_to_annotations(
        sensor_df, average_offset)
    return sensor_df
