import pandas as pd
import os
from loguru import logger
import alive_progress as progress
import datetime
import numpy as np
from . import actigraph


def signify_annotation_files(filepaths, data_id, output_path):
    dfs = []
    with progress.alive_bar(len(filepaths), bar='blocks') as bar:
        for filepath in filepaths:
            bar(text="Convert file to signaligner: {}".format(filepath))
            df = pd.read_csv(
                filepath, header=0, sep=',', compression="infer", quotechar='"', parse_dates=[0, 1, 2], infer_datetime_format=True)
            df = df[['START_TIME', 'STOP_TIME', 'LABEL_NAME']]
            df.rename(columns={'LABEL_NAME': 'PREDICTION'}, inplace=True)
            df['SOURCE'] = data_id
            df['LABELSET'] = data_id
            dfs.append(df)
    # merge and sort
    merged = pd.concat(dfs, axis=0)
    merged = merged.sort_values(by=['START_TIME'])
    merged.to_csv(output_path, index=False)


def signify_sensor_files(filepaths, data_id, output_path, output_annotation_path, session_span, sr):
    logger.debug(sr)
    hourly_markers = pd.date_range(session_span[0], session_span[1],
                                   freq='1H').to_pydatetime().tolist()
    n = len(hourly_markers)
    if os.path.exists(output_path):
        logger.info('Remove the existing file')
        os.remove(output_path)
    if os.path.exists(output_annotation_path):
        logger.info(
            'Remove the existing missing data annotation file')
        os.remove(output_annotation_path)
    last_row = None
    with progress.alive_bar(n, bar='blocks') as bar:
        for marker in hourly_markers:
            bar(text="Convert sensor data to signaligner format: {}".format(marker))
            date_str = "{}-{:02d}-{:02d}-{:02d}".format(marker.year,
                                                        marker.month, marker.day, marker.hour)
            selected_files = list(filter(lambda f: date_str in f, filepaths))
            if len(selected_files) == 0:
                filepath = None
            elif len(selected_files) == 1:
                filepath = selected_files[0]
            else:
                logger.warning(
                    'Multiple files found: {}'.format(selected_files))
                filepath = selected_files[0]
            hourly_df = _regularize_samples(marker, filepath, sr)

            if hourly_df.iloc[0, :].isna().any() and last_row is not None:
                hourly_df.iloc[0, :] = last_row.values
                last_row = None
            if hourly_df.iloc[-1, :].notna().all():
                last_row = hourly_df.iloc[-1, :]
            hourly_df = hourly_df.iloc[:-1, :]
            annotation_df = _data_to_annotation(hourly_df, sr=sr)
            actigraph.save_as_actigraph(
                hourly_df, output_path, session_span[0], session_span[1], sr=sr)
            if not os.path.exists(output_annotation_path):
                annotation_df.to_csv(output_annotation_path, mode='a',
                                     header=True, index=False)
            else:
                annotation_df.to_csv(output_annotation_path, mode='a',
                                     header=False, index=False)


def _regularize_samples(start_time, filepath=None, sr=50):
    freq = str(int(1000 / sr)) + 'L'
    tolerance = str(int(500 / sr)) + 'L'
    sample_ts = pd.date_range(start_time, start_time +
                              datetime.timedelta(hours=1), freq=freq)
    if filepath is None:
        out_df = sample_ts.to_frame(index=False)
        out_df.columns = ['HEADER_TIME_STAMP']
        out_df['X'] = np.nan
        out_df['Y'] = np.nan
        out_df['Z'] = np.nan
        out_df = out_df.set_index('HEADER_TIME_STAMP')
    else:
        input_data = pd.read_csv(
            filepath, header=0, index_col=None, infer_datetime_format=True, parse_dates=[0])
        input_data.columns = ['HEADER_TIME_STAMP', 'X', 'Y', 'Z']
        input_data = input_data.drop_duplicates(
            subset=['HEADER_TIME_STAMP'], keep='first')
        input_data = input_data.sort_values(by=['HEADER_TIME_STAMP'])
        input_data['HEADER_TIME_STAMP'] = input_data['HEADER_TIME_STAMP'] + \
            pd.Timedelta(1, unit='milliseconds')
        input_data = input_data.set_index('HEADER_TIME_STAMP')
        out_df = input_data.reindex(
            sample_ts, axis='index', method='nearest', tolerance=tolerance, limit=1)
        out_df.index.names = ['HEADER_TIME_STAMP']
    return out_df


def _data_to_annotation(hourly_df, sr=50):
    test = hourly_df['X'].copy(deep=True)
    test.loc[test.notna()] = 0
    test = test.fillna(1)
    edges = test.diff()
    if test.iloc[0] == 0:
        edges[0] = 0
    else:
        edges[0] = 1
    sts = hourly_df.loc[edges == 1].index.tolist()
    ets = hourly_df.loc[edges == -1].index.tolist()
    if len(sts) > len(ets):
        ets += [hourly_df.index[-1]]

    out_df = pd.DataFrame(data={
        'START_TIME': sts,
        'STOP_TIME': ets,
        'PREDICTION': "Missing",
        'SOURCE': "sensor",
        'LABELSET': "Missing"
    }, index=range(len(sts)))
    return out_df
