import pandas as pd
import os
from loguru import logger
import tqdm
import datetime
import numpy as np
from . import actigraph
from .. import mhealth_format as mh
from ..extensions.pandas import segment_by_time
import math
import enum


class FileType(enum.Enum):
    SENSOR = enum.auto(),
    ANNOTATION = enum.auto()


def format_time(ts):
    s = ts.strftime('%Y-%m-%d %H:%M:%S')
    ms = '{:03.0f}'.format(math.floor(ts.microsecond / 1000.0))
    result = "{}.{}".format(s, ms)
    return result


def signify_annotation_dataframes(*dfs, data_id, output_path, session_span=None):
    # merge and sort
    merged = pd.concat(dfs, axis=0)
    merged = merged.sort_values(by=['START_TIME'])
    merged = merged.loc[merged['START_TIME']
                        != merged['STOP_TIME'], :]
    # segment by session span
    if session_span is not None:
        segmented = segment_by_time(
            merged, seg_st=session_span[0], seg_et=session_span[1], st_col=0, et_col=1)
    else:
        segmented = merged

    # save as signaligner label file
    save_as_signaligner(segmented, output_path,
                        FileType.ANNOTATION, labelset=data_id, mode='w', index=False, header=True)


def signify_annotation_files(filepaths, data_id, output_path, session_span):
    dfs = []
    with tqdm.tqdm(total=len(filepaths)) as bar:
        for filepath in filepaths:
            bar.set_description(
                "Convert file to signaligner: {}".format(filepath))
            df = pd.read_csv(
                filepath, header=0, sep=',', compression="infer", quotechar='"', parse_dates=[0, 1, 2], infer_datetime_format=True)
            dfs.append(df)
            bar.update()
    signify_annotation_dataframes(
        *dfs, data_id=data_id, output_path=output_path, session_span=session_span)


def auto_split_session_span(session_span, auto_range='W-SUN'):
    sub_sessions = pd.date_range(session_span[0], session_span[1],
                                 freq=auto_range, normalize=True).to_pydatetime().tolist()
    if len(sub_sessions) == 0:
        sub_sessions = list(session_span)
    else:
        if sub_sessions[0].strftime('%Y-%m-%d-%H') != session_span[0].strftime('%Y-%m-%d-%H'):
            sub_sessions = [session_span[0]] + sub_sessions
        if sub_sessions[-1].strftime('%Y-%m-%d-%H') != session_span[1].strftime('%Y-%m-%d-%H'):
            sub_sessions += [session_span[1]]
    return sub_sessions


def shrink_session_span(session_span, date_range=None):
    if date_range is None:
        return session_span
    else:
        if len(date_range) == 1:
            st = datetime.datetime.strptime(date_range[0], '%Y-%m-%d')
            et = session_span[1]
        elif len(date_range) == 2:
            if date_range[0] == '':
                if len(date_range[1].split('-')) == 3:
                    et = datetime.datetime.strptime(date_range[1], '%Y-%m-%d')
                elif len(date_range[1].split('-')) == 4:
                    et = datetime.datetime.strptime(
                        date_range[1], '%Y-%m-%d-%H')
                st = session_span[0]
            else:
                if len(date_range[0].split('-')) == 3:
                    st = datetime.datetime.strptime(date_range[0], '%Y-%m-%d')
                elif len(date_range[0].split('-')) == 4:
                    st = datetime.datetime.strptime(
                        date_range[0], '%Y-%m-%d-%H')
                if len(date_range[1].split('-')) == 3:
                    et = datetime.datetime.strptime(date_range[1], '%Y-%m-%d')
                elif len(date_range[1].split('-')) == 4:
                    et = datetime.datetime.strptime(
                        date_range[1], '%Y-%m-%d-%H')
        if st > session_span[1] or et < session_span[0]:
            logger.warning(
                'Input date range is beyond the available date range of the dataset, ignore it')
            return session_span
        st = max(session_span[0], st or session_span[0])
        et = min(session_span[1], et or session_span[1])
        return st, et


def signify_sensor_files(filepaths, data_id, output_path, output_annotation_path, session_span, sr):
    hourly_markers = pd.date_range(session_span[0], session_span[1],
                                   freq='1H', closed='left').to_pydatetime().tolist()
    n = len(hourly_markers)
    if os.path.exists(output_path):
        logger.info('Remove the existing file')
        os.remove(output_path)
    if os.path.exists(output_annotation_path):
        logger.info(
            'Remove the existing missing data annotation file')
        os.remove(output_annotation_path)
    last_row = None
    with tqdm.tqdm(total=n) as bar:
        for marker in hourly_markers:
            bar.set_description(
                "Convert sensor data to signaligner format: {}".format(marker))
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
            hourly_df = _regularize_samples(marker, filepath, sr, data_id)

            if hourly_df.iloc[0, :].isna().any() and last_row is not None:
                hourly_df.iloc[0, :] = last_row.values
                last_row = None
            if hourly_df.iloc[-1, :].notna().all():
                last_row = hourly_df.iloc[-1, :]
            hourly_df = hourly_df.iloc[:-1, :]
            save_as_signaligner(hourly_df,
                                output_path,
                                file_type=FileType.SENSOR,
                                sid=data_id.split('-')[0],
                                session_st=session_span[0], session_et=session_span[1],
                                sr=sr,
                                mode='a',
                                index=False,
                                header=False,
                                float_format='%.6f')

            annotation_df = _data_to_annotation(hourly_df, sr=sr)
            if not os.path.exists(output_annotation_path):
                save_as_signaligner(annotation_df, output_annotation_path,
                                    FileType.ANNOTATION, labelset=data_id, mode='w', header=True, index=False)
            else:
                save_as_signaligner(annotation_df, output_annotation_path,
                                    FileType.ANNOTATION, labelset=data_id, mode='a', header=False, index=False)
            bar.update()


def _regularize_samples(start_time, filepath=None, sr=50, data_id=None):
    freq = str(1000 / sr) + 'L'
    tolerance = str(500 / sr) + 'L'
    sample_ts = pd.date_range(start_time, start_time +
                              datetime.timedelta(hours=1), freq=freq)

    if data_id is not None:
        data_type = data_id.split('-')[1]
        col_names = mh.parse_column_names_from_data_type(data_type)
    if filepath is None:
        out_df = sample_ts.to_frame(index=False)
        out_df.columns = ['HEADER_TIME_STAMP']
        for col_name in col_names:
            out_df[col_name] = np.nan
        out_df = out_df.set_index('HEADER_TIME_STAMP')
    else:
        input_data = pd.read_csv(
            filepath, header=0, index_col=None, infer_datetime_format=True, parse_dates=[0])
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
    test = hourly_df.iloc[:, 0].copy(deep=True)
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

    out_df = pd.DataFrame.from_dict({
        'START_TIME': sts,
        'STOP_TIME': ets,
        'PREDICTION': ["Missing"]*len(sts)
    })
    return out_df


def save_as_signaligner(df, output_path, file_type: FileType, **kwargs):
    if file_type == FileType.SENSOR:
        actigraph.save_as_actigraph(df, output_path, **kwargs)
    elif file_type == FileType.ANNOTATION:
        _save_annotation(df, output_path, **kwargs)


def _save_annotation(df, output_path, labelset, **kwargs):
    if 'START_TIME' not in df or 'STOP_TIME' not in df or ('LABEL_NAME' not in df and 'PREDICTION' not in df):
        raise ValueError('Input dataframe does not have proper columns.')
    if 'LABEL_NAME' in df:
        predictions = df['LABEL_NAME']
    elif 'PREDICTION' in df:
        predictions = df['PREDICTION']
    converted = {
        'START_TIME': df['START_TIME'].apply(format_time),
        'STOP_TIME': df['STOP_TIME'].apply(format_time),
        'PREDICTION': predictions,
        'SOURCE': ['Algo'] * df.shape[0],
        'LABELSET': [labelset] * df.shape[0]}
    converted = pd.DataFrame.from_dict(converted)
    converted.to_csv(output_path, **kwargs)
