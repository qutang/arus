"""Module to read and write files in mhealth format
"""


import pandas as pd
from .path import extract_file_type
from .path import extract_existing_hourly_filepaths
from .path import build_mhealth_filename
from .path import build_mhealth_filepath
from .data import is_annotation_data
from .data import is_sensor_data
from .data import rename_columns
from .data import get_datetime_columns
from .data import convert_datetime_columns_to_string
from .data import convert_datetime_columns_to_datetime64ms
from .data import convert_string_columns_to_datetime64ms
from functools import partial
import os
from glob import glob
import logging
import numpy as np


def _is_large_file(filepath):
    size_in_bytes = os.path.getsize(filepath)
    size_in_mb = size_in_bytes / 1024 / 1024
    if filepath.endswith('gz'):
        threshold = 3
    else:
        threshold = 20
    if size_in_mb > threshold:
        return True
    else:
        return False


def read_data_csv(filepath, chunksize=None, iterator=False):
    file_type = extract_file_type(filepath)
    result = None if filepath is None else pd.read_csv(
        filepath, parse_dates=get_datetime_columns(file_type), infer_datetime_format=True, chunksize=chunksize, iterator=iterator, engine='c')
    if not iterator:
        return rename_columns(result, file_type)
    else:
        return result

def read_meta_csv(filepath):
    result = None if filepath is None else pd.read_csv(filepath)
    return result


def write_data_csv_no_pandas_(df, output_filepath, append=False, file_type='sensor'):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    text = df.values
    header = ','.join(df.columns.values)
    if file_type == 'sensor':
        fmt = ['%s'] + ['%.3f'] * (len(df.columns) - 1)
    elif file_type == 'annotation':
        fmt = '%s'
    if append == False:
        logging.debug('saving {} with no appending'.format(output_filepath))
        np.savetxt(output_filepath, text, delimiter=",",
                   fmt=fmt, header=header, comments='')
    else:
        logging.debug('saving {} with appending'.format(output_filepath))
        with open(output_filepath, 'ab') as f:
            np.savetxt(f, text, delimiter=",",
                       fmt=fmt)


def write_data_csv_(df, output_filepath, append=False):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    if append == False:
        df.to_csv(output_filepath, index=False, float_format='%.3f')
    else:
        df.to_csv(output_filepath, index=False, header=False,
                  float_format='%.3f', mode='a')


def write_data_csv(df, output_folder, pid, file_type, *,
                   sensor_or_annotation_type='Unknown',
                   data_type='Unknown',
                   version_code='NA',
                   sensor_or_annotator_id='XXX',
                   split_hours=False,
                   flat=False, append=False):

    def _get_existing_or_new_hourly_file(output_filepath):
        existing_files = extract_existing_hourly_filepaths(output_filepath)
        if len(existing_files) > 0:
            existing_files.sort()
            return existing_files[0]
        else:
            return output_filepath

    def _get_saver(path_generator, filename_generator, file_type):
        def _saver(d):
            if file_type == 'sensor':
                timestamp = d['HEADER_TIME_STAMP'].values[0]
            elif file_type == 'annotation':
                timestamp = d['START_TIME'].values[0]
            output_filepath = path_generator(
                timestamp=timestamp, filename=filename_generator(timestamp))
            d = convert_datetime_columns_to_string(d, file_type=file_type)
            if append:
                output_filepath = _get_existing_or_new_hourly_file(
                    output_filepath)
                if os.path.exists(output_filepath):
                    write_data_csv_no_pandas_(
                        d, output_filepath, append=True, file_type=file_type)
                else:
                    write_data_csv_no_pandas_(
                        d, output_filepath, file_type=file_type)
            else:
                write_data_csv_no_pandas_(
                    d, output_filepath, file_type=file_type)
        return _saver

    build_filename_with_ts = partial(build_mhealth_filename, file_type=file_type, sensor_or_annotation_type=sensor_or_annotation_type,
                                     data_type=data_type, version_code=version_code, sensor_or_annotator_id=sensor_or_annotator_id)

    build_filepath = partial(build_mhealth_filepath,
                             rootpath=output_folder, pid=pid, flat=flat)

    saver = _get_saver(build_filepath, build_filename_with_ts, file_type)

    if split_hours:
        if file_type == 'sensor':
            group_key = 'HEADER_TIME_STAMP'
        elif file_type == 'annotation':
            group_key = 'START_TIME'
        groups = df.groupby(pd.Grouper(key=group_key, freq='H'))
        for name, group in groups:
            logging.info('saving {} with {}'.format(name, str(group.shape)))
            saver(group)
    else:
        saver(df)


def read_actigraph_csv(filepath, chunksize=None, iterator=False):
    def format_actigraph_csv(df):
        convert_datetime = partial(
            convert_string_columns_to_datetime64ms, file_type='actigraph')
        rename = partial(rename_columns, file_type='sensor')
        return rename(convert_datetime(df))

    if filepath is None:
        result = None
    else:
        reader = pd.read_csv(
            filepath, skiprows=10, engine='c', chunksize=chunksize, iterator=iterator)
        format_as_mhealth = format_actigraph_csv
        result = reader, format_as_mhealth
    return result


def read_actigraph_meta(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline()
        second_line = f.readline()
        firmware = list(
            filter(lambda token: token.startswith('v'), first_line.split(" ")))[1]
        sr = int(
            list(filter(lambda token: token.isnumeric(), first_line.split(" ")))[0])
        sid = second_line.split(" ")[-1].strip()
    return {
        'VERSION_CODE': firmware,
        'SAMPLING_RATE': sr,
        'SENSOR_ID': sid
    }
