"""Module to read and write files in mhealth format
"""


import pandas as pd
from .path import extract_file_type
from .path import build_mhealth_filename
from .path import build_mhealth_filepath
from .data import is_annotation_data
from .data import is_sensor_data
from .data import rename_columns
from .data import get_datetime_columns
from .data import convert_datetime_columns_to_string
from functools import partial
import os


def read_data_csv(filepath):
    file_type = extract_file_type(filepath)
    result = None if filepath is None else pd.read_csv(
        filepath, parse_dates=get_datetime_columns(file_type), infer_datetime_format=True)
    assert is_sensor_data(result) or is_annotation_data(result)
    return rename_columns(result, file_type)


def read_meta_csv(filepath):
    result = None if filepath is None else pd.read_csv(filepath)
    return result


def write_data_csv_(df, output_filepath, append=False):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    if append == False:
        df.to_csv(output_filepath, index=False)
    else:
        df.to_csv(output_filepath, index=False, header=False, mode='a')


def write_data_csv(df, output_folder, pid, file_type, *,
                   sensor_or_annotation_type='Unknown',
                   data_type='Unknown',
                   version_code='NA',
                   sensor_or_annotator_id='XXX',
                   split_hours=False,
                   flat=False):

    def _get_saver(path_generator, filename_generator, file_type):
        def _saver(d):
            if file_type == 'sensor':
                timestamp = d['HEADER_TIME_STAMP'].values[0]
            elif file_type == 'annotation':
                timestamp = d['START_TIME'].values[0]
            output_filepath = path_generator(
                timestamp=timestamp, filename=filename_generator(timestamp))
            d = convert_datetime_columns_to_string(d, file_type=file_type)
            write_data_csv_(d, output_filepath)
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
        df.groupby(pd.Grouper(key=group_key, freq='H')).apply(saver)
    else:
        saver(df)
