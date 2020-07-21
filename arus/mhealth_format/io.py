"""Module to read and write files in mhealth format
"""


import pandas as pd
from . import helper
from . import constants
from .. import scheduler
import functools
import os
from loguru import logger
import numpy as np
from concurrent import futures


class MhealthFileReader:
    def __init__(self, filepath):
        self._filepath = filepath
        self._data = None
        self._iterator = None

    @staticmethod
    def read_csvs(*filepaths, datetime_cols=[0]):
        dfs = []
        for filepath in filepaths:
            reader = MhealthFileReader(filepath).read_csv(
                datetime_cols=datetime_cols)
            dfs.append(next(reader.get_data()))

        # combine and sort timestamps
        result = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
        result.sort_values(
            by=result.columns[datetime_cols[0]], inplace=True, ignore_index=True)
        return result

    def read_csv(self, chunksize=None, datetime_cols=[0]):
        """
        Known isuee:
        When used with writer, chunksize has to be None
        """
        reader = pd.read_csv(
            self._filepath, parse_dates=datetime_cols, infer_datetime_format=True, chunksize=chunksize, engine='c')
        if type(reader) == pd.DataFrame:
            self._data = reader
        else:
            self._iterator = reader
        return self

    def get_data(self):
        if self._data is not None:
            data = self._data.copy()
            data = helper.format_columns(
                data, filetype=constants.SENSOR_FILE_TYPE)
            yield data
        else:
            for data in self._iterator:
                data = helper.format_columns(
                    data, filetype=constants.SENSOR_FILE_TYPE)
                yield data


class MhealthFileWriter:
    def __init__(self, dataset_path, pid, hourly=False, date_folders=False):
        """
        Known issue: 
        hourly has to be set True
        """
        self._dataset_path = dataset_path
        self._pid = pid
        self._hourly = hourly
        self._date_folders = date_folders
        self._file_type = None
        self._executor = None

    def set_for_sensor(self, sensor_type, data_type, sensor_id, version_code):
        self._sensor_type = sensor_type
        self._data_type = data_type
        self._sensor_id = sensor_id
        self._version_code = version_code
        self._file_type = constants.SENSOR_FILE_TYPE

    def set_for_annotation(self, annotation_type, annotator):
        self._annotation_type = annotation_type
        self._annotator = annotator
        self._file_type = constants.ANNOTATION_FILE_TYPE

    def write_csv(self, data, append=False, block=True):
        writing_tasks = []
        groups = self._split_data(data)
        self._init_executor(len(groups))
        for group in groups:
            task = self._executor.submit(self._write_csv, group, append)
            writing_tasks.append(task)
        if block:
            results = self._executor.get_all_remaining_results()
            if len(results) != len(writing_tasks):
                raise IOError('Some chunks are not written to files correctly')
            else:
                logger.info('All chunks have been written to files.')
                return results
        else:
            return writing_tasks

    def _split_data(self, data):
        # TODO: Annotations should be cropped nicely at the edge of hourly files.
        if self._hourly:
            if self._file_type == constants.SENSOR_FILE_TYPE:
                group_key = constants.TIMESTAMP_COL
            elif self._file_type == constants.ANNOTATION_FILE_TYPE:
                group_key = constants.START_TIME_COL
            groups = data.groupby(pd.Grouper(key=group_key, freq='H'))
            groups = [group for name, group in groups]
            groups = list(filter(lambda group: not group.empty, groups))
        else:
            groups = [data]
        return groups

    def _init_executor(self, n_chunks):
        n = max(min(n_chunks, os.cpu_count() - 8), 1)
        self._executor = scheduler.Scheduler(
            mode=scheduler.Scheduler.Mode.THREAD, scheme=scheduler.Scheduler.Scheme.SUBMIT_ORDER, max_workers=n)

    def _get_output_filename(self, data):
        timestamp_str = helper.format_file_timestamp_from_data(
            data, self._file_type)
        if self._file_type == constants.SENSOR_FILE_TYPE:
            filename = "{}-{}-{}.{}-{}.{}.{}.csv".format(
                self._sensor_type, self._data_type,
                self._version_code, self._sensor_id, self._data_type, timestamp_str, self._file_type)
        else:
            filename = "{}.{}-{}.{}.{}.csv".format(
                self._annotation_type, self._annotator, self._annotation_type, timestamp_str, self._file_type)
        return filename

    def _get_output_folder(self, data):
        base_folder = os.path.join(self._dataset_path, self._pid,
                                   constants.MASTER_FOLDER)
        if self._date_folders:
            date_folder_path = helper.format_date_folder_path_from_data(
                data, self._file_type)
            output_folder = os.path.join(base_folder, date_folder_path)
        else:
            output_folder = base_folder
        return output_folder

    def _get_output_filepath(self, output_folder, output_filename, append):
        if append:
            existing_filenames = os.listdir(output_folder)
            for fname in existing_filenames:
                f1 = os.path.join(output_folder, fname)
                f2 = os.path.join(output_folder, output_filename)
                if helper.compare_two_mhealth_filepaths(
                        f1, f2):
                    return f1
        return os.path.join(output_folder, output_filename)

    def _write_csv(self, data, append):
        output_folder = self._get_output_folder(data)
        os.makedirs(output_folder, exist_ok=True)
        output_filename = self._get_output_filename(data)
        output_filepath = self._get_output_filepath(
            output_folder, output_filename, append)
        text = data.values
        header = ','.join(data.columns.values)

        if self._file_type == constants.SENSOR_FILE_TYPE:
            fmt = ['%.23s'] + ['%.6f'] * (len(data.columns) - 1)
        elif self._file_type == constants.ANNOTATION_FILE_TYPE:
            fmt = ['%.23s', '%.23s', '%.23s', '%s']
        if append == False or not os.path.exists(output_filepath):
            logger.debug(
                'Overwriting existing file for {} if there are any'.format(
                    output_filepath)
            )
            np.savetxt(output_filepath, text, delimiter=",",
                       fmt=fmt, header=header, comments='')
        else:
            logger.debug(
                'Appending to existing file for {} if there are any'.format(output_filepath))
            with open(output_filepath, 'ab') as f:
                np.savetxt(f, text, delimiter=",",
                           fmt=fmt)
        return output_filepath
