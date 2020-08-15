from .. import generator
from .. import mhealth_format as mh
import pandas as pd
from pyarrow import csv
import datetime
import os

ACTIGRAPH_TEMPLATE = """------------ Data File Created By ActiGraph GT3X+ ActiLife v6.13.3 Firmware v2.5.0 date format M/d/yyyy at {} Hz  Filter Normal -----------
Serial Number: {}
Start Time {}
Start Date {}
Epoch Period (hh:mm:ss) 00:00:00
Download Time {}
Download Date {}
Current Memory Address: 0
Current Battery Voltage: 4.21     Mode = 12
--------------------------------------------------"""


class ActigraphSensorFileGenerator(generator.Generator):
    def __init__(self, *filepaths, has_ts=True, has_header=True, **kwargs):
        super().__init__(**kwargs)
        self._filepaths = filepaths
        self._reader = None
        self._has_ts = has_ts
        self._has_header = has_header

    def run(self, values=None, src=None, context={}):
        for filepath in self._filepaths:
            self._reader = ActigraphReader(
                filepath, self._has_ts, self._has_header)
            self._reader.read(chunksize=self._buffer_size)
            for data in self._reader.get_data():
                if self._stop:
                    break
                result = self._buffering(data)
                if result is not None:
                    self._result.put((result, self._context))
            if self._stop:
                break
        self._result.put((None, self._context))


class ActigraphReader:
    def __init__(self, filepath, has_ts=True, has_header=True):
        self._filepath = filepath
        self._data = None
        self._iterator = None
        self._meta = None
        self._has_ts = has_ts
        self._has_header = has_header

    def get_data(self):
        assert self._meta is not None
        ts_func = convert_actigraph_imu_timestamp if self._meta[
            'IMU'] else convert_actigraph_timestamp
        if self._data is not None:
            data = self._data.copy()
            if self._has_ts:
                data.iloc[:, 0] = ts_func(data.iloc[:, 0])
            else:
                ts = generate_timestamps(
                    self._meta['START_TIME'], self._meta['SAMPLING_RATE'], data.shape[0])[:-1]
                data.insert(0, 'ts', ts)
            data = mh.helper.format_columns(
                data, filetype=mh.constants.SENSOR_FILE_TYPE)
            yield data
        else:
            st = self._meta['START_TIME']
            for data in self._iterator:
                if self._has_ts:
                    data.iloc[:, 0] = ts_func(data.iloc[:, 0])
                else:
                    ts = generate_timestamps(
                        st, self._meta['SAMPLING_RATE'], data.shape[0])
                    data.insert(0, 'ts', ts[:-1])
                    st = ts[-1]
                data = mh.helper.format_columns(
                    data, filetype=mh.constants.SENSOR_FILE_TYPE)
                yield data

    def get_meta(self):
        return self._meta.copy()

    def read(self, **kwargs):
        self.read_meta()
        self.read_csv(**kwargs)
        return self

    def read_csv(self, chunksize=None):
        if self._has_ts:
            columns = ['HEADER_TIME_STAMP', 'X', 'Y', 'Z']
        else:
            columns = ['X', 'Y', 'Z']
        if self._has_header:
            header = 0
            columns = None
        else:
            header = None
        if chunksize is None:
            read_opts = csv.ReadOptions(skip_rows=10, column_names=columns)
            reader = csv.read_csv(self._filepath, read_options=read_opts)
            reader = reader.to_pandas()
        else:
            reader = pd.read_csv(
                self._filepath, skiprows=10, chunksize=chunksize, header=header, names=columns, engine='c', memory_map=True)
        if type(reader) == pd.DataFrame:
            self._data = reader
        else:
            self._iterator = reader
        return self

    def read_meta(self):
        with open(self._filepath, 'r') as f:
            first_line = f.readline()
            second_line = f.readline()
            third_line = f.readline()
            fourth_line = f.readline()
            is_imu = 'IMU' in first_line
            firmware = list(
                filter(lambda token: token.startswith('v'), first_line.split(" ")))[1]
            sr = int(
                list(filter(lambda token: token.isnumeric(), first_line.split(" ")))[0])
            sid = second_line.split(" ")[-1].strip()
        if 'TAS' in sid and is_imu:
            g_range = 16
        elif 'TAS' in sid and not is_imu:
            g_range = 8
        else:
            g_range = None

        st_str = fourth_line.split(' ')[2] + ' ' + third_line.split(' ')[2]
        st = pd.Timestamp(st_str)

        self._meta = {
            'START_TIME': st,
            'VERSION_CODE': firmware,
            'SAMPLING_RATE': sr,
            'SENSOR_ID': sid,
            'IMU': is_imu,
            'DYNAMIC_RANGE': g_range
        }
        return self


def generate_timestamps(start_time, sr, n):
    return pd.date_range(start=start_time, periods=n + 1, freq=str(1000.0/sr) + 'L')


def convert_actigraph_timestamp(timestamps):
    """Convert elements in the timestamp columns of the input dataframe to `datetime64[ms]` type.

    Args:
        timestamps

    Returns:
        result (same as input)
    """

    result = pd.to_datetime(
        timestamps, format='%m/%d/%Y %H:%M:%S.%f')
    result = result.astype('datetime64[ms]')
    return result


def convert_actigraph_imu_timestamp(timestamps):
    result = pd.to_datetime(timestamps, infer_datetime_format=True)
    result = result.astype('datetime64[ms]')
    return result


def save_as_actigraph(out_df, output_filepath, sid=None, session_st=None, session_et=None, sr=50, **kwargs):
    sid = sid or 'CLE2B2013XXXX'
    session_st = session_st or out_df.iloc[0, 0].to_datetime()
    session_et = session_et or out_df.iloc[-1, 0].to_datetime()
    meta_sdate_str = '{dt.month}/{dt.day}/{dt.year}'.format(
        dt=session_st)
    meta_stime_str = session_st.strftime('%H:%M:%S')
    meta_edate_str = '{dt.month}/{dt.day}/{dt.year}'.format(
        dt=session_et)
    meta_etime_str = (session_et +
                      datetime.timedelta(hours=1)).strftime('%H:%M:%S')
    col_names = out_df.columns
    col_names = list(map(lambda name: _format_column_name(name), col_names))
    if not os.path.exists(output_filepath):
        # create
        with open(output_filepath, mode='w', encoding='utf-8') as f:
            f.write(ACTIGRAPH_TEMPLATE.format(
                sr, sid, meta_stime_str, meta_sdate_str, meta_etime_str, meta_edate_str))
            f.write('\n')
    # append
    out_df.to_csv(output_filepath, **kwargs)


def _format_column_name(name):
    tokens = name.split('_')
    tokens[0] = tokens[0].lower().title()
    return " ".join(tokens)
