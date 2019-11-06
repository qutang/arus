import pandas as pd
import numpy as np
import datetime
from functools import partial


def is_sensor_data(df):
    """Validate if the given sensor dataframe matches mhealth format

    Args:
        df (pandas.DataFrame): The input dataframe to be validated

    Returns:
        is_from_sensor (bool): `True` if the input sensor dataframe meets mhealth format specification.
    """
    # numerical from the second column
    is_from_sensor = True
    arr = df.iloc[:, 1:].values
    is_from_sensor = is_from_sensor and (np.issubdtype(
        arr.dtype, np.float) or np.issubdtype(arr.dtype, np.integer))
    # first column is timestamp
    ts = df.iloc[:, 0].values
    is_from_sensor = is_from_sensor and ts.dtype.type == np.datetime64
    return is_from_sensor


def is_annotation_data(df):
    """Validate if the given annotation dataframe matches mhealth format

    Args:
        df (pandas.DataFrame): The input dataframe to be validated

    Returns:
        is_from_annotation (bool): `True` if the input annotation dataframe meets mhealth format specification.
    """
    # label name should be string
    is_from_annotation = True
    arr = df.values[:, 3]
    is_from_annotation = is_from_annotation and (
        arr.dtype.type == np.unicode_ or arr.dtype.type == np.string_ or arr.dtype.type == np.object_)
    # first three columns are timestamp
    ts = df.values[:, 0:2]
    is_from_annotation = is_from_annotation and ts.dtype.type == np.datetime64
    return is_from_annotation


def get_datetime_columns(file_type):
    """Utility to get the timestamp column indices given file type

    Args:
        file_type (str): mhealth file type. Now only support `sensor` and `annotation`.

    Returns:
        col_indices (list): list of column indices (0 based)
    """
    if file_type == 'sensor':
        return [0]
    elif file_type == 'annotation':
        return [0, 1, 2]
    else:
        raise NotImplementedError(
            'The given file type {} is not supported'.format(file_type))


def convert_datetime_columns_to_string(df, file_type):
    """Convert elements in the timestamp columns of the input mhealth dataframe to human readable strings

    Args:
        df (pandas.DataFrame): The input mhealth dataframe. Timestamp columns should always be in `np.datetime64` type.
        file_type (str): mhealth file type. Now only support `sensor` and `annotation`.

    Returns:
        result (pandas.DataFrame): Dataframe with timestamp columns converted.
    """
    def _to_timestamp_string(arr):
        result = np.core.defchararray.replace(
            np.datetime_as_string(arr, unit='ms'), 'T', ' ')
        return result
    result = df.copy(deep=True)
    result.iloc[:, 0] = _to_timestamp_string(result.iloc[:, 0].values)
    if file_type == 'annotation':
        result.iloc[:, 1:2] = _to_timestamp_string(result.iloc[:, 1:2].values)
    return result


def convert_datetime_columns_to_datetime64ms(df, file_type):
    """Convert elements in the timestamp columns of the input mhealth dataframe to `datetime64[ms]` type.

    Args:
        df (pandas.DataFrame): The input mhealth dataframe. Timestamp columns should always be in `np.datetime64` type.
        file_type (str): mhealth file type. Now only support `sensor` and `annotation`.

    Returns:
        result (pandas.DataFrame): Dataframe with timestamp columns converted.
    """
    def _to_datetime64ms(arr):
        result = arr.astype('datetime64[ms]')
        return result
    result = df
    result.iloc[:, 0] = _to_datetime64ms(result.iloc[:, 0].values)
    if file_type == 'annotation':
        result.iloc[:, 1:2] = _to_datetime64ms(result.iloc[:, 1:2].values)
    return result


def convert_string_columns_to_datetime64ms(df, file_type):
    """Convert elements in the timestamp columns of the input dataframe to `datetime64[ms]` type.

    Args:
        df (pandas.DataFrame): The input mhealth dataframe. Timestamp columns are still human readable strings and need to be converted
        file_type (str): mhealth file type. Now only support `actigraph`.

    Returns:
        result (pandas.DataFrame): Dataframe with timestamp columns converted.
    """
    def _to_datetime64ms(arr, file_type):
        if file_type == 'sensor' or file_type == 'annotation':
            dt_format = '%Y-%m-%d %H:%M:%S.%f'
        elif file_type == 'actigraph':
            dt_format = '%m/%d/%Y %H:%M:%S.%f'
        vfun = partial(
            pd.to_datetime, format=dt_format, box=False)
        new_arr = vfun(arr)
        result = new_arr.astype('datetime64[ms]')
        return result

    result = df
    result.iloc[:, 0] = _to_datetime64ms(
        result.iloc[:, 0].values, file_type=file_type)
    if file_type == 'annotation':
        result.iloc[:, 1:2] = _to_datetime64ms(
            result.iloc[:, 1:2].values, file_type)
    return result


def rename_columns(df, file_type):
    """Rename column names of input dataframe to meet mhealth specification requirement

    Args:
        df (pandas.DataFrame): The input mhealth dataframe.
        file_type (str): mhealth file type. Now support only `sensor` and `annotation`.

    Returns:
        result (pandas.DataFrame): The mhealth dataframe with columns renamed.
    """
    result = df
    result = result.rename(columns={result.columns[0]: 'HEADER_TIME_STAMP'})
    if file_type == 'annotation':
        result = result.rename(
            columns={result.columns[1]: 'START_TIME', result.columns[2]: 'STOP_TIME'})
    return result


def append_times_as_index(df, st, et):
    df.insert(0, 'HEADER_TIME_STAMP', st)
    df.insert(1, 'START_TIME', st)
    df.insert(2, 'STOP_TIME', et)
    df = df.set_index(['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME'])
    return df


# def offset(df, offset_in_secs, start_time_col=0, stop_time_col=None):
#     df_copy = df.copy(deep=True)
#     if start_time_col is not None:
#         start_time_col = df_copy.columns[start_time_col]
#         df_copy[start_time_col] = df_copy[start_time_col] +
#         pd.Timedelta(offset_in_secs, unit='s')
#     if stop_time_col is not None:
#         stop_time_col = df_copy.columns[stop_time_col]
#         df_copy[stop_time_col] = df_copy[stop_time_col] +
#         pd.Timedelta(offset_in_secs, unit='s')
#     return df_copy


def segment(df, start_time=None, stop_time=None, start_time_col=0,
            stop_time_col=None):
    if stop_time_col is None:
        stop_time_col = start_time_col
    if start_time is None:
        start_time = df.iloc[0, start_time_col]
    if stop_time is None:
        stop_time = df.iloc[-1, stop_time_col]

    if start_time_col == stop_time_col:
        mask = (df.iloc[:, start_time_col] >= start_time) & (
            df.iloc[:, stop_time_col] < stop_time)
        return df[mask].copy(deep=True)
    else:
        mask = (df.iloc[:, start_time_col] <= stop_time) & (
            df.iloc[:, stop_time_col] >= start_time)
        subset_df = df[mask].copy(deep=True)

        start_time_col = df.columns[start_time_col]
        stop_time_col = df.columns[stop_time_col]

        subset_df.loc[subset_df.loc[:, start_time_col] <
                      start_time, start_time_col] = start_time
        subset_df.loc[subset_df.loc[:, stop_time_col] >
                      stop_time, stop_time_col] = stop_time
        return subset_df


def segment_sensor(df, start_time=None, stop_time=None):
    return segment(df, start_time=start_time, stop_time=stop_time)


def segment_annotation(df, start_time=None, stop_time=None):
    return segment(df, start_time=start_time, stop_time=stop_time,
                   start_time_col=1, stop_time_col=2)


def get_start_time(df, start_time_col=0):
    return df.iloc[0, start_time_col]


def get_end_time(df, stop_time_col=0):
    return df.iloc[-1, stop_time_col]


def get_annotation_labels(df):
    labels = df.iloc[:, 3].unique()
    return np.sort(labels)


def append_edge_data(df, before_df=None, after_df=None, duration=120,
                     start_time_col=0, stop_time_col=0):
    lbound_time = df.iloc[0, start_time_col]
    rbound_time = df.iloc[-1, stop_time_col]

    if before_df is not None:
        ledge_df = segment(before_df,
                           start_time=lbound_time -
                           pd.Timedelta(duration, unit='s'),
                           stop_time=lbound_time,
                           start_time_col=start_time_col,
                           stop_time_col=stop_time_col)
    else:
        ledge_df = pd.DataFrame()

    if after_df is not None:
        redge_df = segment(after_df,
                           start_time=rbound_time,
                           stop_time=rbound_time +
                           pd.Timedelta(duration, unit='s'),
                           start_time_col=start_time_col,
                           stop_time_col=stop_time_col)
    else:
        redge_df = pd.DataFrame()

    return pd.concat((ledge_df, df, redge_df))
