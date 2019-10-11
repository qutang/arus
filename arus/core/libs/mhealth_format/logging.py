import numpy as np


def display_start_and_stop_time(df, file_type):
    if file_type == 'sensor':
        st = df['HEADER_TIME_STAMP'].values[0]
        et = df['HEADER_TIME_STAMP'].values[-1]
    else:
        raise NotImplementedError('Given file type is not implemented')
    st_str = np.datetime_as_string(
        st, unit='ms')
    et_str = np.datetime_as_string(
        et, unit='ms')
    return "{} - {}".format(st_str, et_str)
