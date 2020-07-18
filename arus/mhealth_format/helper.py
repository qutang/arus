from . import constants
import datetime as dt
import pandas as pd
import os
import re
from .. import moment


class ParseError(Exception):
    pass


def parse_column_names_from_data_type(data_type):
    if data_type in ['AccelerometerCalibrated', 'IMUAccelerometerCalibrated', 'AccelerationCalibrated']:
        return ['ACCELEROMETER_X', 'ACCELEROMETER_Y', 'ACCELEROMETER_Z']
    elif data_type in ['IMUTenAxes']:
        return ["ACCELEROMETER_X", "ACCELEROMETER_Y", "ACCELEROMETER_Z", "TEMPERATURE", "GYROSCOPE_X", "GYROSCOPE_Y", "GYROSCOPE_Z", "MAGNETOMETER_X", "MAGNETOMETER_Y", "MAGNETOMETER_Z"]
    elif data_type in ['IMUGyroscope']:
        return ["GYROSCOPE_X", "GYROSCOPE_Y", "GYROSCOPE_Z"]
    elif data_type in ['IMUMagnetometer']:
        return ["MAGNETOMETER_X", "MAGNETOMETER_Y", "MAGNETOMETER_Z"]
    elif data_type in ['IMUTemperature']:
        return ['TEMPERATURE']
    else:
        raise NotImplementedError(
            f"The given data type {data_type} is not supported")


def parse_column_names_from_filepath(filepath):
    data_type = parse_data_type_from_filepath(filepath)
    return parse_column_names_from_data_type(data_type)


def parse_placement_from_str(placement_str):
    result = ''
    placement_str = placement_str.lower()
    if 'nondominant' in placement_str or 'non-dominant' in placement_str or 'non dominant' in placement_str or placement_str.startswith('nd'):
        result = 'ND'
    elif 'dominant' in placement_str or placement_str.startswith('d'):
        result = 'D'
    if 'ankle' in placement_str or placement_str.endswith('da'):
        result += 'A'
    elif 'wrist' in placement_str or placement_str.endswith('dw'):
        result += 'W'
    elif 'waist' in placement_str or 'hip' in placement_str or placement_str.endswith('dh'):
        result += 'H'
    elif 'thigh' in placement_str or placement_str.endswith('dt'):
        result += 'T'
    return result


def parse_date_from_filepath(filepath, ignore_tz=True):
    tokens = filepath.split(constants.MASTER_FOLDER)[1].split(os.sep)
    if '-' in tokens[1]:
        sep = '-'
    else:
        sep = os.sep
    if sep == os.sep:
        hour = tokens[4].split('-')[0]
        day = tokens[3]
        month = tokens[2]
        year = tokens[1]
    elif sep == '-':
        hour = tokens[2].split('-')[0]
        sub_tokens = tokens[1].split('-')
        day = sub_tokens[-1]
        month = sub_tokens[1]
        year = sub_tokens[0]
    file_date = dt.datetime(year=int(year), month=int(month), day=int(
        day), hour=int(hour), minute=0, second=0, microsecond=0)
    return file_date


def parse_timestamp_from_filepath_content(filepath, ts_col=0, ignore_tz=True):
    first_row = pd.read_csv(filepath, nrows=1, header=0, parse_dates=[
        ts_col], infer_datetime_format=True)
    return first_row.iloc[0, ts_col]


def parse_timestamp_from_filepath(filepath, ignore_tz=True):
    filename = os.path.basename(filepath)
    if filename.endswith('gz'):
        timestamp_index = -4
    else:
        timestamp_index = -3
    timestamp_str = filename.split('.')[timestamp_index]
    if ignore_tz:
        timestamp_str = timestamp_str[:-6]
        result = dt.datetime.strptime(
            timestamp_str, constants.FILE_TIMESTAMP_FORMAT)
    else:
        timestamp_str = timestamp_str.replace('P', '+').replace('M', '-')
        result = dt.datetime.strptime(
            timestamp_str, constants.FILE_TIMESTAMP_FORMAT_WITH_TZ)
    return result


def parse_annotation_type_from_filepath(filepath):
    return os.path.basename(filepath).split('.')[0]


def parse_annotator_from_filepath(filepath):
    return os.path.basename(filepath).split('.')[1]


def parse_sensor_type_from_filepath(filepath):
    result = os.path.basename(filepath).split('.')[0].split('-')[0]
    return result


def parse_data_type_from_filepath(filepath):
    return os.path.basename(filepath).split('.')[0].split('-')[1]


def parse_version_code_from_filepath(filepath):
    return os.path.basename(filepath).split('.')[0].split('-')[2]


def parse_sensor_id_from_filepath(filepath):
    return os.path.basename(filepath).split('.')[1].split('-')[0]


def parse_data_id_from_filepath(filepath):
    return os.path.basename(filepath).split('.')[1]


def parse_pid_from_filepath(filepath):
    try:
        assert is_mhealth_filepath(
            filepath) or is_mhealth_flat_filepath(filepath)
        pid = os.path.basename(
            os.path.dirname(
                filepath.split(constants.MASTER_FOLDER)[
                    0].split(constants.DERIVED_FOLDER)[0]
            )
        )
        return pid
    except Exception:
        raise ParseError('Fail to parse pid for the given filepath')


def parse_subject_path_from_filepath(filepath):
    try:
        assert is_mhealth_filepath(
            filepath) or is_mhealth_flat_filepath(filepath)
        subject_folder = os.path.dirname(
            filepath.split(constants.MASTER_FOLDER)[
                0].split(constants.DERIVED_FOLDER)[0]
        )
        return subject_folder
    except Exception:
        raise ParseError('Fail to parse pid for the given filepath')


def parse_filetype_from_filepath(filepath):
    filename = os.path.basename(filepath)
    if filename.endswith('gz'):
        return filename.split('.')[-3]
    else:
        return filename.split('.')[-2]


def parse_datetime_columns_from_filepath(filepath):
    """Utility to get the timestamp column indices given file type

    Args:
        filepath (str): mhealth file path.

    Returns:
        col_indices (list): list of column indices (0 based)
    """
    filetype = parse_filetype_from_filepath(filepath)
    if filetype == constants.SENSOR_FILE_TYPE:
        return [0]
    elif filetype in [constants.ANNOTATION_FILE_TYPE, constants.FEATURE_FILE_TYPE, constants.CLASS_FILE_TYPE, constants.FEATURE_SET_FILE_TYPE]:
        return [0, 1, 2]
    else:
        raise NotImplementedError(
            'The given file type {} is not supported'.format(filetype))


def format_columns(data, filetype):
    data = data.rename(columns={data.columns[0]: constants.TIMESTAMP_COL})
    if filetype == constants.ANNOTATION_FILE_TYPE:
        data.columns = constants.FEATURE_SET_TIMESTAMP_COLS + \
            [constants.ANNOTATION_LABEL_COL]
    elif filetype == constants.SENSOR_FILE_TYPE:
        # COLUMN names should be A-Z0-9_
        data.columns = list(
            map(lambda col: col.upper().replace(' ', '_'), data.columns))
    return data


def format_file_timestamp_from_data(data, filetype):
    if filetype == constants.SENSOR_FILE_TYPE:
        col = 0
    else:
        col = 1
    st = moment.Moment(data.iloc[0, col]).to_datetime(
        tz=moment.Moment.get_local_timezone())
    timestamp_str = st.strftime(constants.FILE_TIMESTAMP_FORMAT)[:-3]
    timestamp_str += '-' + \
        st.strftime('%z').replace('-', 'M').replace('+', 'P')
    return timestamp_str


def format_date_folder_path_from_data(data, filetype):
    if filetype == constants.SENSOR_FILE_TYPE:
        col = 0
    else:
        col = 1
    st = moment.Moment(data.iloc[0, col]).to_datetime(
        tz=moment.Moment.get_local_timezone())
    year = st.strftime('%Y')
    month = st.strftime('%m')
    day = st.strftime('%d')
    hour = st.strftime('%H')
    return year + '-' + month + '-' + day + os.sep + hour


def compare_two_mhealth_filepaths(filepath1, filepath2):
    sections1 = [os.path.dirname(filepath1)]
    sections2 = [os.path.dirname(filepath2)]
    name1 = os.path.basename(filepath1).strip('.gz')
    name2 = os.path.basename(filepath2).strip('.gz')
    sections1 += name1.split('.')
    sections2 += name2.split('.')
    if len(sections1) != len(sections2):
        return False
    for section1, section2 in zip(sections1, sections2):
        if section1 != section2 and sections1.index(section1) != 3:
            return False
    return True


def transform_class_category(input_label, class_category, input_category,  output_category):
    cond = class_category[input_category] == input_label
    return class_category.loc[cond, output_category].values[0]


def is_mhealth_filepath(filepath):
    """Validate if input file path is in mhealth format

    Args:
        filepath (str): input file path

    Returns:
        is_mhealth (bool): `True` if the input is in mhealth format
    """
    filepath = os.path.abspath(filepath)
    matched = re.search(
        constants.MHEALTH_FILEPATH_PATTERN,
        filepath)
    return matched is not None


def is_mhealth_flat_filepath(filepath):
    """Validate if input file path is in mhealth format (flat structure)

    The flat structure stores all files directly in the `MasterSynced` folder in the pid folder, ignoring all date and hour folders.

    Args:
        filepath (str): input file path

    Returns:
        is_mhealth (bool): `True` if the input is in mhealth flat format
    """
    matched = re.search(
        constants.MHEALTH_FLAT_FILEPATH_PATTERN,
        os.path.abspath(filepath)
    )
    return matched is not None


def is_mhealth_filename(filepath):
    filename = os.path.basename(filepath)

    sensor_filename_pattern = r"^{}\-{}\-{}\.{}\-{}\.{}\.sensor\.csv(\.gz)*$".format(
        constants.CAMELCASE_PATTERN, constants.CAMELCASE_PATTERN,
        constants.VERSIONCODE_PATTERN, constants.SID_PATTERN, constants.CAMELCASE_PATTERN, constants.FILE_TIMESTAMP_PATTERN
    )

    annotation_filename_pattern = r"^{}\.{}\-{}\.{}\.annotation\.csv(\.gz)*$".format(
        constants.CAMELCASE_PATTERN, constants.ANNOTATOR_PATTERN, constants.CAMELCASE_PATTERN, constants.FILE_TIMESTAMP_PATTERN)

    sensor_matched = re.search(
        sensor_filename_pattern,
        filename
    )

    annotation_matched = re.search(
        annotation_filename_pattern,
        filename
    )
    return sensor_matched is not None or annotation_matched is not None
