from . import constants
import datetime as dt
import os
from .. import moment


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


def parse_pid_from_filepath(filepath):
    return os.path.basename(os.path.dirname(filepath.split(constants.MASTER_FOLDER)))


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
    return data


def format_file_timestamp_from_data(data, filetype):
    if filetype == constants.SENSOR_FILE_TYPE:
        col = 0
    else:
        col = 1
    st = moment.Moment(data.iloc[0, col]).to_datetime(
        tz=moment.Moment.get_local_timezone())
    timestamp_str = st.strftime(constants.FILE_TIMESTAMP_FORMAT_WITH_TZ)
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
    return year + os.sep + month + os.sep + day + os.sep + hour


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