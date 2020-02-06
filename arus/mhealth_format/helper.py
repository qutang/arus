from . import constants
import datetime as dt
import os


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


def transform_class_category(input_label, class_category, input_category,  output_category):
    cond = class_category[input_category] == input_label
    return class_category.loc[cond, output_category].values[0]
