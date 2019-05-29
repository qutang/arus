import re
import os
import datetime
import numpy as np
import pandas as pd
from glob import glob

CAMELCASE_PATTERN = r'(?:[A-Z][A-Za-z0-9]+)+'
VERSIONCODE_PATTERN = r'(?:NA|[0-9x]+)'
SID_PATTERN = r'[A-Z0-9]+'
ANNOTATOR_PATTERN = r'[A-Za-z0-9]+'
FILE_TIMESTAMP_PATTERN = r'[0-9]{4}(?:\-[0-9]{2}){5}-[0-9]{3}-(?:P|M)[0-9]{4}'
FILE_EXTENSION_PATTERN = r'''
(?:sensor|event|log|annotation|feature|class|prediction|model|classmap)\.csv
'''
MHEALTH_FLAT_FILEPATH_PATTERN = r'(\w+)[\/\\]{1}(?:(?:MasterSynced[\/\\]{1})|(?:Derived[\/\\]{1}(?:\w+[\/\\]{1})*))[0-9A-Za-z\-\.]+\.csv(\.gz)*'
MHEALTH_FILEPATH_PATTERN = r'(\w+)[\/\\]{1}(?:(?:MasterSynced[\/\\]{1})|(?:Derived[\/\\]{1}(?:\w+[\/\\]{1})*))\d{4}[\/\\]{1}\d{2}[\/\\]{1}\d{2}[\/\\]{1}\d{2}'
MHEALTH_FILE_TIMESTAMP_FORMAT = '%Y-%m-%d-%H-%M-%S-%f'
MHEALTH_FILE_TIMESTAMP_WITH_TZ_FORMAT = '%Y-%m-%d-%H-%M-%S-%f-%z'

def is_mhealth_filepath(filepath):
    """Validate if input file path is in mhealth format
    
    Args:
        filepath (str): input file path
    
    Returns:
        is_mhealth (bool): `True` if the input is in mhealth format
    """
    filepath = os.path.abspath(filepath)
    matched = re.search(
        MHEALTH_FILEPATH_PATTERN,
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
        MHEALTH_FLAT_FILEPATH_PATTERN,
        os.path.abspath(filepath)
    )
    return matched is not None


def extract_mhealth_rootpath(filepath):
    """Extract the root folder path for the whole dataset from the input mhealth file path
    
    Args:
        filepath (str): The input mhealth file path
    
    Returns:
        rootpath (str): The root folder path of the whole dataset
    """
    assert is_mhealth_filepath(filepath) or is_mhealth_flat_filepath(filepath)
    pid = extract_pid(filepath)
    return filepath.split(pid)[0]


def extract_pid_rootpath(filepath):
    """Extract the pid folder path for the participant from the input mhealth file path
    
    Args:
        filepath (str): The input mhealth file path
    
    Returns:
        pidpath (str): The pid folder path of the participant's data
    """
    assert is_mhealth_filepath(filepath) or is_mhealth_flat_filepath(filepath)
    pid = extract_pid(filepath)
    return os.path.join(filepath.split(pid)[0], pid)


def extract_location_mapping_filepath(filepath):
    """Extract the file path of the `location_mapping` meta file

    If `location_mapping` files are stored separately for each participant, the one that matches the pid of the given input file path will be returned.
    
    Args:
        filepath (str): The input mhealth file path
    
    Returns:
        location_mapping_file (str): The `location_mapping` file path for the entire dataset. If no file is found, return `None`.
    """
    mhealth_root = extract_mhealth_rootpath(filepath)
    result = os.path.join(
        mhealth_root, 'DerivedCrossParticipants', 'location_mapping.csv')
    if os.path.exists(result):
        return result
    else:
        pid_root = extract_pid_rootpath(filepath)
        candicates = glob(os.path.join(
            pid_root, '**', 'location_mapping.csv'), recursive=True)
        if len(candicates) == 1:
            return candicates[0]
        else:
            return None


def extract_offset_mapping_filepath(filepath):
    """Extract the file path of the `offset_mapping` meta file

    If `offset_mapping` files are stored separately for each participant, the one that matches the pid of the given input file path will be returned.
    
    Args:
        filepath (str): The input mhealth file path
    
    Returns:
        offset_mapping_file (str): The `offset_mapping` file path for the entire dataset. If no file is found, return `None`.
    """
    pid_root = extract_pid_rootpath(filepath)
    candicates = glob(os.path.join(
        pid_root, '**', 'offset_mapping.csv'), recursive=True)
    if len(candicates) == 1:
        return candicates[0]
    else:
        mhealth_root = extract_mhealth_rootpath(filepath)
        result = os.path.join(
            mhealth_root, 'DerivedCrossParticipants', 'offset_mapping.csv')
        if os.path.exists(result):
            return result
        else:
            return None

def extract_pid_exceptions_filepath(filepath):
    """Extract the file path of the `pid_exceptions` meta file

    If `pid_exceptions` files are stored separately for each participant, the one that matches the pid of the given input file path will be returned.
    
    Args:
        filepath (str): The input mhealth file path
    
    Returns:
        pid_exceptions_file (str): The `pid_exceptions` file path for the entire dataset. If no file is found, return `None`.
    """
    mhealth_root = extract_mhealth_rootpath(filepath)
    result = os.path.join(
        mhealth_root, 'DerivedCrossParticipants', 'pid_exceptions.csv')
    if os.path.exists(result):
        return result
    else:
        return None


def is_pid_included(filepath):
    exceptions = pd.read_csv(extract_pid_exceptions_filepath(filepath))
    pid = extract_pid(filepath)
    if np.any(pid == exceptions['PID'].values):
        return False
    else:
        return True


def extract_orientation_corrections_filepath(filepath):
    pid_root = extract_pid_rootpath(filepath)
    candicates = glob(os.path.join(
        pid_root, '**', 'orientation_corrections.csv'), recursive=True)
    if len(candicates) == 1:
        return candicates[0]
    else:
        mhealth_root = extract_mhealth_rootpath(filepath)
        result = os.path.join(
            mhealth_root, 'DerivedCrossParticipants',
            'orientation_corrections.csv')
        if os.path.exists(result):
            return result
        else:
            return False


def is_mhealth_filename(filepath):
    filename = os.path.basename(filepath)

    sensor_filename_pattern = '^' + CAMELCASE_PATTERN + '\-' + \
        CAMELCASE_PATTERN + \
        '\-' + VERSIONCODE_PATTERN + '\.' + \
        SID_PATTERN + '\-' + CAMELCASE_PATTERN + '\.' + \
        FILE_TIMESTAMP_PATTERN + '\.sensor\.csv(\.gz)*$'

    annotation_filename_pattern = '^' + CAMELCASE_PATTERN + '\.' + \
        ANNOTATOR_PATTERN + '\-' + CAMELCASE_PATTERN + '\.' + \
        FILE_TIMESTAMP_PATTERN + '\.annotation\.csv(\.gz)*$'

    sensor_matched = re.search(
        sensor_filename_pattern,
        filename
    )

    annotation_matched = re.search(
        annotation_filename_pattern,
        filename
    )
    return sensor_matched is not None or annotation_matched is not None


def extract_pid(filepath):
    if is_mhealth_filepath(filepath):
        matched = re.search(MHEALTH_FILEPATH_PATTERN, filepath)
    elif is_mhealth_flat_filepath(filepath):
        matched = re.search(MHEALTH_FLAT_FILEPATH_PATTERN, filepath)
    else:
        return None
    return matched.group(1) if matched is not None else None


def get_pids_list(rootpath):
    return list(filter(lambda name: os.path.isdir(os.path.join(rootpath, name)), os.listdir(rootpath)))


def extract_sensor_type(filepath):
    assert is_mhealth_filename(filepath)
    filename = os.path.basename(filepath)
    result = filename.split('.')[0].split('-')[0]
    return result


def extract_data_type(filepath):
    assert is_mhealth_filename(filepath)
    filename = os.path.basename(filepath)
    tokens = filename.split('.')[0]
    tokens = tokens.split('-')
    if len(tokens) >= 2:
        return tokens[1]
    else:
        return None


def extract_version_code(filepath):
    assert is_mhealth_filename(filepath)
    filename = os.path.basename(filepath)
    tokens = filename.split('.')[0]
    tokens = tokens.split('-')
    if len(tokens) >= 3:
        return tokens[2]
    else:
        return None


def extract_sid(filepath):
    assert is_mhealth_filename(filepath)
    filename = os.path.basename(filepath)
    return filename.split('.')[1].split('-')[0]


def extract_file_type(filepath):
    assert is_mhealth_filename(filepath)
    filename = os.path.basename(filepath)
    if filename.endswith('gz'):
        return filename.split('.')[-3]
    else:
        return filename.split('.')[-2]


def extract_file_timestamp(filepath, ignore_tz=True):
    assert is_mhealth_filename(filepath)
    filename = os.path.basename(filepath)
    if filename.endswith('gz'):
        timestamp_index = -4
    else:
        timestamp_index = -3
    timestamp_str = filename.split('.')[timestamp_index]
    if ignore_tz:
        timestamp_str = timestamp_str[:-6]
        result = datetime.datetime.strptime(
            timestamp_str, '%Y-%m-%d-%H-%M-%S-%f')
    else:
        timestamp_str = timestamp_str.replace('P', '+').replace('M', '-')
        result = datetime.datetime.strptime(
            timestamp_str, '%Y-%m-%d-%H-%M-%S-%f-%z')
    return result


def extract_session_start_time(filepath, filepaths):
    pid = extract_pid(filepath)
    smallest = datetime.datetime.now()
    for path in filepaths:
        if extract_pid(path) == pid:
            timestamp = extract_file_timestamp(path, ignore_tz=True)
            if timestamp < smallest:
                smallest = timestamp
    smallest = smallest.replace(microsecond=0, second=0, minute=0)
    return smallest


def extract_session_end_time(filepath, filepaths):
    pid = extract_pid(filepath)
    largest = datetime.datetime.fromtimestamp(100000)
    for path in filepaths:
        if extract_pid(path) == pid:
            timestamp = extract_file_timestamp(path, ignore_tz=True)
            if timestamp > largest:
                largest = timestamp
    if largest.minute != 0 or largest.second != 0 or largest.microsecond != 0:
        largest = largest.replace(microsecond=0, second=0,
                                  minute=0) + datetime.timedelta(hours=1)
    return largest


def extract_timezone(filepath):
    """Extract time zone object from the input mhealth file path
    
    Args:
        filepath (str): File path in mhealth format
    
    Returns:
        tz (tzinfo): The extracted time zone object
    """
    dt = extract_file_timestamp(filepath, ignore_tz=False)
    return dt.tzinfo


def extract_timezone_name(filepath):
    """Extract time zone name from the input mhealth file path
    
    Args:
        filepath (str): File path in mhealth format
    
    Returns:
        tz_name (str): Time zone name as formatted with `%Z`
    """
    dt = extract_file_timestamp(filepath, ignore_tz=False)
    tz_name = dt.strftime('%Z')
    return tz_name
