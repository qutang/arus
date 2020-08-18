import glob
import os
from . import constants
from . import helper
import pandas as pd
import datetime as dt
from loguru import logger


def get_processed_path(dataset_path):
    return os.path.join(
        dataset_path, constants.PROCESSED_FOLDER)


def get_processed_files(dataset_path):
    processed_files = glob.glob(os.path.join(
        get_processed_path(dataset_path), '*.csv'))
    results = {}
    for f in processed_files:
        results[os.path.basename(f).split('.')[0]] = f
    return results


def get_location_mappings(dataset_path):
    filepath = os.path.join(
        dataset_path, constants.META_FOLDER, constants.META_LOCATION_MAPPING_FILENAME)
    if not os.path.exists(filepath):
        return None
    mappings = pd.read_csv(filepath, header=0)
    return mappings


def get_sensor_placement(dataset_path, pid, sid):
    mapping = get_location_mappings(dataset_path)
    condition = (mapping.iloc[:, 0] == pid) & (
        mapping.iloc[:, 1] == sid)
    placement = mapping.loc[condition, :].iloc[:, 2].values[0]
    return placement.lower()


def get_subjects_info(dataset_path):
    filepath = os.path.join(
        dataset_path, constants.META_FOLDER, constants.META_SUBJECTS_FILENAME)
    if not os.path.exists(filepath):
        return None
    subjects = pd.read_csv(filepath, header=0)
    return subjects


def get_class_category(dataset_path):
    filepath = os.path.join(
        dataset_path, constants.META_FOLDER, 'muss_class_labels.csv')
    if not os.path.exists(filepath):
        return None
    class_category = pd.read_csv(filepath, header=0)
    return class_category


def get_offset_mappings(dataset_path):
    filepath = os.path.join(
        dataset_path, constants.META_FOLDER, 'offset_mappings.csv')
    if not os.path.exists(filepath):
        return None
    offset_mappings = pd.read_csv(filepath, header=0)
    return offset_mappings


def get_orientation_corrections(dataset_path):
    filepath = os.path.join(
        dataset_path, constants.META_FOLDER, 'orientation_corrections.csv')
    if not os.path.exists(filepath):
        return None
    orientation_corrections = pd.read_csv(filepath, header=0)
    return orientation_corrections


def get_pids(dataset_path):
    pids = filter(lambda name: name not in [
        constants.PROCESSED_FOLDER, constants.META_FOLDER, '.git', '.gitignore'], os.listdir(dataset_path))
    return list(sorted(pids))


def get_subject_log(dataset_path, pid, log_filename=None):
    if log_filename is None:
        return os.path.join(dataset_path, pid, constants.SUBJECT_LOG_FOLDER)
    else:
        return os.path.join(dataset_path, pid, constants.SUBJECT_LOG_FOLDER, log_filename)


def get_subject_meta(dataset_path, pid, meta_filename=None):
    if meta_filename is None:
        return os.path.join(dataset_path, pid, constants.SUBJECT_META_FOLDER)
    else:
        return os.path.join(dataset_path, pid, constants.SUBJECT_META_FOLDER, meta_filename)


def get_sensor_files(pid, dataset_path, sid="", given_date=None, data_type=""):
    if given_date is not None:
        date_hour_str = given_date.strftime(f'%Y-%m-%d{os.sep}%H')
    name_pattern = f'*{data_type}*{sid}*.sensor.csv*'
    path_pattern = '**' if given_date is None else f'{date_hour_str}'
    files = glob.glob(os.path.join(dataset_path, pid,
                                   constants.MASTER_FOLDER, path_pattern, name_pattern), recursive=True)
    return sorted(files)


def get_annotation_files(pid, dataset_path, annotation_type=None, annotator=None):
    annotation_type = annotation_type or ""
    annotator = annotator or ""
    files = glob.glob(os.path.join(dataset_path, pid,
                                   constants.MASTER_FOLDER, '**', '*{}*{}*.annotation.csv*'.format(annotation_type, annotator)), recursive=True)
    return sorted(files)


def get_date_folders(pid, dataset_path):
    folder_names = os.listdir(os.path.join(
        dataset_path, pid, constants.MASTER_FOLDER))
    if '-' in folder_names[0]:
        sep = '-'
    else:
        sep = os.sep
    if sep == os.sep:
        date_folders = glob.glob(os.path.join(
            dataset_path, pid, constants.MASTER_FOLDER, '*', '*', '*', '*'), recursive=True)
    elif sep == '-':
        date_folders = glob.glob(os.path.join(
            dataset_path, pid, constants.MASTER_FOLDER, '*', '*'), recursive=True)
    date_folders = list(filter(lambda path: os.path.isdir(path), date_folders))
    return date_folders


def get_session_span(pid, dataset_path):
    date_folders = get_date_folders(pid, dataset_path)
    folders_as_ts = list(
        map(helper.parse_date_from_filepath, date_folders))
    folders_as_ts = sorted(folders_as_ts)
    return folders_as_ts[0], folders_as_ts[-1] + dt.timedelta(hours=1)


def get_session_start_time(pid, dataset_path, round_to='hour'):
    smallest = dt.datetime.now()
    filepaths = get_sensor_files(
        pid, dataset_path) + get_annotation_files(pid, dataset_path)
    for path in filepaths:
        timestamp = helper.parse_timestamp_from_filepath(path, ignore_tz=True)
        if timestamp < smallest:
            smallest = timestamp
    if round_to == 'minute':
        smallest = smallest.replace(microsecond=0, second=0)
    elif round_to == 'second':
        smallest = smallest.replace(microsecond=0)
    else:
        smallest = smallest.replace(microsecond=0, second=0, minute=0)
    return smallest


def traverse_dataset(dataset_path):
    """Traverse the given raw mhealth dataset to parse meta info, sensor and annotation files for each participant.

    Args:
        dataset_path (str): The filepath to the raw dataset.

    Returns:
        dict: python dict object storing the traversed paths and meta info for the given dataset.

        {
            "meta": {
                "location_mapping": pandas.DataFrame, location_mappings,
                "name": str, dataset name,
                "root": str, dataset_path,
                "subjects": pandas.DataFrame, subjects' geographic info
            },
            "processed": {
                "muss": "muss.csv",
                ...
            },
            "subjects": {
                "pid_1": {
                    "sensors": {
                        "DW": [
                            "/path/to/file1.sensor.csv",
                            ...
                        ],
                        ...
                    },
                    "annotations": {
                        "annotation_type": [
                            "/path/to/file1.annotation.csv",
                            ...
                        ],
                        ...
                    }
                }
            }
        }
    """
    logger.warning(
        'This method has been deprecated. And will be removed in version 1.2.0.')

    def _parse_placements(placements):
        return [helper.parse_placement_from_str(p) for p in placements]

    def _get_placements_from_location_mappings(pid, location_mappings):
        filter_pid = location_mappings[constants.FEATURE_SET_PID_COL] == pid
        return location_mappings.loc[filter_pid, 'SENSOR_PLACEMENT'].values.tolist()

    def _get_sids_from_location_mappings(pid, location_mappings):
        filter_pid = location_mappings[constants.FEATURE_SET_PID_COL] == pid
        return location_mappings.loc[filter_pid, 'SENSOR_ID'].values.tolist()

    def _parse_annotation_types(annotation_files):
        return list(set([helper.parse_annotation_type_from_filepath(filepath)
                         for filepath in annotation_files]))

    dataset_dict = {'meta': {}, 'subjects': {}}
    ids = get_pids(dataset_path)

    # Get meta files
    location_mappings = get_location_mappings(
        dataset_path)
    subjects = get_subjects_info(dataset_path)
    class_category = get_class_category(dataset_path)
    offset_mappings = get_offset_mappings(dataset_path)
    orientation_corrections = get_orientation_corrections(dataset_path)
    dataset_dict['meta']['location_mapping'] = location_mappings
    dataset_dict['meta']['subjects'] = subjects
    dataset_dict['meta']['class_category'] = class_category
    dataset_dict['meta']['offset_mapping'] = offset_mappings
    dataset_dict['meta']['orientation_corrections'] = orientation_corrections
    dataset_dict['meta']['root'] = dataset_path
    dataset_dict['meta']['name'] = os.path.basename(dataset_path)

    # Get processed files
    dataset_dict['processed'] = get_processed_files(dataset_path)

    for pid in ids:
        dataset_dict['subjects'][pid] = {'sensors': {}, 'annotations': {}}

        # Get sensor files
        sids = _get_sids_from_location_mappings(pid, location_mappings)
        placements = _get_placements_from_location_mappings(
            pid, location_mappings)
        placements = _parse_placements(placements)

        for sid, p in zip(sids, placements):
            dataset_dict['subjects'][pid]['sensors'][p] = get_sensor_files(
                pid, dataset_path, sid=sid)

        # Get annotation files
        annotation_files = get_annotation_files(pid, dataset_path)
        annotation_types = _parse_annotation_types(annotation_files)
        for annotation_type in annotation_types:
            dataset_dict['subjects'][pid]['annotations'][annotation_type] = get_annotation_files(
                pid, dataset_path, annotation_type=annotation_type)
    return dataset_dict
