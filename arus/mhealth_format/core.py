import glob
import os
from . import constants
from . import helper
import pandas as pd
import datetime as dt


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
    mappings = pd.read_csv(filepath, header=0)
    return mappings


def get_subjects_info(dataset_path):
    filepath = os.path.join(
        dataset_path, constants.META_FOLDER, constants.META_SUBJECTS_FILENAME)
    subjects = pd.read_csv(filepath, header=0)
    return subjects


def get_class_category(dataset_path):
    filepath = os.path.join(
        dataset_path, constants.META_FOLDER, 'muss_class_labels.csv')
    class_category = pd.read_csv(filepath, header=0)
    return class_category


def get_pids(dataset_path):
    return list(filter(lambda name: name not in [
        constants.PROCESSED_FOLDER, constants.META_FOLDER], os.listdir(dataset_path)))


def get_sensor_files(pid, dataset_path, sid=None):
    if sid is None:
        files = glob.glob(os.path.join(dataset_path, pid,
                                       constants.MASTER_FOLDER, '**', '*.sensor.csv*'), recursive=True)
    else:
        files = glob.glob(os.path.join(dataset_path, pid,
                                       constants.MASTER_FOLDER, '**', '*{}*.sensor.csv*'.format(sid)), recursive=True)
    return sorted(files)


def get_annotation_files(pid, dataset_path, annotation_type=None, annotator=None):
    annotation_type = annotation_type or ""
    annotator = annotator or ""
    files = glob.glob(os.path.join(dataset_path, pid,
                                   constants.MASTER_FOLDER, '**', '*{}*{}*.annotation.csv*'.format(annotation_type, annotator)), recursive=True)
    return sorted(files)


def get_session_start_time(pid, dataset_path):
    smallest = dt.datetime.now()
    filepaths = get_sensor_files(
        pid, dataset_path) + get_annotation_files(pid, dataset_path)
    for path in filepaths:
        timestamp = helper.parse_timestamp_from_filepath(path, ignore_tz=True)
        if timestamp < smallest:
            smallest = timestamp
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
    dataset_dict['meta']['location_mapping'] = location_mappings
    dataset_dict['meta']['subjects'] = subjects
    dataset_dict['meta']['class_category'] = class_category
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
