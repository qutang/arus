"""
Functions to manipulate dataset stored in mhealth format.

Author: Qu Tang
Date: 01/31/2020
License: GNU v3
"""

import glob
import os
import pandas as pd
import datetime as dt

SENSOR_PLACEMENTS = ['DW', 'NDW', 'DA', 'NDA', 'DT', 'NDT', 'DH', 'NDH']


def get_location_mappings(dataset_path):
    filepath = os.path.join(
        dataset_path, 'MetaCrossParticipants', 'location_mapping.csv')
    mappings = pd.read_csv(filepath, header=0)
    return mappings


def get_subjects_info(dataset_path):
    filepath = os.path.join(
        dataset_path, 'MetaCrossParticipants', 'subjects.csv')
    subjects = pd.read_csv(filepath, header=0)
    return subjects


def get_class_category(dataset_path):
    filepath = os.path.join(
        dataset_path, 'MetaCrossParticipants', 'muss_class_labels.csv')
    class_category = pd.read_csv(filepath, header=0)
    return class_category


def transform_class_category(class_category, input_category, input_label, output_category):
    cond = class_category[input_category] == input_label
    return class_category.loc[cond, output_category].values[0]


def get_processed_path(dataset_path):
    return os.path.join(
        dataset_path, 'DerivedCrossParticipants')


def get_processed_files(dataset_path):
    processed_files = glob.glob(os.path.join(
        get_processed_path(dataset_path), '*.csv'))
    results = {}
    for f in processed_files:
        results[os.path.basename(f).split('.')[0]] = f
    return results


def parse_placement_str(placement_str):
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


def parse_file_timestamp(filepath, ignore_tz=True):
    filename = os.path.basename(filepath)
    if filename.endswith('gz'):
        timestamp_index = -4
    else:
        timestamp_index = -3
    timestamp_str = filename.split('.')[timestamp_index]
    if ignore_tz:
        timestamp_str = timestamp_str[:-6]
        result = dt.datetime.strptime(
            timestamp_str, '%Y-%m-%d-%H-%M-%S-%f')
    else:
        timestamp_str = timestamp_str.replace('P', '+').replace('M', '-')
        result = dt.datetime.strptime(
            timestamp_str, '%Y-%m-%d-%H-%M-%S-%f-%z')
    return result


def get_annotation_type(filename):
    return filename.split('.')[0]


def get_pids(dataset_path):
    return list(filter(lambda name: name not in [
        'DerivedCrossParticipants', 'MetaCrossParticipants'], os.listdir(dataset_path)))


def get_sensor_files(pid, dataset_path, sid=None):
    if sid is None:
        files = glob.glob(os.path.join(dataset_path, pid,
                                       'MasterSynced', '**', '*.sensor.csv*'), recursive=True)
    else:
        files = glob.glob(os.path.join(dataset_path, pid,
                                       'MasterSynced', '**', '*{}*.sensor.csv*'.format(sid)), recursive=True)
    return sorted(files)


def get_annotation_files(pid, dataset_path, annotation_type=None, annotator=None):
    annotation_type = annotation_type or ""
    annotator = annotator or ""
    files = glob.glob(os.path.join(dataset_path, pid,
                                   'MasterSynced', '**', '*{}*{}*.annotation.csv*'.format(annotation_type, annotator)), recursive=True)
    return sorted(files)


def get_session_start_time(pid, dataset_path):
    smallest = dt.datetime.now()
    filepaths = get_sensor_files(
        pid, dataset_path) + get_annotation_files(pid, dataset_path)
    for path in filepaths:
        timestamp = parse_file_timestamp(path, ignore_tz=True)
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
        return [parse_placement_str(p) for p in placements]

    def _get_placements_from_location_mappings(pid, location_mappings):
        filter_pid = location_mappings['PID'] == pid
        return location_mappings.loc[filter_pid, 'SENSOR_PLACEMENT'].values.tolist()

    def _get_sids_from_location_mappings(pid, location_mappings):
        filter_pid = location_mappings['PID'] == pid
        return location_mappings.loc[filter_pid, 'SENSOR_ID'].values.tolist()

    def _parse_annotation_types(annotation_files):
        return list(set([get_annotation_type(os.path.basename(filepath))
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


if __name__ == "__main__":
    from arus import developer
    from arus import env
    dataset_dict = traverse_dataset(
        os.path.join(env.get_data_home(), 'spades_lab'))
    developer.logging_dict(dataset_dict)
