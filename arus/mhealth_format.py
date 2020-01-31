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


def get_location_mappings(dataset_path):
    filepath = os.path.join(
        dataset_path, 'MetaCrossParticipants', 'location_mapping.csv')
    mappings = pd.read_csv(filepath, header=0)
    return mappings


def parse_placement_str(placement_str):
    result = ''
    placement_str = placement_str.lower()
    if 'nondominant' in placement_str or 'non-dominant' in placement_str or 'non dominant' in placement_str or placement_str.startswith('nd'):
        result = 'ND'
    elif 'dominant' in placement_str or placement_str.startswith('d'):
        result = 'D'

    if 'ankle' in placement_str or placement_str.endswith('a'):
        result += 'A'
    elif 'wrist' in placement_str or placement_str.endswith('w'):
        result += 'W'
    elif 'waist' in placement_str or 'hip' in placement_str or placement_str.endswith('h'):
        result += 'H'
    elif 'thigh' in placement_str or placement_str.endswith('t'):
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

    dataset_dict = {}
    ids = get_pids(dataset_path)
    location_mappings = get_location_mappings(dataset_path)
    for pid in ids:
        dataset_dict[pid] = {'sensors': {}, 'annotations': {}}
        sids = _get_sids_from_location_mappings(pid, location_mappings)
        placements = _get_placements_from_location_mappings(
            pid, location_mappings)
        placements = _parse_placements(placements)
        for sid, p in zip(sids, placements):
            dataset_dict[pid]['sensors'][p] = get_sensor_files(
                pid, dataset_path, sid=sid)
        annotation_files = get_annotation_files(pid, dataset_path)
        annotation_types = _parse_annotation_types(annotation_files)
        for annotation_type in annotation_types:
            dataset_dict[pid]['annotations'][annotation_type] = get_annotation_files(
                pid, dataset_path, annotation_type=annotation_type)
    return dataset_dict


if __name__ == "__main__":
    from arus import developer
    dataset_dict = traverse_dataset('D:/data/muss_data')
    developer.print_dict(dataset_dict)
