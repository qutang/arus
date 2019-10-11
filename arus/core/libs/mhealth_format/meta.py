from ..mhealth_format import path
from ..mhealth_format import io
import numpy as np
import pandas as pd


def get_offset(filepath, offset_column):
    offset_mapping_file = path.extract_offset_mapping_filepath(filepath)
    pid = path.extract_pid(filepath)
    if bool(offset_mapping_file):
        offset_mapping = io.load_offset_mapping(offset_mapping_file)
        offset_in_secs = float(
            offset_mapping.loc[offset_mapping.iloc[:, 0] == pid,
                               offset_mapping.columns[offset_column]].values[0]
        )
    else:
        offset_in_secs = 0
    return offset_in_secs


def get_orientation_correction(filepath):
    orientation_corrections_file = path.extract_orientation_corrections_filepath(
        filepath)
    pid = path.extract_pid(filepath)
    sid = path.extract_sid(filepath)
    if bool(orientation_corrections_file):
        orientation_corrections = io.load_orientation_corrections(
            orientation_corrections_file)
        orientation_correction = orientation_corrections.loc[
            (orientation_corrections.iloc[:, 0] == pid) & (
                orientation_corrections.iloc[:, 1] == sid),
            orientation_corrections.columns[3:6]
        ]
        if orientation_correction.empty:
            orientation_correction = np.array(['x', 'y', 'z'])
        else:
            orientation_correction = orientation_correction.values[0]
    else:
        orientation_correction = np.array(['x', 'y', 'z'])
    return orientation_correction

def get_init_placement(filepath, mapping_file):
    assert path.is_mhealth_filename(filepath)
    mapping = pd.read_csv(mapping_file)
    sid = path.extract_sid(filepath)
    pid = path.extract_pid(filepath)
    pid_col = mapping.columns[0]
    sid_col = mapping.columns[1]
    placement_col = mapping.columns[2]
    loc = mapping.loc[(mapping[pid_col] == pid) & (
        mapping[sid_col] == sid), placement_col].values[0]
    return loc

def auto_init_placement(filepath):
    assert path.is_mhealth_filename(filepath)
    mapping_file = path.extract_location_mapping_filepath(filepath)
    if mapping_file:
        mapping = pd.read_csv(mapping_file)
    else:
        return None
    sid = path.extract_sid(filepath)
    pid = path.extract_pid(filepath)

    if len(mapping.columns) == 3:
        pid_col = mapping.columns[0]
        sid_col = mapping.columns[1]
        placement_col = mapping.columns[2]
        mask = (mapping[pid_col] == pid) & (mapping[sid_col] == sid)
    else:
        sid_col = mapping.columns[0]
        placement_col = mapping.columns[1]
        mask = mapping[sid_col] == sid
    loc = mapping.loc[mask, placement_col]
    if loc.empty:
        return None
    else:
        return loc.values[0]
    return loc

def get_placement_abbr(placement):
    tokens = placement.split(' ')
    tokens = list(map(lambda token: token[0].upper(), tokens))
    return ''.join(tokens)
