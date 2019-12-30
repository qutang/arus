"""
Module of testing utility functions

This module provides utility functions for testing modules in `arus.core`.

Author: Qu Tang

Date: 2019-11-15

License: see LICENSE file
"""

import os
import glob


def load_test_data(file_type='mhealth', sensor_type='sensor', file_num='multiple', exception_type='consistent_sr'):
    """Load test data that are bundled in the package.

    Args:
        file_type (str, optional): either files stored in 'mhealth' or 'actigraph' csv format. Defaults to 'mhealth'.
        sensor_type (str, optional): {'sensor', 'annotation', 'feature', 'class_labels'}. Defaults to 'sensor'.
        file_num (str, optional): either 'single' or 'multiple' files. Defaults to 'multiple'.
        exception_type (str, optional): 'consistent_sr' (consistent sampling rate), 'inconsistent_sr' (inconsistent sampling rate), 'missing' (with missing data), 'no_missing' (with no missing data), 'multi_placements' (features from multiple placements), 'single_placement' (features from single placement), 'multi_tasks' (multiple class columns), 'single_task' (single class column). Defaults to 'consistent_sr'.

    Returns:
        tuple: the first element is the list of filepaths found and sorted, the second element is the sampling rate if `sensor_type` is 'sensor', otherwise it is `None`.
    """

    cwd = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(cwd, '..', 'data', file_num,
                               file_type, sensor_type, exception_type)
    meta_file = os.path.join(data_folder, 'meta.csv')
    if file_num == 'multiple':
        data = list(filter(lambda f: 'meta.csv' != os.path.basename(f), glob.glob(
            os.path.join(data_folder, '*.csv*'))))
        data = sorted(data)
    else:
        data = list(filter(lambda f: 'meta.csv' != os.path.basename(f), glob.glob(
            os.path.join(data_folder, '*.csv*'))))[0]
    if sensor_type == 'sensor':
        with open(meta_file, 'r') as meta:
            sr = int(meta.readline().split(',')[1])
    else:
        sr = None
    return data, sr
