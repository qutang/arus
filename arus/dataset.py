"""
Datasets for activity recognition. This module provides functions to load the raw, processed datasets. It also provides functions to reproduce processed datasets from raw.

Author: Qu Tang
Date: 01/28/2020
License: GNU v3
"""

import os
from . import env


def get_dataset_names():
    """Report available example datasets, useful for reporting issues."""
    # delayed import to not demand bs4 unless this function is actually used
    return [
        'spades_lab',
        'spades_freeliving',
        'camspades_lab',
        'camspades_freeliving'
    ]


def cache_data(dataset_name, data_home=None):
    pass


def get_dataset_path(dataset_name):
    pass


def load_processed_dataset(dataset_name, cache=True):
    pass


def load_raw_dataset(dataset_name, pid=None, cache=True):
    pass


def process_raw_dataset(dataset_name):
    pass
