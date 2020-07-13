"""
Datasets for activity recognition. This module provides functions to load the raw, processed datasets. It also provides functions to reproduce processed datasets from raw.

Author: Qu Tang
Date: 01/28/2020
License: GNU v3
"""

import functools
from loguru import logger
import os
import subprocess
import tarfile
import pkg_resources

import numpy as np
import pandas as pd
import wget

from .. import developer, env, generator, moment, segmentor, stream
from .. import mhealth_format as mh
# from . import _process_mhealth, _process_annotations


def get_available_sample_data():
    data_names = os.listdir(pkg_resources.resource_filename(__name__, 'data'))
    data_names = list(map(lambda name: name.replace('.csv', ''), data_names))
    return data_names


def get_sample_datapath(name):
    if name in get_available_sample_data():
        filepath = pkg_resources.resource_filename(
            __name__, f'data/{name}.csv')
    else:
        raise FileNotFoundError('The given sample data name is not supported.')
    return filepath


def get_dataset_names():
    """Report available example datasets, useful for reporting issues."""
    # delayed import to not demand bs4 unless this function is actually used
    return [
        'spades_lab'
    ]


def cache_data(dataset_name, data_home=None):
    if dataset_name == 'spades_lab':
        url = "https://github.com/qutang/MUSS/releases/latest/download/muss_data.tar.gz"
        name = dataset_name + '.tar.gz'
        original_name = 'muss_data'
    dataset_path = os.path.join(env.get_data_home(), dataset_name)
    if os.path.exists(dataset_path):
        return dataset_path
    else:
        compressed_dataset_path = download_dataset(url, name)
        dataset_path = decompress_dataset(
            compressed_dataset_path, original_name)
        os.remove(compressed_dataset_path)
    return dataset_path


def get_dataset_path(dataset_name):
    return cache_data(dataset_name)


def load_dataset(dataset_name):
    dataset_path = cache_data(dataset_name)
    if dataset_name == 'spades_lab':
        return mh.traverse_dataset(dataset_path)


# def process_dataset(dataset_name, approach='muss'):
#     dataset_dict = load_dataset(dataset_name)
#     if dataset_name == 'spades_lab':
#         sr = 80
#         processed_dataset = _process_mhealth.process_mehealth_dataset(
#             dataset_dict, approach=approach, sr=sr)
#     else:
#         raise NotImplementedError('Only "spades_lab" dataset is supported.')
#     output_path = os.path.join(
#         mh.get_processed_path(dataset_dict['meta']['root']), approach + '.csv')
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     processed_dataset.to_csv(
#         output_path, float_format='%.6f', header=True, index=False)
#     logger.info('Processed {} dataset is saved to {}'.format(
#         dataset_name, output_path))
#     dataset_dict['processed'][approach] = output_path
#     return dataset_dict


def download_dataset(url, name):
    spades_lab_url = url
    output_path = os.path.join(env.get_data_home(), name)
    if os.path.exists(output_path):
        return output_path
    else:
        result = wget.download(spades_lab_url, out=output_path)
    return result


def decompress_dataset(dataset_path, original_name):
    cwd = os.path.dirname(dataset_path)
    name = os.path.basename(dataset_path).split('.')[0]
    if developer.command_is_available('tar --version'):
        logger.info('Using system tar command to decompress dataset file')
        decompress_cmd = ['tar', '-xzf', dataset_path]
        subprocess.run(' '.join(decompress_cmd),
                       check=True, shell=True, cwd=cwd)
    else:
        logger.info('Using Python tar module to decompress data file')
        tar = tarfile.open(dataset_path)
        tar.extractall(path=cwd)
        tar.close()
    os.rename(os.path.join(cwd, original_name), os.path.join(cwd, name))
    output_path = os.path.join(cwd, name)
    return output_path


def parse_annotations(dataset_name, annot_df, pid, st, et):
    if dataset_name == 'spades_lab':
        return _process_annotations._parse_spades_lab_annotations(annot_df, pid, st, et)
    else:
        raise NotImplementedError('Only support spades_lab dataset for now')
