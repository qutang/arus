"""
Datasets for activity recognition. This module provides functions to load the raw, processed datasets. It also provides functions to reproduce processed datasets from raw.

Author: Qu Tang
Date: 01/28/2020
License: GNU v3
"""

import os
from . import env
import wget
import tarfile
from . import developer
from . import mhealth_format as mh
import logging
import subprocess
from .core.stream import SensorFileSlidingWindowStream
from .models import muss as arus_muss
import pandas as pd


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
    if dataset_name == 'spades_lab':
        url = "https://github.com/qutang/MUSS/releases/latest/download/muss_data.tar.gz"
        name = dataset_name + '.tar.gz'
    dataset_path = os.path.join(env.get_data_home(), dataset_name)
    if os.path.exists(dataset_path):
        return dataset_path
    else:
        compressed_dataset_path = download_dataset(url, name)
        dataset_path = decompress_dataset(compressed_dataset_path)
        os.remove(compressed_dataset_path)
    return dataset_path


def get_dataset_path(dataset_name):
    return cache_data(dataset_name)


def load_processed_dataset(dataset_name, cache=True):
    pass


def load_raw_dataset(dataset_name):
    dataset_path = cache_data(dataset_name)
    if dataset_name == 'spades_lab':
        return mh.traverse_dataset(dataset_path)


def process_raw_dataset(dataset_name, approach='muss'):
    dataset_path = get_dataset_path(dataset_name)
    dataset_dict = load_raw_dataset(dataset_name)
    processed_dataset = _process_mehealth_dataset(
        dataset_path, dataset_dict, approach=approach)
    output_path = os.path.join(
        env.get_data_home(), dataset_name + '.' + approach + '.feature.csv')
    processed_dataset.to_csv(
        output_path, float_format='%.6f', header=True, index=False)
    logging.info('Processed {} dataset is saved to {}'.format(
        dataset_name, output_path))


def download_dataset(url, name):
    spades_lab_url = url
    output_path = os.path.join(env.get_data_home(), name)
    if os.path.exists(output_path):
        return output_path
    else:
        result = wget.download(spades_lab_url, out=output_path)
    return result


def decompress_dataset(dataset_path):
    cwd = os.path.dirname(dataset_path)
    name = os.path.basename(dataset_path).split('.')[0]
    if developer.command_is_available('tar --version'):
        logging.info('Using system tar command to decompress dataset file')
        decompress_cmd = ['tar', '-xzf', dataset_path]
        subprocess.run(' '.join(decompress_cmd),
                       check=True, shell=True, cwd=cwd)
    else:
        logging.info('Using Python tar module to decompress data file')
        tar = tarfile.open(dataset_path)
        tar.extractall(path=cwd)
        tar.close()
    os.rename(os.path.join(cwd, 'muss_data'), os.path.join(cwd, name))
    output_path = os.path.join(cwd, name)
    return output_path


def _process_mehealth_dataset(dataset_path, dataset_dict, approach='muss', **kwargs):
    if 'window_size' in kwargs:
        window_size = kwargs['window_size'] or 12.8
    else:
        window_size = 12.8
    if 'sr' in kwargs:
        sr = kwargs['sr'] or 80
    else:
        sr = 80

    results = []
    for pid in dataset_dict.keys():
        logging.info('Start processing {}'.format(pid))
        pid_processed = None
        start_time = mh.get_session_start_time(pid, dataset_path)
        pipeline_kwargs = {}
        streams = []
        for p in dataset_dict[pid]['sensors'].keys():
            stream_name = p
            pid_sid_stream = SensorFileSlidingWindowStream(
                dataset_dict[pid]['sensors'][p], window_size=window_size, sr=sr, name=stream_name)
            streams.append(pid_sid_stream)
            pipeline_kwargs[stream_name] = {
                'sr': sr
            }
        pipeline = arus_muss.MUSSModel.get_mhealth_dataset_pipeline(
            *streams, name='{}-pipeline'.format(pid), scheduler='processes', max_processes=os.cpu_count() - 4, **pipeline_kwargs)
        pipeline.start(start_time=start_time)
        for df, st, et, prev_st, prev_et, name in pipeline.get_iterator():
            if df.empty:
                continue
            if pid_processed is not None:
                pid_processed = pd.concat(
                    (pid_processed, df), sort=False, axis=0)
            else:
                pid_processed = df
        logging.info('Pipeline {} has completed.'.format(pid))
        pipeline.stop()
        pid_processed['PID'] = pid
        results.append(pid_processed)
    processed_dataset = pd.concat(results, axis=0, sort=False)
    processed_dataset.sort_values(
        by=['PID', 'PLACEMENT', 'HEADER_TIME_STAMP', 'START_TIME'], inplace=True)
    return processed_dataset
