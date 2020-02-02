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
from .core.stream import AnnotationFileSlidingWindowStream
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
    dataset_dict = load_raw_dataset(dataset_name)
    if dataset_name == 'spades_lab':
        sr = 80
        processed_dataset = process_mehealth_dataset(
            dataset_dict, approach=approach, sr=sr)
    else:
        raise NotImplementedError('Only "spades_lab" dataset is supported.')
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


def process_mehealth_dataset(dataset_dict, approach='muss', **kwargs):
    if approach == 'muss':
        window_size = 12.8
    else:
        if 'window_size' in kwargs:
            window_size = kwargs['window_size']
        else:
            raise NotImplementedError('You must provide a valid window size')

    if 'sr' in kwargs:
        sr = kwargs['sr']
    else:
        raise NotImplementedError('You must provide a valid sampling rate')

    developer.logging_dict(kwargs, level=logging.INFO)
    logging.info('sr: {}'.format(sr))
    logging.info('window size: {}'.format(window_size))

    results = []
    dataset_path = dataset_dict['meta']['root']

    for pid in dataset_dict['subjects'].keys():
        logging.info('Start processing {}'.format(pid))

        start_time = mh.get_session_start_time(pid, dataset_path)

        streams, streams_kwargs = _prepare_mhealth_streams(
            dataset_dict, pid, window_size, sr)

        if approach == 'muss':
            pipeline = arus_muss.MUSSModel.get_mhealth_dataset_pipeline(
                *streams, name='{}-pipeline'.format(pid), scheduler='processes', max_processes=os.cpu_count() - 4, **streams_kwargs)
        else:
            raise NotImplementedError('Only "muss" approach is implemented.')

        pipeline.start(start_time=start_time)

        processed = _prepare_mhealth_pipeline_output(pipeline, pid)
        results.append(processed)

    processed_dataset = pd.concat(results, axis=0, sort=False)
    processed_dataset.sort_values(
        by=['PID', 'PLACEMENT', 'HEADER_TIME_STAMP', 'START_TIME'], inplace=True)
    return processed_dataset


def parse_spades_lab_annotations(annot_df):
    pass


def _prepare_mhealth_streams(dataset_dict, pid, window_size, sr):
    streams = []
    subject_data_dict = dataset_dict['subjects']
    streams_kwargs = {}
    # sensor streams
    for p in subject_data_dict[pid]['sensors'].keys():
        stream_name = p
        pid_sid_stream = SensorFileSlidingWindowStream(
            subject_data_dict[pid]['sensors'][p],
            window_size=window_size,
            sr=sr,
            name=stream_name
        )
        streams.append(pid_sid_stream)
        streams_kwargs[stream_name] = {
            'sr': sr
        }

    # annotation streams
    for a in subject_data_dict[pid]['annotations'].keys():
        stream_name = a
        annotation_stream = AnnotationFileSlidingWindowStream(
            subject_data_dict[pid]['annotations'][a],
            window_size=window_size,
            name=stream_name
        )
        streams.append(annotation_stream)

    return streams, streams_kwargs


def _prepare_mhealth_pipeline_output(pipeline, pid):
    processed = None
    for df, st, et, prev_st, prev_et, name in pipeline.get_iterator():
        if df.empty:
            continue
        if processed is not None:
            processed = pd.concat(
                (processed, df), sort=False, axis=0)
        else:
            processed = df
    logging.info('Pipeline {} has completed.'.format(pid))
    pipeline.stop()
    processed['PID'] = pid
    return processed
