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
from .core.libs import date as arus_date
from .models import muss as arus_muss
import pandas as pd
import numpy as np
import functools


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


def process_dataset(dataset_name, approach='muss'):
    dataset_dict = load_dataset(dataset_name)
    if dataset_name == 'spades_lab':
        sr = 80
        processed_dataset = process_mehealth_dataset(
            dataset_dict, approach=approach, sr=sr)
    else:
        raise NotImplementedError('Only "spades_lab" dataset is supported.')
    output_path = os.path.join(
        mh.get_processed_path(dataset_dict['meta']['root']), approach + '.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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


def decompress_dataset(dataset_path, original_name):
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
    os.rename(os.path.join(cwd, original_name), os.path.join(cwd, name))
    output_path = os.path.join(cwd, name)
    return output_path


def process_mehealth_dataset(dataset_dict, approach='muss', **kwargs):
    if approach == 'muss' and 'window_size' not in kwargs:
        window_size = 12.8
    elif approach == 'muss' and 'window_size' in kwargs:
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
            streams_kwargs['dataset_name'] = dataset_dict['meta']['name']
            streams_kwargs['pid'] = pid
            pipeline = arus_muss.MUSSModel.get_mhealth_dataset_pipeline(
                *streams, name='{}-pipeline'.format(pid), scheduler='processes', max_processes=os.cpu_count() - 4, **streams_kwargs)  # os.cpu_count() - 4
        else:
            raise NotImplementedError('Only "muss" approach is implemented.')

        pipeline.start(start_time=start_time)

        processed = _prepare_mhealth_pipeline_output(pipeline, pid)
        results.append(processed)
    processed_dataset = pd.concat(results, axis=0, sort=False)
    processed_dataset.sort_values(
        by=[mh.FEATURE_SET_PID_COL, mh.FEATURE_SET_PLACEMENT_COL] + mh.FEATURE_SET_TIMESTAMP_COLS, inplace=True)
    return processed_dataset


def parse_annotations(dataset_name, annot_df, pid, st, et):
    if dataset_name == 'spades_lab':
        return _parse_spades_lab_annotations(annot_df, pid, st, et)
    else:
        raise NotImplementedError('Only support spades_lab dataset for now')


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
    processed[mh.FEATURE_SET_PID_COL] = pid
    return processed


def _get_annotation_durations(annot_df):
    durations = annot_df.groupby(annot_df.columns[3]).apply(
        lambda rows: np.sum(rows.iloc[:, 2] - rows.iloc[:, 1]))
    return durations


def _parse_spades_lab_annotations(annot_df, pid, st, et):
    label_list = annot_df[mh.ANNOTATION_LABEL_COL].str.lower()
    annot_df[mh.ANNOTATION_LABEL_COL] = label_list
    annot_df = annot_df.loc[(label_list != 'wear on')
                            & (label_list != 'wearon'), :]
    if annot_df.shape[0] == 0:
        return "Unknown"
    labels = annot_df.iloc[:, 3].unique()
    labels.sort()
    label = ' '.join(labels).lower().strip()

    # filter if it does not cover the entire 12.8s
    durations = _get_annotation_durations(annot_df)
    interval = int(arus_date.compute_interval(st, et, unit='ms'))

    if not np.all(durations.values >= np.timedelta64(interval, 'ms')):
        return "Transition"
    # special cases
    if pid == 'SPADES_26':
        if 'biking' in label and st.hour == 11 and st.minute > 26:
            return "Stationary cycle ergometry"
    elif pid == 'SPADES_19':
        if '3 mph' in label and 'arms on desk' in label and 'treadmill' in label:
            return "Level treadmill walking at 3 mph with arms on desk"

    if 'stairs' in label and 'up' in label:
        return 'Walking upstairs'
    elif 'stairs' in label and 'down' in label:
        return 'Walking downstairs'
    if 'mbta' in label or 'city' in label or 'outdoor' in label:
        return 'Unknown'
    if "sitting" in label and 'writing' in label:
        return 'Sitting and writing'
    elif 'stand' in label and 'writ' in label:
        return 'Standing and writing at a table'
    elif 'sit' in label and ('web' in label or 'typ' in label):
        return 'Sitting and typing on a keyboard'
    elif 'reclin' in label and ('text' in label or 'web' in label):
        return 'Reclining and using phone'
    elif 'sit' in label and 'story' in label and ('city' not in label and 'outdoor' not in label):
        return "Sitting and talking"
    elif "reclin" in label and 'story' in label:
        return 'Reclining and talking'
    elif "stand" in label and ('web' in label or 'typ' in label):
        return "Standing and typing on a keyboard"
    elif 'bik' in label and ('stationary' in label or '300' in label):
        return "Stationary cycle ergometry"
    elif ('treadmill' in label or 'walk' in label) and '1' in label:
        return "Level treadmill walking at 1 mph with arms on desk"
    elif ('treadmill' in label or 'walk' in label) and '2' in label:
        return "Level treadmill walking at 2 mph with arms on desk"
    elif 'treadmill' in label and 'phone' in label:
        return "Level treadmill walking at 3-3.5 mph while holding a phone to the ear and talking"
    elif 'treadmill' in label and 'bag' in label:
        return "Level treadmill walking at 3-3.5 mph and carrying a bag"
    elif 'treadmill' in label and 'story' in label:
        return "Level treadmill walking at 3-3.5 mph while talking"
    elif ('treadmill' in label or 'walk' in label) and 'drink' in label:
        return 'Level treadmill walking at 3-3.5 mph and carrying a drink'
    elif ('treadmill' in label or 'walk' in label) and ('3.5' in label or '3' in label):
        return 'Level treadmill walking at 3-3.5 mph'
    elif '5.5' in label or 'jog' in label or 'run' in label:
        return 'Treadmill running at 5.5 mph & 5% grade'
    elif 'laundry' in label:
        return 'Standing and folding towels'
    elif 'sweep' in label:
        return 'Standing and sweeping'
    elif 'shelf' in label and 'load' in label:
        return 'Standing loading/unloading shelf'
    elif 'lying' in label:
        return "Lying on the back"
    elif label == 'sitting' or ('sit' in label and 'still' in label):
        return "Sitting still"
    elif label == "still" or 'standing' == label or label == 'standing still':
        return "Self-selected free standing"
    else:
        return 'Unknown'
