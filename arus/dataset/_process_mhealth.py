import numpy as np
from .. import generator, segmentor, stream, developer
import logging
import pandas as pd
from .. import mhealth_format as mh
from ..models import muss as arus_muss
from ..models import nn
from ..core import pipeline as arus_pipeline
import os


def process_mehealth_dataset(dataset_dict, approach='muss', **kwargs):
    kwargs = _parse_kwargs(approach, kwargs)

    results = []

    for pid in dataset_dict['subjects'].keys():
        logging.info('Start processing {}'.format(pid))
        if approach == 'muss':
            processed = _process_muss(
                dataset_dict, pid, sr=kwargs['sr'], window_size=kwargs['window_size'])
            results.append(processed)
        elif approach == 'nn':
            processed = _process_nn(dataset_dict, pid, sr=kwargs['sr'])
            if processed is not None:
                processed.reset_index(drop=False, inplace=True)
                results.append(processed)
        else:
            raise NotImplementedError('Only "muss" approach is implemented.')

    processed_dataset = pd.concat(results, axis=0, sort=False)
    if approach == 'muss':
        processed_dataset.sort_values(
            by=[mh.FEATURE_SET_PID_COL, mh.FEATURE_SET_PLACEMENT_COL] + mh.FEATURE_SET_TIMESTAMP_COLS, inplace=True)
    return processed_dataset


def _parse_kwargs(approach, kwargs):
    if approach == 'muss' and 'window_size' not in kwargs:
        kwargs['window_size'] = 12.8
    elif approach == 'muss' and 'window_size' in kwargs:
        pass
    elif approach == 'nn':
        kwargs['window_size'] = 3600
    else:
        raise NotImplementedError('You must provide a valid window size')

    if 'sr' not in kwargs:
        raise NotImplementedError('You must provide a valid sampling rate')

    developer.logging_dict(kwargs)
    logging.info('sr: {}'.format(kwargs['sr']))
    logging.info('window size: {}'.format(kwargs['window_size']))
    return kwargs


def _process_muss(dataset_dict, pid, sr, window_size=12.8):
    dataset_path = dataset_dict['meta']['root']
    start_time = mh.get_session_start_time(pid, dataset_path)
    streams, streams_kwargs = _prepare_mhealth_streams(
        dataset_dict, pid, window_size, sr)
    streams_kwargs['dataset_name'] = dataset_dict['meta']['name']
    streams_kwargs['pid'] = pid
    pipeline = arus_muss.MUSSModel.get_mhealth_dataset_pipeline(
        *streams, name='{}-pipeline'.format(pid), scheduler='processes', max_processes=os.cpu_count() - 4, **streams_kwargs)  # os.cpu_count() - 4
    pipeline.start(start_time=start_time)
    processed = _prepare_pipeline_output(pipeline, pid)
    return processed


def _prepare_mhealth_streams(dataset_dict, pid, window_size, sr, use_annotations=True):
    streams = []
    subject_data_dict = dataset_dict['subjects']
    streams_kwargs = {}
    # sensor streams
    for p in subject_data_dict[pid]['sensors'].keys():
        stream_name = p
        gr = generator.MhealthSensorFileGenerator(
            *subject_data_dict[pid]['sensors'][p])
        seg = segmentor.SlidingWindowSegmentor(window_size=window_size)
        pid_sid_stream = stream.Stream(gr, seg, name=stream_name)
        streams.append(pid_sid_stream)
        streams_kwargs[stream_name] = {
            'sr': sr
        }

    if use_annotations:
        # annotation streams
        for a in subject_data_dict[pid]['annotations'].keys():
            stream_name = a
            gr = generator.MhealthAnnotationFileGenerator(
                *subject_data_dict[pid]['annotations'][a])
            seg = segmentor.SlidingWindowSegmentor(
                window_size=window_size, st_col=1, et_col=2)
            annotation_stream = stream.Stream(gr, seg, name=stream_name)
            streams.append(annotation_stream)

    return streams, streams_kwargs


def _prepare_pipeline_output(pipeline, pid):
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
    if processed is not None:
        processed[mh.FEATURE_SET_PID_COL] = pid
    return processed


def _process_nn(dataset_dict, pid, sr):
    window_size = 3600
    dataset_path = dataset_dict['meta']['root']
    start_time = mh.get_session_start_time(
        pid, dataset_path, round_to='minute')
    streams, streams_kwargs = _prepare_mhealth_streams(
        dataset_dict, pid, window_size, sr, use_annotations=False)
    streams_kwargs['dataset_name'] = dataset_dict['meta']['name']
    streams_kwargs['pid'] = pid
    pipeline = arus_pipeline.Pipeline(
        max_processes=os.cpu_count() - 4, scheduler='processes', name='process-nn-pipeline')
    for stream in streams:
        pipeline.add_stream(stream)
    pipeline.set_processor(nn.preprocess_processor, **streams_kwargs)
    pipeline.start(start_time=start_time)
    pid = pid.split('_')[1]
    processed = _prepare_pipeline_output(pipeline, pid)
    return processed
