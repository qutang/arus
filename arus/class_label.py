import numpy as np
import pandas as pd
from . import stream2, generator, segmentor, synchronizer, processor, scheduler, pipeline, node
from loguru import logger
import sys
from . import extensions as ext
from . import mhealth_format as mh
from .error_code import ErrorCode
import tqdm


class ClassSet:
    def __init__(self, raw_sources, aids):
        self._raw_sources = raw_sources
        self._aids = aids

    def _validate_input_as_df(self):
        if len(self._raw_sources) == 0:
            return
        if type(self._raw_sources[0]) is not pd.DataFrame:
            logger.error(
                '[Error code: {ErrorCode.INPUT_ARGUMENT_FORMAT_ERROR.name}] To compute class labels offline, the input raw data should be annotation dataframe stored in mhealth format.')
            sys.exit(ErrorCode.INPUT_ARGUMENT_FORMAT_ERROR.name)

    def _validate_input_as_generator(self):
        if type(self._raw_sources[0]) is not generator.Generator:
            logger.error(
                f'[Error code: {ErrorCode.INPUT_ARGUMENT_FORMAT_ERROR.name}] To compute class labels online, the input raw data should be arus Generator object.')
            sys.exit(ErrorCode.INPUT_ARGUMENT_FORMAT_ERROR.name)

    def compute_offline(self, window_size, class_func, task_names, start_time=None, stop_time=None, step_size=None, show_progress=True, **kwargs):
        self._validate_input_as_df()
        self._task_names = task_names
        step_size = step_size or window_size
        window_start_markers = ext.pandas.split_into_windows(
            *self._raw_sources, step_size=step_size, st=start_time, et=stop_time, st_col=1, et_col=2)
        class_vectors = []
        with tqdm.tqdm(total=len(window_start_markers), disable=not show_progress) as bar:
            for window_st in window_start_markers:
                window_et = window_st + pd.Timedelta(window_size, unit='s')
                dfs = []
                for raw_df in self._raw_sources:
                    df = ext.pandas.segment_by_time(
                        raw_df, seg_st=window_st, seg_et=window_et, st_col=1, et_col=2)
                    dfs.append(df)
                class_vector = class_func(*dfs, st=window_st, et=window_et,
                                          task_names=task_names, aids=self._aids, **kwargs)
                class_vectors.append(class_vector)
                bar.update()
                bar.set_description(
                    f'Computed class set for window: {window_st}')

        if len(class_vectors) == 0:
            self._class_set = None
        else:
            self._class_set = pd.concat(
                class_vectors, axis=0, ignore_index=True, sort=False)

    def compute_per_window(self, class_func, task_names, **kwargs):
        self._validate_input_as_df()
        class_vector = class_func(
            self._raw_dfs, task_names=task_names, aids=self._aids, **kwargs)
        self._class_set = class_vector
        self._task_names = task_names

    def compute_online(self, seg, class_func, task_names, start_time, aids, **kwargs):
        src_streams = []
        for src_generator, aid in zip(self._raw_sources, self._aids):
            src_stream = stream2.Stream(
                src_generator, seg, name=f'{aid}-stream')
            src_stream.set_essential_context(
                start_time=start_time, stream_id=aid)
            src_streams.append(src_stream)

        sync = synchronizer.Synchronizer()
        sync.add_sources(n=len(self._raw_sources))

        proc = processor.Processor(class_func,
                                   mode=scheduler.Scheduler.Mode.PROCESS,
                                   scheme=scheduler.Scheduler.Scheme.SUBMIT_ORDER,
                                   max_workers=10)
        proc.set_context(**kwargs)

        self._pip = node.Node(op=pipeline.Pipeline(*src_streams,
                                                   synchronizer=sync,
                                                   processor=proc, name='online-classset-pipeline'),
                              t=node.Node.Type.INPUT, name='online-classet-pipeline')
        self._pip.start()
        for pack in self._pip.produce():
            if pack.signal == node.Node.Signal.DATA:
                if pack.values is not None:
                    yield pack.values
            elif pack.signal == node.Node.Signal.STOP:
                break
        self._pip.stop()

    def reset(self):
        self._class_set = None
        self._task_names = None

    def get_class_set(self):
        if self._class_set is not None:
            return self._class_set.copy(deep=True)
        else:
            return None

    def get_task_names(self):
        return self._task_names

    @staticmethod
    def get_annotation_durations(annot_df, label_col):
        durations = annot_df.groupby(label_col).apply(
            lambda rows: np.sum(rows.loc[:, mh.STOP_TIME_COL] - rows.loc[:, mh.START_TIME_COL]))
        return durations
