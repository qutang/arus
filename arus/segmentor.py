"""
segmentor functions that takes a streams of dataframes and generate chunks in various ways.

Author: Qu Tang
Date: 02/04/2020
License: GNU v3
"""
import queue
import pandas as pd
from . import extensions
from . import moment
import logging


class Segmentor:
    def __init__(self, ref_st=None, st_col=0, et_col=None):
        self._st_col = st_col
        self._et_col = et_col or self._st_col
        self._ref_st = ref_st
        self.reset()

    def set_ref_time(self, ts):
        self._ref_st = ts

    def reset(self):
        pass

    def segment(self, data):
        if data is None:
            raise StopIteration
        for index, row in data.iterrows():
            yield row


class SlidingWindowSegmentor(Segmentor):
    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        if window_size == 0:
            raise ValueError('Window size should be greater than zero.')
        self._ws = window_size

    def reset(self):
        self._current_segment = []
        self._current_seg_st = None
        self._current_seg_et = None
        self._previous_seg_st = None
        self._previous_seg_et = None

    def segment(self, data):
        et = data.iloc[-1, self._et_col]
        if self._ref_st is not None and moment.Moment(self._ref_st).to_unix_timestamp() > moment.Moment(et).to_unix_timestamp():
            logging.warn(
                'Referenced start time is after the end time of the input data, this generates no segments from the data.')
            raise StopIteration
        if data is None:
            raise StopIteration
        else:
            segments = self._extract_segments(data)
            for segment, seg_st, seg_et in segments:
                self._current_seg_st = self._current_seg_st or seg_st
                self._current_seg_et = self._current_seg_et or seg_et
                if self._current_seg_st == seg_st and self._current_seg_et == seg_et:
                    self._current_segment.append(segment)
                else:
                    self._current_segment = pd.concat(
                        self._current_segment, axis=0, sort=False)
                    result = (
                        self._current_segment,
                        self._current_seg_st,
                        self._current_seg_et,
                        self._previous_seg_st,
                        self._previous_seg_et
                    )
                    self._current_segment = [segment]
                    self._previous_seg_st = self._current_seg_st
                    self._previous_seg_et = self._current_seg_et
                    self._current_seg_st = seg_st
                    self._current_seg_et = seg_et
                    yield result

    def _extract_segments(self, data):
        if data.empty:
            return []
        data_et = data.iloc[-1, self._et_col]
        data_st = data.iloc[0, self._st_col]
        if self._ref_st is None:
            self._ref_st = data_st
        window_ts_marks = pd.date_range(start=self._ref_st,
                                        end=data_et,
                                        freq=str(self._ws * 1000) + 'ms'
                                        )
        self._ref_st = window_ts_marks[-1]
        segments = []
        for seg_st in window_ts_marks:
            seg_et = seg_st + \
                pd.Timedelta(self._ws * 1000, unit='ms')
            segment = extensions.pandas.segment_by_time(
                data,
                seg_st=seg_st,
                seg_et=seg_et,
                st_col=self._st_col,
                et_col=self._et_col
            )
            segments.append((segment, seg_st, seg_et))
        return segments
