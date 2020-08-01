"""
segmentor classes that takes a streams of dataframes and generate chunks in various ways.

Author: Qu Tang

Date: 02/04/2020

License: GNU v3
"""
import queue
import pandas as pd
from . import extensions as ext
from . import moment
from loguru import logger
from . import operator


class Segmentor(operator.Operator):
    """Base class for segmentors.

    Segmentors are used to segment streaming data and generate chunks in different ways.
    """

    def __init__(self, ref_st: "str, datetime, numpy.datetime64, pandas.Timestamp" = None, st_col: int = 0, et_col: int = None):
        """Create Segmentor instance.

        Arguments:
            ref_st: The reference start time for the first segmented window of data.
            st_col: The column with start time timestamps in the streaming data.
            et_col: The column with stop time timestamps in the streaming data. If it is `None`, `et_col = st_col`.
        """
        super().__init__()
        self._st_col = st_col
        self._et_col = et_col or self._st_col
        self._ref_st = ref_st
        self.reset()

    def set_ref_time(self, ts: "str, datetime, numpy.datetime64, pandas.Timestamp"):
        """Set reference start time.

        Arguments:
            ts: The timestamp to be set.
        """
        self._ref_st = ts

    def reset(self):
        """Reset the segmentor.
        """
        pass

    def run(self, values=None, src=None, context={}):
        self._context = {**self._context, **context}
        for result, new_context in self.segment(values):
            if self._stop:
                break
            self._result.put((result, new_context))

    def segment(self, data: "pandas.Dataframe"):
        """A python generator function to output segmented data.

        The default behavior is to output each row of the burst of streaming data.

        Arguments:
            data: the input burst of streaming data.
        """
        if data is None:
            return
        for index, row in data.iterrows():
            if self._stop:
                break
            yield row, self._context


class SlidingWindowSegmentor(Segmentor):
    """Segment straming data with sliding window method.
    """

    def __init__(self, window_size: float, **kwargs):
        """Create SlidingWindowSegmentor instance.

        Arguments:
            window_size: The window size in seconds.

        Raises:
            ValueError: Raise when window size is smaller than zero.
        """
        super().__init__(**kwargs)
        if window_size == 0:
            raise ValueError('Window size should be greater than zero.')
        self._ws = window_size

    def stop(self):
        super().stop()
        self.reset()

    def reset(self):
        self._current_segment = []
        self._current_seg_st = None
        self._current_seg_et = None
        self._previous_seg_st = None
        self._previous_seg_et = None

    def segment(self, data: 'pandas.Dataframe'):
        """A python generator function to output segmented data.

        It will segment the incoming streaming data with sliding window method and yield segmented data.

        Arguments:
            data: the input burst of streaming data.
        """

        if data is None:
            yield data, self._context
        else:
            et = data.iloc[-1, self._et_col]
            if self._ref_st is not None and moment.Moment(self._ref_st).to_unix_timestamp() > moment.Moment(et).to_unix_timestamp():
                logger.warning(
                    'Referenced start time is after the end time of the input data, this generates no segments from the data.')
                return
            segments = self._extract_segments(data)
            for segment, seg_st, seg_et in segments:
                if self._stop:
                    break
                self._current_seg_st = self._current_seg_st or seg_st
                self._current_seg_et = self._current_seg_et or seg_et
                if self._current_seg_st == seg_st and self._current_seg_et == seg_et:
                    self._current_segment.append(segment)
                else:
                    self._current_segment = pd.concat(
                        self._current_segment, axis=0, sort=False, ignore_index=True)
                    result = self._current_segment
                    new_context = {
                        "start_time": self._current_seg_st,
                        "stop_time": self._current_seg_et,
                        "prev_start_time": self._previous_seg_st,
                        "prev_stop_time": self._previous_seg_et,
                    }
                    self._context = {**self._context, **new_context}
                    self._current_segment = [segment]
                    self._previous_seg_st = self._current_seg_st
                    self._previous_seg_et = self._current_seg_et
                    self._current_seg_st = seg_st
                    self._current_seg_et = seg_et
                    yield result, self._context

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
            segment = ext.pandas.segment_by_time(
                data,
                seg_st=seg_st,
                seg_et=seg_et,
                st_col=self._st_col,
                et_col=self._et_col
            )
            segments.append((segment, seg_st, seg_et))
        return segments
