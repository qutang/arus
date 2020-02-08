"""
synchronizer class that takes a set of chunks from multiple sources, sync and assemble and output them as a list.

Author: Qu Tang
Date: 02/07/2020
License: GNU v3
"""
from . import moment


class Synchronizer:
    def __init__(self):
        self._buffer = {}
        self._num = 0
        pass

    def add_sources(self, n):
        self._num += n

    def remove_sources(self, n):
        self._num -= n

    def add_source(self):
        self._num += 1

    def remove_source(self):
        self._num -= 1
        self._num = max(0, self._num)

    def reset(self):
        self._buffer.clear()

    def sync(self, data, st, et, source_id, **kwargs):
        st = moment.Moment(st)
        et = moment.Moment(et)
        if st.to_unix_timestamp() not in self._buffer:
            self._buffer[st.to_unix_timestamp()] = {}
        # If source id exists, new data will overwrite old data
        self._buffer[st.to_unix_timestamp()][source_id] = (
            data, st, et, kwargs)
        assembled = self._buffer[st.to_unix_timestamp()]
        if len(assembled.keys()) == self._num:
            # now the assemble is ready
            assembled = self._format_assembled(assembled)
            del self._buffer[st.to_unix_timestamp()]
            return assembled

    def _format_assembled(self, assembled):
        data = []
        start_time = None
        stop_time = None
        source_ids = []
        kwargs_list = []
        for source_id, item in assembled.items():
            source_ids.append(source_id)
            data.append(item[0])
            start_time = item[1]
            stop_time = item[2]
            kwargs_list.append(item[3])
        result = (data, source_ids, kwargs_list, start_time, stop_time)
        return result
