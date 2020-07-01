"""
synchronizer class that takes a set of chunks from multiple sources, sync and assemble and output them as a list.

Author: Qu Tang
Date: 02/07/2020
License: GNU v3
"""
from . import moment
from . import operator


class Synchronizer(operator.Operator):
    def __init__(self):
        super().__init__()
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

    def stop(self):
        super().stop()
        self.reset()

    def reset(self):
        self._buffer.clear()

    def run(self, values, src=None, context={}):
        """[summary]

        Arguments:
            values {[type]} -- [description]

        Keyword Arguments:
            src {[type]} -- [description] (default: {None})
            context {dict} -- It has to provide `data_id` as an indicator of the incoming data source, `start_time` and `stop_time` as indicators of the start and stop boundary of the data. (default: {{}})
        """
        new_context = self.merge_context(context)
        del new_context['data_id']
        st = new_context['start_time']
        et = new_context['stop_time']
        data_id = context['data_id']
        result, new_context = self._sync(values, st, et, data_id, new_context)
        if result is not None:
            self._result.put((result, new_context))

    def _sync(self, data, st, et, data_id, new_context):
        st = moment.Moment(st)
        et = moment.Moment(et)
        if st.to_unix_timestamp() not in self._buffer:
            self._buffer[st.to_unix_timestamp()] = {}
        # If source id exists, new data will overwrite old data
        self._buffer[st.to_unix_timestamp()][data_id] = data
        assembled = self._buffer[st.to_unix_timestamp()]
        if len(assembled.keys()) == self._num:
            # now the assemble is ready
            assembled, new_context = self._format_assembled(
                assembled, new_context)
            del self._buffer[st.to_unix_timestamp()]
            return assembled, new_context
        return None, new_context

    def _format_assembled(self, assembled, context):
        result = []
        data_ids = []
        for data_id, values in assembled.items():
            data_ids.append(data_id)
            result.append(values)
        new_context = {
            **context, "data_ids": data_ids
        }
        return result, new_context
