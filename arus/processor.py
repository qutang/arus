from . import o
from . import scheduler


class Processor(o.BaseOperator):
    def __init__(self, func, **kwargs):
        super().__init__()
        self._func = func
        self._scheduler = scheduler.Scheduler(**kwargs)

    def run(self, values=None, src=None, context={}):
        self._scheduler.submit(self._func, values, src=src, **context)

    def stop(self):
        self._scheduler.close()
        self._scheduler.reset()

    def get_result(self):
        try:
            result, new_context = self._scheduler.get_result(timeout=0.1)
            yield result, new_context
        except scheduler.Scheduler.ResultNotAvailableError:
            pass
