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
        super().stop()
        self._scheduler.close()
        self._scheduler.reset()

    def shutdown(self):
        self._scheduler.shutdown()

    def get_result(self):
        try:
            while True:
                if self._stop:
                    break
                result, new_context = self._scheduler.get_result(timeout=0.1)
                yield result, new_context
        except scheduler.Scheduler.ResultNotAvailableError:
            pass
