import queue
import abc


class Operator(abc.ABC):
    def __init__(self):
        self._context = {}
        self._stop = False
        self._result = queue.Queue()

    @abc.abstractmethod
    def run(self, *, values=None, src=None, context={}):
        pass

    def set_context(self, **context):
        self._context = context

    def merge_context(self, added_context):
        new_context = {**self._context, **added_context}
        return new_context

    def stop(self):
        self._stop = True

    def get_result(self):
        try:
            while True:
                if self._stop:
                    break
                result, new_context = self._result.get(timeout=0.1)
                yield result, new_context
        except queue.Empty:
            pass
