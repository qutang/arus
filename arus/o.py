import queue
import logging
import threading
import time
import enum
import collections
import abc


class O:
    """The elementary node to build up computational graph/pipeline
    """

    Pack = collections.namedtuple(
        "Package", "values signal context src")

    class Status(enum.Enum):
        """Operator Status codes."""
        OFF = enum.auto()
        ON = enum.auto()
        START = enum.auto()
        RUN = enum.auto()
        STOP = enum.auto()

    class Signal(enum.Enum):
        "Operator Signal codes."
        STOP = enum.auto()
        WAIT = enum.auto()
        DATA = enum.auto()

    class Type(enum.Enum):
        INPUT = enum.auto()
        OUTPUT = enum.auto()
        PIPE = enum.auto()

    def __init__(self, op, t=Type.PIPE, name='default'):
        self._name = name
        self._input_buffer = queue.Queue()
        self._output_buffer = queue.Queue()
        self._operator = op
        self._produce_thread = None
        self._result_thread = None
        self._status = O.Status.ON
        self._type = t

    def get_name(self):
        return self._name

    def get_type(self):
        return self._type

    def toggle(self, status=Status.OFF):
        self._status = O.Status.ON

    def get_op(self):
        return self._operator

    def start(self):
        if self._status == O.Status.OFF:
            logging.warn('Please turn on the operator at first')
            return
        self._status = O.Status.START
        logging.info('Operator is starting.')
        self._produce_thread = threading.Thread(
            target=self._produce, name=self._name + '-produce')
        self._produce_thread.daemon = True
        self._produce_thread.start()
        self._result_thread = threading.Thread(
            target=self._get_result, name=self._name + '-result'
        )
        self._result_thread.daemon = True
        self._result_thread.start()
        while not self._produce_thread.is_alive() or not self._result_thread.is_alive():
            time.sleep(0.1)
        logging.info('Operator started.')

    def stop(self):
        logging.info('Operator is stopping.')
        self._operator.stop()
        self._status = O.Status.STOP
        self._produce_thread.join()
        self._result_thread.join()
        logging.info('Operator thread stopped.')
        with self._input_buffer.mutex:
            self._input_buffer.queue.clear()
        with self._output_buffer.mutex:
            self._output_buffer.queue.clear()
        self._status = O.Status.ON
        logging.info('Stream stopped.')

    def consume(self, pack):
        if self._type != O.Type.INPUT:
            if pack.signal in [O.Signal.WAIT, O.Signal.STOP]:
                pass
            else:
                self._input_buffer.put(pack)
        else:
            logging.warn('INPUT operator does not support consume method')

    def produce(self):
        """A python generator function to get the output of the operator.

        Returns:
            output of the operator.
        """
        try:
            data = self._output_buffer.get(timeout=0.1)
            yield data
        except queue.Empty:
            yield O.Pack(values=None, signal=O.Signal.WAIT, context={}, src=self._name)
        finally:
            if self._status == O.Status.STOP:
                return
            pass

    def get_status(self) -> "O.Status":
        """Get the status code of the operator.

        Returns:
            The status code of the operator.
        """
        return self._status

    def _get_result(self):
        while True:
            for values, context in self._operator.get_result():
                if self._status == O.Status.STOP:
                    break
                else:
                    self._output_buffer.put(
                        O.Pack(values=values, signal=O.Signal.DATA,
                               context=context, src=self._name)
                    )
            if self._status == O.Status.STOP:
                break
            else:
                pass
        self._output_buffer.put(
            O.Pack(values=None, signal=O.Signal.STOP, context={}, src=self._name))

    def _produce_from_input(self):
        self._operator.run()

    def _produce_from_pipe(self):
        src = self._name
        while True:
            try:
                data = self._input_buffer.get(timeout=0.1)
                self._operator.run(
                    data.values, context=data.context, src=data.src)
            except queue.Empty:
                pass
            finally:
                if self._status == O.Status.STOP:
                    break
        self._output_buffer.put(
            O.Pack(values=None, signal=O.Signal.STOP, context={}, src=src))

    def _produce_from_output(self):
        while True:
            try:
                data = self._input_buffer.get(timeout=0.1)
                self._operator.run(
                    data.values, context=data.context, src=data.src)
            except queue.Empty:
                pass
            finally:
                if self._status == O.Status.STOP:
                    break
            self._output_buffer.put(
                O.Pack(values=None, signal=O.Signal.STOP, context={}, src=self._name))

    def _produce(self):
        logging.info('Operator thread started.')
        if self._type == O.Type.INPUT:
            self._produce_from_input()
        elif self._type == O.Type.OUTPUT:
            self._produce_from_output()
        elif self._type == O.Type.PIPE:
            self._produce_from_pipe()
        else:
            logging.error('Unknown type for the operator')
        logging.info('Operator thread is stopping.')


class BaseOperator(abc.ABC):

    def __init__(self):
        self._context = {}
        self._result = queue.Queue()

    @abc.abstractmethod
    def run(self, *, values=None, src=None, context={}):
        pass

    def set_context(self, **context):
        self._context = context

    @abc.abstractmethod
    def stop(self):
        pass

    def get_result(self):
        try:
            while True:
                result, new_context = self._result.get(timeout=0.1)
                yield result, new_context
        except queue.Empty:
            pass
