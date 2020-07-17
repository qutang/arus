import queue
from loguru import logger
import threading
import time
import enum
import collections


class Node:
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
        self._status = Node.Status.ON
        self._type = t

    def get_name(self):
        return self._name

    def get_type(self):
        return self._type

    def toggle(self, status=Status.OFF):
        self._status = Node.Status.ON

    def get_op(self):
        return self._operator

    def start(self):
        if self._status == Node.Status.OFF:
            logger.warning('Please turn on the operator at first')
            return
        self._status = Node.Status.START
        logger.info(f'{self._name} Operator is starting.')
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
        logger.info(f'{self._name} Operator started.')

    def stop(self):
        logger.info(f'{self._name} Operator is stopping.')
        self._operator.stop()
        self._status = Node.Status.STOP
        self._produce_thread.join()
        self._result_thread.join()
        logger.info(f'{self._name} Operator thread stopped.')
        with self._input_buffer.mutex:
            self._input_buffer.queue.clear()
        with self._output_buffer.mutex:
            self._output_buffer.queue.clear()
        self._status = Node.Status.ON
        logger.info(f'{self._name} Operator stopped.')

    def consume(self, pack):
        if self._type != Node.Type.INPUT:
            if pack.signal in [Node.Signal.WAIT, Node.Signal.STOP]:
                pass
            else:
                self._input_buffer.put(pack)
        else:
            logger.warning('INPUT operator does not support consume method')

    def produce(self):
        """A python generator function to get the output of the operator.

        Returns:
            output of the operator.
        """
        try:
            data = self._output_buffer.get(timeout=0.1)
            yield data
        except queue.Empty:
            yield Node.Pack(values=None, signal=Node.Signal.WAIT, context={}, src=self._name)
        finally:
            if self._status == Node.Status.STOP:
                return
            pass

    def get_status(self) -> "Node.Status":
        """Get the status code of the operator.

        Returns:
            The status code of the operator.
        """
        return self._status

    def _get_result(self):
        while True:
            for values, context in self._operator.get_result():
                if self._status == Node.Status.STOP:
                    break
                else:
                    self._output_buffer.put(
                        Node.Pack(values=values, signal=Node.Signal.DATA,
                                  context=context, src=self._name)
                    )
            if self._status == Node.Status.STOP:
                break
            else:
                pass
        self._output_buffer.put(
            Node.Pack(values=None, signal=Node.Signal.STOP, context={}, src=self._name))

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
                if self._status == Node.Status.STOP:
                    break
        self._output_buffer.put(
            Node.Pack(values=None, signal=Node.Signal.STOP, context={}, src=src))

    def _produce_from_output(self):
        while True:
            try:
                data = self._input_buffer.get(timeout=0.1)
                self._operator.run(
                    data.values, context=data.context, src=data.src)
            except queue.Empty:
                pass
            finally:
                if self._status == Node.Status.STOP:
                    break
            self._output_buffer.put(
                Node.Pack(values=None, signal=Node.Signal.STOP, context={}, src=self._name))

    def _produce(self):
        logger.info(f'{self._name} Operator thread started.')
        if self._type == Node.Type.INPUT:
            self._produce_from_input()
        elif self._type == Node.Type.OUTPUT:
            self._produce_from_output()
        elif self._type == Node.Type.PIPE:
            self._produce_from_pipe()
        else:
            logger.error(f'Unknown type for the operator {self._name}')
        logger.info(f'{self._name} Operator thread is stopping.')
