"""
scheduler class that handles the execution order of functions when using multiprocessing.

Author: Qu Tang

Date: 02/07/2020

License: GNU v3
"""
import loky
import enum
from concurrent import futures
import queue
import threading
import logging


class Scheduler:
    """Scheduler to support different scheduling schemes when running tasks on subprocess workers.
    """
    class ClosedError(Exception):
        pass

    class ResultNotAvailableError(Exception):
        pass

    class Mode(enum.Enum):
        THREAD = enum.auto()
        PROCESS = enum.auto()

    class Scheme(enum.Enum):
        AFTER_PREVIOUS_DONE = enum.auto()
        EXECUTION_ORDER = enum.auto()
        SUBMIT_ORDER = enum.auto()

    def __init__(self, mode: "Scheduler.Mode" = Mode.PROCESS, scheme: "Scheduler.Scheme" = Scheme.EXECUTION_ORDER, max_workers: int = None):
        """Create Scheduler instance.

        Arguments:
            mode: run tasks in process or in thread.
            scheme: the scheduling scheme when executing tasks.
            max_workers: the max number of workers.
        """
        if mode == Scheduler.Mode.PROCESS:
            self._executor = loky.get_reusable_executor(
                max_workers=max_workers, timeout=10)
            self._module = loky
        elif mode == Scheduler.Mode.THREAD:
            self._executor = futures.ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix='default-scheduler')
            self._module = futures
        self._tasks = queue.Queue()
        self._scheme = scheme
        self._close = False
        self._results = queue.PriorityQueue()
        self._priority_number = 0

    def reset(self):
        """Reset scheduler.

        Clear results and tasks queue.
        """
        with self._results.mutex:
            self._results.queue.clear()
        with self._tasks.mutex:
            self._tasks.queue.clear()
        self._close = False
        self._priority_number = 0

    def close(self):
        """Close a scheduler from accepting new tasks.
        """
        self._close = True

    def shutdown(self):
        """Shut down a scheduler.

        Reset the scheduler and shut down the inner executor.
        """
        self.reset()
        self._executor.shutdown(wait=True)

    def submit(self, func: object, *args: object, **kwargs: object) -> futures.Future:
        """Submit a new task to the scheduler.

        Arguments:
            func: the task function to be submitted to the scheduler. It should be picklable.
            args: the positional arguments passed to `func`.
            kwargs: the keyword arguments passed to `func`.

        Raises:
            Scheduler.ClosedError: Raise when trying to submit task to a closed scheduler.

        Returns:
            A future instance of the pending submitted task.
        """
        if self._close:
            logging.warning('Scheduler is closed for new tasks.')
            raise Scheduler.ClosedError('Scheduler is closed for new tasks.')
        if self._scheme == Scheduler.Scheme.AFTER_PREVIOUS_DONE:
            try:
                prev_task = self._tasks.get(timeout=0.05)
                self._add_to_results(prev_task)
            except queue.Empty:
                pass
            task = self._executor.submit(func, *args, **kwargs)
            self._tasks.put(task)
        else:
            task = self._executor.submit(func, *args, **kwargs)
            self._tasks.put(task)
            task.add_done_callback(self._add_to_results)
        return task

    def _add_to_results(self, task):
        if self._scheme == Scheduler.Scheme.EXECUTION_ORDER:
            self._results.put((self._priority_number, task))
            self._priority_number += 1
        elif self._scheme == Scheduler.Scheme.SUBMIT_ORDER:
            i = self._tasks.queue.index(task)
            self._results.put((i, task))
        elif self._scheme == Scheduler.Scheme.AFTER_PREVIOUS_DONE:
            task.result()
            self._results.put((self._priority_number, task))
            self._priority_number += 1

    def get_all_remaining_results(self) -> list:
        """Get all remaining results from the submitted tasks.

        Returns:
            A list of results of the remaining submitted tasks.
        """
        self._close = True
        self._module.wait(self._tasks.queue)
        results = []
        if self._scheme == Scheduler.Scheme.AFTER_PREVIOUS_DONE:
            while not self._results.empty():
                t = self._results.get()
                results.append(t[1].result())
            t = self._tasks.get()
            results.append(t.result())
        else:
            while not self._results.empty():
                t = self._results.get()
                self._tasks.get()
                results.append(t[1].result())
        return results

    def get_result(self, timeout: float = None) -> object:
        """Get the next result of the submitted tasks.

        Arguments:
            timeout: the time out in seconds to wait for the next results. If it is `None`, wait in infinite time.

        Raises:
            Scheduler.ResultNotAvailableError: Raise when result is not available yet.

        Returns:
            The next result of the submitted tasks.
        """
        try:
            if self._scheme == Scheduler.Scheme.EXECUTION_ORDER:
                task = self._results.get(timeout=timeout)
                self._tasks.get()
                return task[1].result()
            elif self._scheme == Scheduler.Scheme.SUBMIT_ORDER:
                if not self._results.empty():
                    first_task = self._results.queue[0]
                    if self._tasks.queue.index(first_task[1]) != 0:
                        raise queue.Empty
                    else:
                        first_task = self._results.get()
                        self._tasks.get()
                        result = first_task[1].result()
                        return result
                else:
                    raise queue.Empty
            elif self._scheme == Scheduler.Scheme.AFTER_PREVIOUS_DONE:
                if not self._results.empty():
                    task = self._results.get()
                    return task[1].result()
                elif not self._tasks.empty():
                    return self._tasks.get().result()
                else:
                    raise queue.Empty
        except queue.Empty:
            raise Scheduler.ResultNotAvailableError(
                'None of the results are available.')