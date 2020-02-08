import pytest
from .. import scheduler
import os
import time


def task1():
    print('task1 start on {}'.format(os.getpid()))
    time.sleep(2)
    print('task1 stop on {}'.format(os.getpid()))
    return 'task1', time.time(), os.getpid()


def task2():
    print('task2 start on {}'.format(os.getpid()))
    time.sleep(0.5)
    print('task2 stop on {}'.format(os.getpid()))
    return 'task2', time.time(), os.getpid()


def task3():
    print('task3 start on {}'.format(os.getpid()))
    time.sleep(1)
    print('task3 stop on {}'.format(os.getpid()))
    return 'task3', time.time(),  os.getpid()


class TestScheduler:
    @pytest.mark.parametrize('scheme', [scheduler.Scheduler.Scheme.SUBMIT_ORDER, scheduler.Scheduler.Scheme.EXECUTION_ORDER, scheduler.Scheduler.Scheme.AFTER_PREVIOUS_DONE])
    @pytest.mark.parametrize('mode', [scheduler.Scheduler.Mode.THREAD, scheduler.Scheduler.Mode.PROCESS])
    def test_modes_and_schemes_with_get_all_remaining_results(self, scheme, mode):
        sch = scheduler.Scheduler(mode=mode, scheme=scheme, max_workers=5)
        sch.submit(task1)
        sch.submit(task2)
        sch.submit(task3)
        results = sch.get_all_remaining_results()
        if mode == scheduler.Scheduler.Mode.THREAD:
            assert len(set([r[2] for r in results])) == 1
        else:
            assert len(set([r[2] for r in results])) == 3
        if scheme == scheduler.Scheduler.Scheme.EXECUTION_ORDER:
            assert results[0][0] == 'task2'
            assert results[1][0] == 'task3'
            assert results[2][0] == 'task1'
        else:
            assert results[0][0] == 'task1'
            assert results[1][0] == 'task2'
            assert results[2][0] == 'task3'
        if scheme == scheduler.Scheduler.Scheme.AFTER_PREVIOUS_DONE:
            assert results[2][1] > results[1][1]
            assert results[1][1] > results[0][1]
        elif scheme == scheduler.Scheduler.Scheme.EXECUTION_ORDER:
            assert results[2][1] > results[1][1]
            assert results[1][1] > results[0][1]
        else:
            assert results[0][1] > results[2][1]
            assert results[2][1] > results[1][1]
        sch.shutdown()

    @pytest.mark.parametrize('scheme', [scheduler.Scheduler.Scheme.SUBMIT_ORDER, scheduler.Scheduler.Scheme.EXECUTION_ORDER, scheduler.Scheduler.Scheme.AFTER_PREVIOUS_DONE])
    @pytest.mark.parametrize('mode', [scheduler.Scheduler.Mode.THREAD, scheduler.Scheduler.Mode.PROCESS])
    def test_modes_and_schemes_with_get_result(self, scheme, mode):
        sch = scheduler.Scheduler(mode=mode, scheme=scheme, max_workers=5)
        sch.submit(task1)
        sch.submit(task2)
        sch.submit(task3)
        results = []
        while True:
            try:
                result = sch.get_result()
                results.append(result)
            except scheduler.Scheduler.ResultNotAvailableError:
                continue
            if len(results) == 3:
                break
        if mode == scheduler.Scheduler.Mode.THREAD:
            assert len(set([r[2] for r in results])) == 1
        else:
            assert len(set([r[2] for r in results])) == 3
        if scheme == scheduler.Scheduler.Scheme.EXECUTION_ORDER:
            assert results[0][0] == 'task2'
            assert results[1][0] == 'task3'
            assert results[2][0] == 'task1'
        else:
            assert results[0][0] == 'task1'
            assert results[1][0] == 'task2'
            assert results[2][0] == 'task3'
        if scheme == scheduler.Scheduler.Scheme.AFTER_PREVIOUS_DONE:
            assert results[2][1] > results[1][1]
            assert results[1][1] > results[0][1]
        elif scheme == scheduler.Scheduler.Scheme.EXECUTION_ORDER:
            assert results[2][1] > results[1][1]
            assert results[1][1] > results[0][1]
        else:
            assert results[0][1] > results[2][1]
            assert results[2][1] > results[1][1]
        sch.shutdown()
