"""
Scheduler using different schemes
============================================================================

"""
# %%
# Imports
# -----------
import os
import arus
import logging
import time


# %%
# Set up test functions
def task1():
    print('task1 start on {}'.format(os.getpid()))
    time.sleep(2)
    print('task1 stop on {}'.format(os.getpid()))
    return 'task1'


def task2():
    print('task2 start on {}'.format(os.getpid()))
    time.sleep(1)
    print('task2 stop on {}'.format(os.getpid()))
    return 'task2'


def task3():
    print('task3 start on {}'.format(os.getpid()))
    time.sleep(1)
    print('task3 stop on {}'.format(os.getpid()))
    return 'task3'


# %%
# Set up schedulers
mode = arus.Scheduler.Mode.PROCESS
scheme = arus.Scheduler.Scheme.EXECUTION_ORDER
execute_scheduler = arus.Scheduler(mode=mode, scheme=scheme, max_workers=3)

mode = arus.Scheduler.Mode.PROCESS
scheme = arus.Scheduler.Scheme.SUBMIT_ORDER
submit_scheduler = arus.Scheduler(mode=mode, scheme=scheme, max_workers=3)

mode = arus.Scheduler.Mode.PROCESS
scheme = arus.Scheduler.Scheme.AFTER_PREVIOUS_DONE
sequential_scheduler = arus.Scheduler(mode=mode, scheme=scheme, max_workers=3)

# %%
# Test no order scheduler
print('Test scheduler with results in execution order')
execute_scheduler.submit(task1)
execute_scheduler.submit(task2)
execute_scheduler.submit(task3)
results = execute_scheduler.get_all_remaining_results()
print(results)
execute_scheduler.reset()

# %%
# Test in order scheduler
print('Test scheduler with results in submit order')
submit_scheduler.submit(task1)
submit_scheduler.submit(task2)
submit_scheduler.submit(task3)
results = submit_scheduler.get_all_remaining_results()
print(results)
submit_scheduler.reset()
# # %%
# # Test sequential scheduler
print('Test scheduler with both execution and results in sequential order')
sequential_scheduler.submit(task1)
sequential_scheduler.submit(task2)
sequential_scheduler.submit(task3)
results = sequential_scheduler.get_all_remaining_results()
sequential_scheduler.reset()
print(results)

# # %%
# # Test get_result on the fly
print('Test scheduler with results in execution order and get results on the fly')
execute_scheduler.submit(task1)
execute_scheduler.submit(task2)
execute_scheduler.submit(task3)
results = []
while True:
    result = execute_scheduler.get_result()
    results.append(result)
    print('get result:' + result)
    if len(results) == 3:
        break
execute_scheduler.reset()

print('Test scheduler with results in submit order and get results on the fly')
submit_scheduler.submit(task1)
submit_scheduler.submit(task2)
submit_scheduler.submit(task3)
results = []
while True:
    try:
        result = submit_scheduler.get_result()
        results.append(result)
        print('get result:' + result)
    except arus.Scheduler.ResultNotAvailableError:
        continue
    if len(results) == 3:
        break
submit_scheduler.reset()

print('Test scheduler with results in sequential order and get results on the fly')
sequential_scheduler.submit(task1)
sequential_scheduler.submit(task2)
sequential_scheduler.submit(task3)
results = []
while True:
    try:
        result = sequential_scheduler.get_result()
        results.append(result)
        print('get result:' + result)
    except arus.Scheduler.ResultNotAvailableError:
        continue
    if len(results) == 3:
        break
sequential_scheduler.reset()
