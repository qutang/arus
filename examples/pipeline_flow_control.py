from arus.core.pipeline import Pipeline
from arus.core.stream.generator_stream import GeneratorSlidingWindowStream
from arus.core.accelerometer import generator
from datetime import datetime
import pandas as pd
import time
import logging


def _pipeline_test_processor(chunk_list, **kwargs):
    import pandas as pd
    result = {'NAME': [],
              'START_TIME': [], 'STOP_TIME': []}
    for data, st, et, prev_st, prev_et, name in chunk_list:
        result['NAME'].append(name)
        result['START_TIME'].append(data.iloc[0, 0])
        result['STOP_TIME'].append(data.iloc[-1, 0])
    result = pd.DataFrame.from_dict(result)
    return result


if __name__ == "__main__":
    # test on a single stream
    stream1_config = {
        "generator": generator.normal_dist,
        'kwargs': {
            "grange": 8,
            "buffer_size": 100,
            "sleep_interval": 0,
            "sigma": 1,
            "sr": 80
        }
    }

    window_size = 2
    start_time = datetime.now()
    stream1 = GeneratorSlidingWindowStream(
        stream1_config, window_size=window_size, simulate_reality=True, start_time_col=0, stop_time_col=0, name='stream-1')

    pipeline = Pipeline(max_processes=2, scheduler='processes')
    pipeline.add_stream(stream1)
    pipeline.set_processor(_pipeline_test_processor)

    # connect, there will be no incoming data, get_iteratnor will be blocking
    pipeline.connect()
    results = []
    count_none = 0
    for result, st, et, prev_st, prev_et, name in pipeline.get_iterator(timeout=0.1):
        if result is not None:
            print('Connect is not working')
            assert False
        else:
            count_none = count_none + 1
        if count_none == 50:
            break
    print('Connected yet no data is coming')
    pipeline.stop()

    # connect, wait for 3 seconds and then start processing, get_iterator will be blocking for 3 + 4 seconds until receiving the first window
    pipeline.connect()
    connect_time = pd.Timestamp(datetime.now())
    print('Connect time: ' + str(connect_time))
    count_none = 0
    for result, st, et, prev_st, prev_et, name in pipeline.get_iterator(timeout=0.1):
        if result is not None:
            first_block_time = pd.Timestamp(datetime.now())
            print('First block time: ' + str(first_block_time) + ', ' +
                  str(first_block_time.timestamp() - connect_time.timestamp()) + ' seconds since connection.')
            print('First block st: ' + str(st))
            break
        else:
            count_none = count_none + 1
        if count_none == 30:
            process_time = pd.Timestamp(datetime.now())
            pipeline.process(start_time=process_time)
            print('Process time: ' + str(process_time) + ', ' +
                  str(process_time.timestamp() - connect_time.timestamp()) + ' seconds since connection.')
    print('Stop')
    pipeline.stop()

    # connect and process for 4 seconds and then pause for 2 seconds and then process again for 4 seconds
    st = datetime.now()
    pipeline.start(process_start_time=st)
    print('Start at: ' + str(st))
    count = 0
    count_none = 0
    restarted = False
    stt = time.time() + 100000
    for result, st, et, prev_st, prev_et, name in pipeline.get_iterator(timeout=0.1):
        if result is not None:
            count = count + 1
            print('Blocks at: ' + str(st))
            if count == 2:
                print('Pause')
                pipeline.pause()
                stt = time.time()
            if count == 4:
                break
        if time.time() - stt >= 2 and not restarted:
            restarted = True
            ts = datetime.now()
            process_time = pd.Timestamp(ts)
            pipeline.process(start_time=process_time)
            print('Start again at:' + str(ts))

    print('Stop')
    pipeline.stop()
