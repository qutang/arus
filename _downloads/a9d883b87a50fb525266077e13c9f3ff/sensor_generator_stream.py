"""
Demonstration of the usage of arus.core.stream.GeneratorSlidingWindowStream
================================================================================

"""

from arus.core.stream.generator_stream import GeneratorSlidingWindowStream
from arus.core.accelerometer.generator import normal_dist

if __name__ == "__main__":
    data_source = {'generator': normal_dist, 'kwargs': {
        "grange": 4,
        "buffer_size": 1800,
        "sleep_interval": 1,
        "sigma": 1,
        "sr": 80
    }}
    stream = GeneratorSlidingWindowStream(data_source,
                                          window_size=12.8,  simulate_reality=False,
                                          start_time_col=0, stop_time_col=0, name='sensor-generator-stream')
    stream.start()

    n = 5
    for window, st, et, prev_st, prev_et, name in stream.get_iterator():
        print(
            "{}-{}-{}".format(window.iloc[0, 0], window.iloc[-1, 0], window.shape))
        n -= 1
        if n == 0:
            break
    stream.stop()
