"""
Stream using annotation generator
=================================
"""

from arus.core.annotation.generator import normal_dist
from arus.core.stream.generator_stream import GeneratorSlidingWindowStream

if __name__ == "__main__":
    data_source = {
        "generator": normal_dist,
        "kwargs": {
            "duration_mu": 5,
            "duration_sigma": 1,
            "num_mu": 3,
            "labels": ['Sitting', 'Standing', 'Lying'],
            "sleep_interval": 0
        }
    }

    stream = GeneratorSlidingWindowStream(data_source, window_size=12.8,
                                          simulate_reality=False, start_time_col=1, stop_time_col=2, name='annotator-generator-stream')
    stream.start()

    n = 5
    for window, st, et, prev_st, prev_et, name in stream.get_iterator():
        print("{}-{}-{}".format(window.iloc[0, 0],
                                window.iloc[-1, 0], '-'.join(window.iloc[:, 3])))
        n -= 1
        if n == 0:
            break
    stream.stop()
