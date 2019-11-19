from arus.core.stream.generator_stream import GeneratorSlidingWindowStream
from arus.core.accelerometer.generator import normal_dist

if __name__ == "__main__":
    data_source = {'generator': normal_dist, 'kwargs': {
        "grange": 4,
        "start_time": None, 
        "buffer_size": 1800, 
        "sleep_interval": 1, 
        "sigma": 1
    }}
    sr = 80
    stream = GeneratorSlidingWindowStream(data_source, window_size=12.8, sr=80, start_time=None, simulate_reality=False, name='sensor-generator-stream')
    stream.start()

    for window, st, prev_st, name in stream.get_iterator():
        print("{}-{}-{}".format(window.iloc[0,0], window.iloc[-1,0], window.shape))