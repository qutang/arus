"""
Stream from metawear device
================================================================================

"""

import arus
from datetime import datetime

if __name__ == "__main__":
    generator = arus.plugins.metawear.MetaWearAccelDataGenerator(
        "F9:DE:3A:BD:B2:84", sr=50, grange=8, max_retries=10, buffer_size=100)
    print(generator.get_device_name())
    segmentor = arus.segmentor.SlidingWindowSegmentor(window_size=4)
    stream = arus.Stream(generator, segmentor, name='metawear-stream')
    stream.start()
    i = 0
    for data, _, _, _, _, _ in stream.generate():
        if data.empty:
            print(data)
            continue
        i = i + 1
        print(data.head())
        if i == 2:
            break
    stream.stop()
