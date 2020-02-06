"""
Demonstration of the usage of arus.plugins.metawear.stream.MetaWearSlidingWindowStream
================================================================================

"""

import arus
import logging
from datetime import datetime

if __name__ == "__main__":
    arus.developer.set_default_logging()
    generator = arus.plugins.metawear.MetaWearAccelDataGenerator(
        "FF:EE:B8:99:0C:64", sr=50, grange=8, max_retries=10, buffer_size=100)
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
