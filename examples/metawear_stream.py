from arus.plugins.metawear.stream import MetaWearSlidingWindowStream
import logging
from datetime import datetime

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format='[%(levelname)s]%(asctime)s <P%(process)d-%(threadName)s> %(message)s')
    stream = MetaWearSlidingWindowStream("FF:EE:B8:99:0C:64", sr=50, grange=8,
                                         window_size=4, start_time=datetime.now(), max_retries=10, name='metawear-stream')
    stream.start()
    i = 0
    for data, _, _, _, _, _ in stream.get_iterator():
        if data.empty:
            print(data)
            continue
        i = i + 1
        print(data.head())
        if i == 2:
            break
    stream.stop()
