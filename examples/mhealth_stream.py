from arus.core.stream import MhealthFileStream
from glob import glob
import os

if __name__ == "__main__":
    files = sorted(glob(os.path.expanduser(
        '~/Projects/data/MDCAS-model/Test/SPADES_2/MasterSynced/2015/10/06/**/*.sensor.csv'), recursive=True))
    stream = MhealthFileStream(
        data_source=files, chunk_size=12.8, start_time=None, sr=80, name='spades_2')
    stream.start(scheduler='thread')
    for data in stream.get_iterator():
        print("{} - {}".format(data.iloc[0, 0], data.iloc[-1, 0]))
