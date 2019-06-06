from arus.core.stream import MhealthFileStream
from arus.core.pipeline import Pipeline
from glob import glob
from pathos.multiprocessing import ProcessPool
import pandas as pd


def processor(stream):
    stream.start()
    all_data = []
    for data in stream.get_iterator():
        all_data.append(data.iloc[:, 1:4].mean(axis=0))
    return pd.concat(all_data, axis=1).transpose()


if __name__ == "__main__":
    stream1_files = glob(
        '/mnt/d/data/spades-2day/SPADES_2/MasterSynced/**/*TAS1E23150075*.sensor.csv*', recursive=True)
    stream2_files = glob(
        '/mnt/d/data/spades-2day/SPADES_2/MasterSynced/**/*TAS1E23150084*.sensor.csv*', recursive=True)
    stream1 = MhealthFileStream(
        data_source=stream1_files, sr=80, name='TAS1E23150075')
    stream2 = MhealthFileStream(
        data_source=stream2_files, sr=80, name='TAS1E23150084')
    pipeline = Pipeline(name='mean-of-each-column')
    pipeline.add_streams(stream1, stream2)
    pipeline.start()
    for result in pipeline.get_iterator():
        print(result)
