from arus.core.stream import MhealthFileStream
from glob import glob

if __name__ == "__main__":
    files = glob(
        '/mnt/d/data/MDCAS/DINESH_00/MasterSynced/**/*.sensor.csv*', recursive=True)
    stream = MhealthFileStream(data_source=files, sr=80, name='dinesh_00')
    stream.start(scheduler='thread')
    for data in stream.get_iterator():
        print(data.iloc[:, 1].mean())
