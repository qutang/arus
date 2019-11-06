import os
from glob import glob


def load_test_data(file_type='mhealth', sensor_type='sensor', file_num='multiple', exception_type='consistent_sr'):
    cwd = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(cwd, '..', 'data', file_num,
                                file_type, sensor_type, exception_type)
    meta_file = os.path.join(data_folder, 'meta.csv')
    if file_num == 'multiple':
        data = list(filter(lambda f: 'meta' not in f, glob(
            os.path.join(data_folder, '*.csv'))))
        data = sorted(data)
    else:
        data = list(filter(lambda f: 'meta' not in f, glob(
            os.path.join(data_folder, '*.csv'))))[0]
    if sensor_type != 'annotation':
        with open(meta_file, 'r') as meta:
            sr = int(meta.readline().split(',')[1])
    else:
        sr = None
    return data, sr
