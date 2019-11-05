import os
from glob import glob


def load_test_data(file_type='mhealth', sensor_type='sensor', file_num='multiple', sr_type='consistent'):
    cwd = os.path.dirname(os.path.realpath(__file__))
    if sensor_type == 'sensor':
        data_folder = os.path.join(cwd, '..', 'data', file_num,
                                   file_type, sensor_type, sr_type + '_sr')
    elif sensor_type == 'annotation':
        data_folder = os.path.join(cwd, '..', 'data', file_num,
                                   file_type, sensor_type)
    meta_file = os.path.join(data_folder, 'meta.csv')
    if file_num == 'multiple':
        data = list(filter(lambda f: 'meta' not in f, glob(
            os.path.join(data_folder, '*.csv'))))
    else:
        data = list(filter(lambda f: 'meta' not in f, glob(
            os.path.join(data_folder, '*.csv'))))[0]
    if sensor_type != 'annotation':
        with open(meta_file, 'r') as meta:
            sr = int(meta.readline().split(',')[1])
    else:
        sr = None
    data = sorted(data)
    return data, sr
