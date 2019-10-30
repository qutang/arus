import os
from glob import glob


def load_test_data(file_type='mhealth', file_num='multiple', sr_type='consistent'):
    cwd = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(cwd, '..', 'data', file_num,
                               file_type, sr_type + '_sr')
    meta_file = os.path.join(data_folder, 'meta.csv')
    if file_num == 'multiple':
        data = list(filter(lambda f: 'meta' not in f, glob(
            os.path.join(data_folder, '*.csv'))))
    else:
        data = list(filter(lambda f: 'meta' not in f, glob(
            os.path.join(data_folder, '*.csv'))))[0]
    with open(meta_file, 'r') as meta:
        sr = int(meta.readline().split(',')[1])
    return data, sr
