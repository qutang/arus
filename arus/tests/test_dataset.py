from .. import dataset
from .. import env
import os
import shutil
import pandas as pd


class TestSpadesLab:
    def test_get_dataset_path(self):
        dataset_path = dataset.get_dataset_path('spades_lab')
        assert dataset_path == os.path.join(
            env.get_data_home(), 'spades_lab')
        assert os.path.exists(dataset_path)

    def test_load_dataset(self):
        dataset_dict = dataset.load_dataset('spades_lab')
        assert type(dataset_dict) == dict
        assert 'subjects' in dataset_dict.keys()
        assert 'processed' in dataset_dict.keys()
