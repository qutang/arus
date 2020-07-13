from .. import dataset
from .. import env
import os
import shutil
import pandas as pd
from dataclasses import asdict


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


class TestMHDataset:
    def test_mh_dataset(self):
        dataset_path = dataset.get_dataset_path('spades_lab')
        mh_ds = dataset.MHDataset(
            path=dataset_path, name='spades_lab', input_type=dataset.InputType.MHEALTH_FORMAT)
        assert mh_ds.name == 'spades_lab'
        assert mh_ds.subjects[0].pid == 'SPADES_1'
        assert mh_ds.subjects[0].demography is None
        assert mh_ds.subjects[0].sensors[0].data_type == 'AccelerationCalibrated'
        assert mh_ds.subjects[0].annotations[0].annotation_type == 'SPADESInLab'
        assert len(mh_ds.subjects[0].sensors[0].paths) == 3
