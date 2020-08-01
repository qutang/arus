from .. import dataset as ds
from .. import models
import numpy as np
import uuid
from sklearn.pipeline import Pipeline
import pytest
import shutil


class TestMUSSHARModel:
    @pytest.mark.parametrize('placements', [['DW'], ['DW', 'DA']])
    @pytest.mark.parametrize('task_name', ['ACTIVITY_VALIDATED', 'POSTURE_VALIDATED', 'INTENSITY'])
    def test_mussharmodel(self, placements, task_name):
        from .. import spades_lab as slab
        spades_lab = ds.MHDataset(path=ds.get_dataset_path('spades_lab'),
                                  name='spades_lab', input_type=ds.InputType.MHEALTH_FORMAT)
        spades_1_ds = spades_lab.subset(
            name='spades_lab_spades_1', pids=['SPADES_1'])

        spades_1_ds.set_class_set_parser(slab.class_set)
        spades_1_ds.set_placement_parser(slab.get_sensor_placement)

        model = models.MUSSHARModel(
            mid=str(uuid.uuid4()), used_placements=placements, window_size=12.8, sr=80)

        model.load_dataset(spades_1_ds)

        model.compute_features()
        model.compute_class_set(
            task_names=['ACTIVITY_VALIDATED', 'POSTURE_VALIDATED', 'INTENSITY'])

        model.train(task_name=task_name, verbose=True)

        # assert len(model.get_feature_names()) == len(
        #     placements)*len(feat.preset_names())
        assert type(model.model) == Pipeline
        assert model.train_perf['acc'] > 0.9

        shutil.rmtree(model.get_processed_path(), ignore_errors=True)
