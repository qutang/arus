from .. import dataset as ds
from .. import models
import uuid


class TestMUSSHARModel:
    def test_mussharmodel(self):
        spades_lab = ds.MHDataset(path=ds.get_dataset_path('spades_lab'),
                                  name='spades_lab', input_type=ds.InputType.MHEALTH_FORMAT)
        spades_1_ds = spades_lab.subset(
            name='spades_lab_spades_1', pids=['SPADES_1'])

        model = models.MUSSHARModel(
            mid=uuid.uuid4(), used_placements=['DW', 'DA'])
        model.load_dataset(spades_1_ds)
        model.compute_features(window_size=12.8, sr=80)
        model.train(verbose=True)
        assert model.model is None
        assert model.scaler is None
