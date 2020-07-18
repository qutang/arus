from .. import mhealth_format as mh
from .. import dataset as ds
from dataclasses import dataclass, field
import typing
from loguru import logger
from .. import error_code


@dataclass
class HARModel:
    mid: str
    name: str = "HAR"
    model: typing.Any = None
    data_set: ds.MHDataset = field(init=False)

    def reset_model(self):
        self.model = None

    def load_dataset(self, data: ds.MHDataset):
        self.data_set = data

    def compute_features(self, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def compute_class_set(self, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def train(self, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def predict(self, *input_objs, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    @staticmethod
    def ignore_classes(fcs, task_name, remove_classes=['Unknown', 'Transition']):
        is_valid_label = ~fcs[task_name].isin(remove_classes).values
        fcs = fcs.loc[is_valid_label, :]
        return fcs
