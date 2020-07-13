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
    model: typing.Any = field(init=False)
    data_set: ds.MHDataset = field(init=False)

    def reset_model(self):
        self.model = None

    def load_dataset(self, data: ds.MHDataset):
        self.data_set = data

    def compute_features(self, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def compute_classes(self, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')

    def train(self, **kwargs):
        raise NotImplementedError(
            'This method must be implemented by sub classes!')
